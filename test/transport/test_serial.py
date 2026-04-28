"""Test the serialx-backed serial transport wiring.

End-to-end serial round-trips run in ``test_comm.py`` via serialx's
``socket://`` backend.  These tests focus on the small bit of
pymodbus-specific glue that used to live in ``serialtransport.py``:
parameter translation (``parity`` strings → ``serialx.Parity`` enum,
``bytesize`` → ``byte_size``), failure paths that bubble up from
serialx, and a real PTY round-trip to prove the configuration also
works against a hardware-style backend, not just the socket adapter.
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest import mock

import pytest
import serialx

from pymodbus.transport import CommParams, CommType, ModbusProtocol
from pymodbus.transport.transport import _to_parity


def _comm_params(host: str, *, parity: str = "N") -> CommParams:
    return CommParams(
        comm_name="test-serial",
        comm_type=CommType.SERIAL,
        reconnect_delay=0,
        reconnect_delay_max=0,
        timeout_connect=2.0,
        host=host,
        port=0,
        baudrate=9600,
        bytesize=8,
        parity=parity,
        stopbits=1,
    )


class _CollectingProtocol(ModbusProtocol):
    """Concrete ModbusProtocol that records received bytes."""

    def __init__(self, params: CommParams, is_server: bool = False) -> None:
        super().__init__(params, is_server)
        self.received = bytearray()

    def callback_new_connection(self) -> ModbusProtocol:
        return _CollectingProtocol(params=self.comm_params, is_server=False)

    def callback_connected(self) -> None:
        pass

    def callback_disconnected(self, exc: Exception | None) -> None:
        pass

    def callback_data(self, data: bytes, addr: tuple | None = None) -> int:
        self.received.extend(data)
        return len(data)


class TestParityHelper:
    """Cover the small parity translation helper used by the transport."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("N", serialx.Parity.NONE),
            ("n", serialx.Parity.NONE),
            ("O", serialx.Parity.ODD),
            ("E", serialx.Parity.EVEN),
            ("M", serialx.Parity.MARK),
            ("S", serialx.Parity.SPACE),
            ("", serialx.Parity.NONE),
            ("?", serialx.Parity.NONE),
        ],
    )
    def test_translation(self, value, expected):
        assert _to_parity(value) is expected

    def test_passthrough_enum(self):
        assert _to_parity(serialx.Parity.ODD) is serialx.Parity.ODD

    def test_none_falls_back_to_parity_none(self):
        assert _to_parity(None) is serialx.Parity.NONE


class TestTransportWiring:
    """Make sure ModbusProtocol forwards its serial params into serialx."""

    async def test_connect_passes_translated_kwargs(self):
        """parity='E' ⇒ Parity.EVEN, bytesize ⇒ byte_size, no `timeout=`."""
        params = _comm_params("/dev/ttyDOES_NOT_EXIST_test_serial", parity="E")
        with mock.patch(
            "pymodbus.transport.transport.serialx.create_serial_connection",
            new=mock.AsyncMock(side_effect=FileNotFoundError("no device")),
        ) as call:
            client = _CollectingProtocol(params)
            assert not await client.connect()
        assert call.await_count == 1
        kwargs = call.await_args.kwargs
        assert kwargs["parity"] is serialx.Parity.EVEN
        assert kwargs["byte_size"] == 8
        assert kwargs["baudrate"] == 9600
        assert kwargs["stopbits"] == 1
        assert "timeout" not in kwargs
        assert "bytesize" not in kwargs

    async def test_connect_missing_device_returns_false(self):
        """FileNotFoundError from serialx is absorbed by ModbusProtocol.connect."""
        client = _CollectingProtocol(
            _comm_params("/dev/tty007pymodbus_does_not_exist")
        )
        assert not await client.connect()
        assert client.transport is None

    async def test_unknown_uri_scheme_propagates(self):
        """A bogus URI scheme raises serialx.SerialException at connect time.

        ModbusProtocol.connect() only swallows OSError/TimeoutError,
        so unknown-scheme errors must remain visible to the caller.
        """
        from serialx.common import UnknownUriScheme

        client = _CollectingProtocol(_comm_params("loop://"))
        with pytest.raises(UnknownUriScheme):
            await client.connect()


@pytest.mark.skipif(
    sys.platform == "win32" or not hasattr(os, "openpty"),
    reason="pty round-trip requires a POSIX system",
)
class TestPtyRoundTrip:
    """Exercise the real serialx hardware backend over a pseudo-terminal."""

    async def test_round_trip(self):
        master_fd, slave_fd = os.openpty()
        slave_path = os.ttyname(slave_fd)
        os.close(slave_fd)
        try:
            client = _CollectingProtocol(_comm_params(slave_path))
            assert await client.connect()
            assert client.transport is not None

            outgoing = b"\x01\x03\x00\x00\x00\x01\x84\x0a"
            client.send(outgoing)
            received = await asyncio.get_running_loop().run_in_executor(
                None, _read_exact, master_fd, len(outgoing)
            )
            assert received == outgoing

            incoming = b"\x01\x03\x02\x00\x00\xb8\x44"
            os.write(master_fd, incoming)
            for _ in range(200):
                if bytes(client.received) == incoming:
                    break
                await asyncio.sleep(0.01)
            assert bytes(client.received) == incoming

            client.close()
        finally:
            os.close(master_fd)


def _read_exact(fd: int, n: int) -> bytes:
    """Block until ``n`` bytes have been read from ``fd``."""
    buf = bytearray()
    while len(buf) < n:
        chunk = os.read(fd, n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)
