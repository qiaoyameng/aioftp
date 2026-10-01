import asyncio
import ipaddress
from pathlib import PurePosixPath

import pytest

import aioftp


@pytest.mark.asyncio
async def test_active_upload_download_and_list(pair_factory, Client):
    async with pair_factory(client=Client(data_connection_mode="active")) as pair:
        await pair.client.make_directory("directory")
        async with pair.client.upload_stream("file") as stream:
            await stream.write(b"active-data")
        async with pair.client.download_stream("file") as stream:
            assert await stream.read() == b"active-data"
        paths = await pair.client.list()
        assert PurePosixPath("file") in {path for path, _ in paths}
        assert pair.client._active_data_server is None
        connection = next(iter(pair.server.connections.values()))
        assert not connection.future.data_endpoint.done()


@pytest.mark.asyncio
async def test_active_port_ipv4(pair_factory, Client):
    client = Client(data_connection_mode="active", active_commands=("port",))
    async with pair_factory(client=client) as pair:
        if pair.host != "127.0.0.1":
            return
        async with pair.client.upload_stream("file") as stream:
            await stream.write(b"port-data")
        async with pair.client.download_stream("file") as stream:
            assert await stream.read() == b"port-data"


@pytest.mark.asyncio
async def test_active_connection_rejected(pair_factory, Client, Server):
    server = Server(wait_future_timeout=0.1)
    async with pair_factory(None, server, client=Client(data_connection_mode="active")) as pair:
        active_connection = await pair.client.get_active_connection()
        await pair.client._close_active_data_server()
        transfer = asyncio.create_task(pair.client.get_stream("LIST", "1xx"))
        with pytest.raises(aioftp.StatusCodeError) as exc_info:
            await transfer
        assert exc_info.value.received_codes[-1].matches("425")
        assert pair.client._active_data_server is None
        await pair.client.command("PWD", "257")
        connection = next(iter(pair.server.connections.values()))
        assert not connection.future.data_endpoint.done()
        assert not connection.future.data_connection.done()
        assert active_connection.server.is_serving() is False


@pytest.mark.asyncio
async def test_active_command_cancellation(pair_factory, Client):
    async with pair_factory(client=Client(data_connection_mode="active")) as pair:
        active_connection = await pair.client.get_active_connection()
        transfer = asyncio.create_task(pair.client.get_stream("LIST", "1xx"))
        connection = next(iter(pair.server.connections.values()))
        for _ in range(100):
            if connection.extra_workers:
                break
            await asyncio.sleep(0.001)
        assert connection.extra_workers
        transfer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await transfer
        await pair.client.command("PWD", "257")
        assert pair.client._active_data_server is None
        assert active_connection.server.is_serving() is False


@pytest.mark.asyncio
async def test_continuous_active_and_passive_switching(pair_factory, Client):
    async with pair_factory(client=Client(data_connection_mode="active")) as pair:
        connection = next(iter(pair.server.connections.values()))

        await pair.client.list()
        assert not connection.future.passive_server.done()
        assert not connection.future.data_endpoint.done()
        assert pair.client._active_data_server is None

        pair.client.data_connection_mode = "passive"
        await pair.client.list()
        assert connection.future.passive_server.done()
        assert not connection.future.data_endpoint.done()
        assert pair.client._active_data_server is None

        pair.client.data_connection_mode = "active"
        await pair.client.list()
        assert not connection.future.passive_server.done()
        assert not connection.future.data_endpoint.done()
        assert pair.client._active_data_server is None

        pair.client.data_connection_mode = "passive"
        await pair.client.list()
        assert connection.future.passive_server.done()
        assert not connection.future.data_endpoint.done()
        assert pair.client._active_data_server is None


def test_port_and_eprt_encoding():
    assert aioftp.Client.format_port_command("127.0.0.1", 1025) == "PORT 127,0,0,1,4,1"
    assert aioftp.Client.format_eprt_command("127.0.0.1", 1025) == "EPRT |1|127.0.0.1|1025|"
    assert aioftp.Client.format_eprt_command("::1", 1025) == "EPRT |2|::1|1025|"
    assert aioftp.Client.parse_port_command("127,0,0,1,4,1") == (ipaddress.IPv4Address("127.0.0.1"), 1025)
    assert aioftp.Client.parse_eprt_command("|2|::1|1025|") == (ipaddress.IPv6Address("::1"), 1025)


@pytest.mark.asyncio
async def test_server_rejects_dangerous_ports(pair_factory):
    async with pair_factory() as pair:
        connection = next(iter(pair.server.connections.values()))
        for port in (21, 23, 1023):
            command = aioftp.Client.format_eprt_command(pair.host, port)
            code, _ = await pair.client.command(command, "501")
            assert code == "501"
        assert not connection.future.data_endpoint.done()


@pytest.mark.asyncio
async def test_server_rejects_other_peer_address(pair_factory):
    async with pair_factory() as pair:
        foreign_host = "127.0.0.2" if pair.host == "127.0.0.1" else "::2"
        command = aioftp.Client.format_eprt_command(foreign_host, 2000)
        code, _ = await pair.client.command(command, "501")
        assert code == "501"
        connection = next(iter(pair.server.connections.values()))
        assert not connection.future.data_endpoint.done()
        await pair.client.command("PWD", "257")


@pytest.mark.asyncio
async def test_server_active_endpoint_validation():
    server = aioftp.Server(path_io_factory=aioftp.MemoryPathIO)
    connection = aioftp.Connection(client_host="127.0.0.1", server_port=2121)
    server._validate_active_endpoint(
        connection,
        ipaddress.IPv4Address("127.0.0.1"),
        1024,
    )
    for port in (0, 21, 1023, 2121, 65536):
        with pytest.raises(ValueError):
            server._validate_active_endpoint(
                connection,
                ipaddress.IPv4Address("127.0.0.1"),
                port,
            )
    with pytest.raises(ValueError):
        server._validate_active_endpoint(
            connection,
            ipaddress.IPv4Address("127.0.0.2"),
            2000,
        )

    ipv6_connection = aioftp.Connection(client_host="::1", server_port=0)
    server._validate_active_endpoint(
        ipv6_connection,
        ipaddress.IPv6Address("::1"),
        1024,
    )
    with pytest.raises(ValueError):
        server._validate_active_endpoint(
            ipv6_connection,
            ipaddress.IPv6Address("::2"),
            2000,
        )


def test_server_port_command_parsing():
    assert aioftp.Server.parse_port_command("127,0,0,1,4,1") == (
        ipaddress.IPv4Address("127.0.0.1"),
        1025,
    )
    assert aioftp.Server.parse_eprt_command("|2|::1|1025|") == (
        ipaddress.IPv6Address("::1"),
        1025,
    )
    for command in ("1,2,3,4,5", "|3|::1|1025|", "|1|127.0.0.1|"):
        with pytest.raises(ValueError):
            aioftp.Server.parse_eprt_command(command)
    with pytest.raises(ValueError):
        aioftp.Server.parse_port_command("256,0,0,1,4,1")
