import asyncio
from pathlib import Path

import pytest

import aioftp


@pytest.mark.parametrize(
    ("host", "command"),
    [
        ("127.0.0.1", "PORT 127,0,0,1,4,210"),
        ("::1", "EPRT |2|::1|1234|"),
    ],
)
def test_active_command_encoding(host, command):
    if host.startswith("127"):
        assert aioftp.Client.format_port_command(host, 1234) == command
    else:
        assert aioftp.Client.format_eprt_command(host, 1234) == command


@pytest.mark.parametrize(
    ("value", "endpoint"),
    [
        ("127,0,0,1,19,138", ("127.0.0.1", 5002)),
        ("127,0,0,1,0,80", ("127.0.0.1", 80)),
        ("127,0,0,1,256,0", None),
        ("127,0,0,1,0", None),
    ],
)
def test_parse_port_endpoint(value, endpoint):
    assert aioftp.Server._parse_port_endpoint(value) == endpoint


@pytest.mark.parametrize(
    ("value", "endpoint"),
    [
        ("|1|127.0.0.1|5002|", ("127.0.0.1", 5002)),
        ("|2|::1|5002|", ("::1", 5002)),
        ("|1|127.0.0.1|65536|", None),
        ("|1|127.0.0.1|-1|", None),
        ("|1|::1|5002|", None),
        ("|2|127.0.0.1|5002|", None),
        ("|3|127.0.0.1|5002|", None),
    ],
)
def test_parse_eprt_endpoint(value, endpoint):
    assert aioftp.Server._parse_eprt_endpoint(value) == endpoint


@pytest.mark.asyncio
async def test_active_list_upload_download(pair_factory, Client):
    async with pair_factory(client=Client(data_connection_mode="active")) as pair:
        await pair.make_server_files("/directory/file.txt")
        listing = await pair.client.list()
        assert {path.name for path, _ in listing} == {"directory", "file.txt"}

        await pair.make_client_files("/upload.txt")
        await pair.client.upload("/upload.txt", "/uploaded.txt", write_into=True)
        await pair.client.download("/uploaded.txt", "/downloaded.txt", write_into=True)

        async with pair.client.path_io.open(Path("/downloaded.txt"), "rb") as source:
            uploaded = await source.read()
        assert uploaded == b"-" * aioftp.DEFAULT_BLOCK_SIZE * 3


@pytest.mark.asyncio
async def test_active_port_ipv4(pair_factory, Client):
    async with pair_factory(
        host="127.0.0.1",
        client=Client(data_connection_mode="active", active_commands=("port",)),
    ) as pair:
        await pair.make_server_files("/file.txt")
        await pair.client.download("/file.txt", "/downloaded.txt", write_into=True)


@pytest.mark.asyncio
async def test_active_connection_refused_and_control_recovery(
    pair_factory,
    Client,
    unused_tcp_port,
):
    async with pair_factory(
        client=Client(data_connection_mode="active"),
        do_quit=False,
    ) as pair:
        await pair.make_server_files("/file.txt")
        endpoint = pair.client.format_eprt_command(pair.host, unused_tcp_port)
        await pair.client.command(endpoint, "200")
        with pytest.raises(aioftp.StatusCodeError) as exc_info:
            await pair.client.command("RETR /file.txt", "150")
        assert set(exc_info.value.received_codes) == {"425"}
        assert await pair.client.get_current_directory() == Path("/")


@pytest.mark.asyncio
async def test_active_rejects_foreign_address_and_dangerous_ports(
    pair_factory,
    Client,
    expect_codes_in_exception,
):
    async with pair_factory(client=Client(data_connection_mode="active")) as pair:
        if ":" in pair.host:
            foreign_endpoint = "|2|2001:db8::1|5000|"
        else:
            foreign_endpoint = "|1|127.0.0.2|5000|"
        with expect_codes_in_exception("501"):
            await pair.client.command("EPRT " + foreign_endpoint, "200")

        dangerous_port = pair.client.format_eprt_command(pair.host, 1023)
        with expect_codes_in_exception("501"):
            await pair.client.command(dangerous_port, "200")


@pytest.mark.asyncio
async def test_active_command_cancellation_releases_pending_connection(pair_factory, Client):
    async with pair_factory(
        client=Client(data_connection_mode="active"),
    ) as pair:
        await pair.make_server_files("file.txt", size=1)
        active_server, _, active_command, _ = await pair.client._start_active_data_server("eprt", "I")
        await pair.client.command(active_command, "200")
        transfer = asyncio.create_task(pair.client.command("RETR file.txt", "150"))
        connection_state = next(iter(pair.server.connections.values()))
        for _ in range(100):
            if "active_open_task" in connection_state:
                break
            await asyncio.sleep(0)
        await asyncio.sleep(0.01)
        transfer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await transfer
        await pair.client.abort()
        assert await pair.client.get_current_directory() == Path("/")
        assert "active_open_task" not in connection_state
        assert "active_host" not in connection_state
        await pair.client._close_active_data_server()
        assert active_server.is_closed()


@pytest.mark.asyncio
async def test_active_stream_cancellation_releases_listener(pair_factory, Client, Server):
    class NoDataServer(Server):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.retr_started = asyncio.Event()

        async def retr(self, connection, rest):
            self.retr_started.set()
            pending = asyncio.create_task(asyncio.Event().wait())
            connection.extra_workers.add(pending)
            try:
                await pending
            except asyncio.CancelledError:
                connection.response("426", "transfer aborted")
                connection.response("226", "abort successful")
                return True

    async with pair_factory(
        client=Client(data_connection_mode="active"),
        server_factory=NoDataServer,
    ) as pair:
        transfer = asyncio.create_task(pair.client.download_stream("file.txt"))
        await pair.server.retr_started.wait()
        await asyncio.sleep(0.01)
        transfer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await transfer
        assert pair.client._active_data_server is None
        await pair.client.abort()
        assert await pair.client.get_current_directory() == Path("/")


@pytest.mark.asyncio
async def test_repeated_active_passive_switching(pair_factory, Client):
    async with pair_factory(host="127.0.0.1") as pair:
        connection_state = next(iter(pair.server.connections.values()))

        active_server, _, active_command, _ = await pair.client._start_active_data_server(
            "eprt",
            "I",
        )
        await pair.client.command(active_command, "200")
        assert active_server.is_serving()

        await pair.client.command("EPSV", "229")
        assert pair.client._active_data_server is None
        assert "active_host" not in connection_state
        assert "passive_server" in connection_state

        active_server, _, active_command, _ = await pair.client._start_active_data_server(
            "eprt",
            "I",
        )
        await pair.client.command(active_command, "200")
        assert "passive_server" not in connection_state
        assert "active_host" in connection_state
        pair.client.data_connection_mode = "active"

        listing = await pair.client.list()
        assert listing == []
        assert pair.client._active_data_server is None
        assert "active_host" not in connection_state


@pytest.mark.asyncio
async def test_server_switch_cancels_pending_active_connection(pair_factory):
    async with pair_factory(host="127.0.0.1") as pair:
        connection = next(iter(pair.server.connections.values()))
        pending = asyncio.create_task(asyncio.sleep(3600))
        connection.active_host = "127.0.0.1"
        connection.active_port = 5000
        connection.active_open_task = pending
        connection.data_connection_prepared = True

        await pair.server._switch_to_passive(connection)

        assert pending.cancelled()
        assert "active_host" not in connection
        assert "active_port" not in connection
        assert "active_open_task" not in connection
        assert "data_connection_prepared" not in connection
