import asyncio
import socket
import ssl
import sys

import pytest
import trustme

import aioftp

if sys.version_info < (3, 11):
    pytest.skip(reason="required python 3.11+", allow_module_level=True)


@pytest.fixture
def tls_contexts():
    ca = trustme.CA()
    server_cert = ca.issue_server_cert("127.0.0.1", "::1")
    server_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    server_cert.configure_cert(server_context)
    client_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ca.configure_trust(client_context)
    untrusted_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    return server_context, client_context, untrusted_context


async def read_response(stream):
    return (await stream.readline()).decode().rstrip()


def simple_response(code, message):
    async def command(connection, rest):
        connection.response(code, message)
        return True

    return command


AUTH_RESPONSE = simple_response("234", ":P")
PBSZ_RESPONSE = simple_response("200", ":P")
PROT_RESPONSE = simple_response("200", ":P")
OK_RESPONSE = simple_response("200", ":P")


@pytest.mark.asyncio
async def test_upgrade_to_tls(mocker, pair_factory):
    ssl_context = object()
    create_default_context = mocker.patch("aioftp.client.ssl.create_default_context", return_value=ssl_context)

    async with pair_factory(logged=False, do_quit=False) as pair:
        pair.server.commands_mapping["auth"] = AUTH_RESPONSE
        pair.server.commands_mapping["pbsz"] = PBSZ_RESPONSE
        pair.server.commands_mapping["prot"] = PROT_RESPONSE

        start_tls = mocker.patch.object(pair.client.stream, "start_tls")
        command_spy = mocker.spy(pair.client, "command")

        await pair.client.upgrade_to_tls()

    create_default_context.assert_called_once_with()
    start_tls.assert_called_once_with(sslcontext=ssl_context, server_hostname=pair.client.server_host)
    assert command_spy.call_count == 3
    command_spy.assert_has_calls(
        [
            mocker.call("AUTH TLS", "234"),
            mocker.call("PBSZ 0", "200"),
            mocker.call("PROT P", "200"),
        ],
    )


@pytest.mark.asyncio
async def test_upgrade_to_tls_custom_ssl_context(mocker, pair_factory):
    ssl_context = object()
    create_default_context = mocker.patch("aioftp.client.ssl.create_default_context")

    async with pair_factory(logged=False, do_quit=False) as pair:
        pair.server.commands_mapping["auth"] = AUTH_RESPONSE
        pair.server.commands_mapping["pbsz"] = PBSZ_RESPONSE
        pair.server.commands_mapping["prot"] = PROT_RESPONSE

        start_tls = mocker.patch.object(pair.client.stream, "start_tls")
        command_spy = mocker.spy(pair.client, "command")

        await pair.client.upgrade_to_tls(sslcontext=ssl_context)

    create_default_context.assert_not_called()
    start_tls.assert_called_once_with(sslcontext=ssl_context, server_hostname=pair.client.server_host)
    assert command_spy.call_count == 3
    command_spy.assert_has_calls(
        [
            mocker.call("AUTH TLS", "234"),
            mocker.call("PBSZ 0", "200"),
            mocker.call("PROT P", "200"),
        ],
    )


@pytest.mark.asyncio
async def test_upgrade_to_tls_when_logged_in(mocker, pair_factory):
    mocker.patch("aioftp.client.ssl.create_default_context")

    async with pair_factory(logged=False, do_quit=False) as pair:
        pair.server.commands_mapping["auth"] = AUTH_RESPONSE
        pair.server.commands_mapping["pbsz"] = OK_RESPONSE
        pair.server.commands_mapping["prot"] = OK_RESPONSE

        mocker.patch.object(pair.client.stream, "start_tls")
        command_spy = mocker.spy(pair.client, "command")

        await pair.client.upgrade_to_tls()

    command_spy.assert_has_calls(
        [
            mocker.call("AUTH TLS", "234"),
            mocker.call("PBSZ 0", "200"),
            mocker.call("PROT P", "200"),
        ],
    )


@pytest.mark.asyncio
async def test_explicit_tls_login_and_passive_data_connection(pair_factory, Server, tls_contexts):
    server_context, client_context, _ = tls_contexts

    async with pair_factory(
        None,
        Server(ssl_context=server_context),
        logged=False,
        do_quit=False,
    ) as pair:
        await pair.client.upgrade_to_tls(client_context)
        await pair.client.login()

        _, writer = await pair.client.get_passive_connection()
        assert writer.transport.get_extra_info("ssl_object") is not None

        await pair.client.prot_c()
        _, clear_writer = await pair.client.get_passive_connection()
        assert clear_writer.transport.get_extra_info("ssl_object") is None

        await pair.client.prot_p()
        _, protected_writer = await pair.client.get_passive_connection()
        assert protected_writer.transport.get_extra_info("ssl_object") is not None

        writer.close()
        clear_writer.close()
        protected_writer.close()
        await pair.client.quit()


@pytest.mark.asyncio
async def test_explicit_tls_certificate_validation_failure(pair_factory, Server, tls_contexts):
    server_context, _, untrusted_context = tls_contexts

    async with pair_factory(
        None,
        Server(ssl_context=server_context),
        logged=False,
        do_quit=False,
    ) as pair:
        before_upgrade = set(pair.server.connections.values())
        with pytest.raises(ssl.SSLCertVerificationError):
            await pair.client.upgrade_to_tls(untrusted_context)

        if before_upgrade:
            await asyncio.wait_for(before_upgrade.pop()._dispatcher, timeout=1)
        assert pair.server.connections == {}
        pair.client.close()


@pytest.mark.asyncio
async def test_duplicate_auth_rejected_after_upgrade(pair_factory, Server, tls_contexts):
    server_context, client_context, _ = tls_contexts

    async with pair_factory(
        None,
        Server(ssl_context=server_context),
        logged=False,
        do_quit=False,
    ) as pair:
        await pair.client.upgrade_to_tls(client_context)

        code, _ = await pair.client.command("AUTH TLS", "503")
        assert code == "503"


@pytest.mark.asyncio
async def test_require_tls_rejects_plain_login(pair_factory, Server, tls_contexts):
    server_context, client_context, _ = tls_contexts

    async with pair_factory(
        None,
        Server(ssl_context=server_context, require_tls=True),
        logged=False,
        do_quit=False,
    ) as pair:
        with pytest.raises(aioftp.StatusCodeError) as exc_info:
            await pair.client.login()
        assert exc_info.value.received_codes[-1] == "534"

        await pair.client.upgrade_to_tls(client_context)
        await pair.client.login()


@pytest.mark.asyncio
async def test_pipelined_command_before_auth_is_rejected(pair_factory, Server, tls_contexts):
    server_context, _, _ = tls_contexts

    async with pair_factory(
        None,
        Server(ssl_context=server_context),
        connected=False,
        logged=False,
        do_quit=False,
    ) as pair:
        reader, writer = await asyncio.open_connection(
            pair.server.server_host,
            pair.server.server_port,
        )
        stream = aioftp.StreamIO(reader, writer)
        assert (await read_response(stream)).startswith("220 ")
        writer.write(b"AUTH TLS\r\nPBSZ 0\r\n")
        await writer.drain()

        assert (await read_response(stream)).startswith("503 ")
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_handshake_interruption_releases_command_connection(pair_factory, Server, tls_contexts):
    server_context, _, _ = tls_contexts

    async with pair_factory(
        None,
        Server(ssl_context=server_context),
        connected=False,
        logged=False,
        do_quit=False,
    ) as pair:
        reader, writer = await asyncio.open_connection(
            pair.server.server_host,
            pair.server.server_port,
        )
        stream = aioftp.StreamIO(reader, writer)
        assert (await read_response(stream)).startswith("220 ")
        writer.write(b"AUTH TLS\r\n")
        await writer.drain()
        assert (await read_response(stream)).startswith("234 ")
        dispatcher = next(iter(pair.server.connections.values()))._dispatcher

        writer.close()
        await writer.wait_closed()

        await asyncio.wait_for(dispatcher, timeout=1)


@pytest.mark.asyncio
async def test_interrupted_passive_tls_handshake_releases_data_connection(pair_factory, Server, tls_contexts):
    server_context, client_context, _ = tls_contexts

    async with pair_factory(
        None,
        Server(ssl_context=server_context),
        logged=False,
        do_quit=False,
    ) as pair:
        await pair.client.upgrade_to_tls(client_context)
        await pair.client.login()

        _, info = await pair.client.command("EPSV", "229")
        protected_port = int(info[0].rsplit("|", 2)[-2])
        with socket.socket() as handshake_socket:
            handshake_socket.connect((pair.server.server_host, protected_port))
        await pair.client.quit()

        assert pair.server.connections == {}
