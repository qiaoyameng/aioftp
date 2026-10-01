import asyncio
import ssl
import sys

import pytest

import aioftp

if sys.version_info < (3, 11):
    pytest.skip(reason="required python 3.11+", allow_module_level=True)


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


async def wait_for_connections(server, count=0):
    for _ in range(100):
        if len(server.connections) == count:
            return
        await asyncio.sleep(0.01)
    raise RuntimeError(f"server still has {len(server.connections)} connections")


@pytest.mark.asyncio
async def test_explicit_tls_upgrade_and_passive_data_connection(
    pair_factory,
    Server,
    ssl_server_context,
    ssl_client_context,
):
    async with pair_factory(
        server=Server(ssl=ssl_server_context, explicit_tls=True),
        logged=False,
        do_quit=False,
    ) as pair:
        await pair.client.upgrade_to_tls(sslcontext=ssl_client_context)
        assert pair.client.stream.writer.transport.get_extra_info("ssl_object") is not None
        await pair.client.login()
        await pair.make_server_files("foo.txt")
        async with pair.client.download_stream("foo.txt") as stream:
            assert stream.writer.transport.get_extra_info("ssl_object") is not None
            data = await stream.read()
        assert len(data) == aioftp.DEFAULT_BLOCK_SIZE * 3


@pytest.mark.asyncio
async def test_explicit_tls_prot_switch_affects_only_new_data_connections(
    pair_factory,
    Server,
    ssl_server_context,
    ssl_client_context,
):
    async with pair_factory(
        server=Server(ssl=ssl_server_context, explicit_tls=True),
        logged=False,
        do_quit=False,
    ) as pair:
        await pair.client.upgrade_to_tls(sslcontext=ssl_client_context)
        await pair.client.login()
        await pair.make_server_files("foo.txt")
        async with pair.client.download_stream("foo.txt") as stream:
            assert stream.writer.transport.get_extra_info("ssl_object") is not None
            await stream.read()
        await pair.client.set_data_channel_protection(False)
        async with pair.client.download_stream("foo.txt") as stream:
            assert stream.writer.transport.get_extra_info("ssl_object") is None
            await stream.read()
        await pair.client.set_data_channel_protection(True)
        async with pair.client.download_stream("foo.txt") as stream:
            assert stream.writer.transport.get_extra_info("ssl_object") is not None
            await stream.read()


@pytest.mark.asyncio
async def test_explicit_tls_repeated_auth(pair_factory, Server, ssl_server_context, ssl_client_context):
    async with pair_factory(
        server=Server(ssl=ssl_server_context, explicit_tls=True),
        logged=False,
        do_quit=False,
    ) as pair:
        await pair.client.upgrade_to_tls(sslcontext=ssl_client_context)
        await pair.client.command("AUTH TLS", "503")
        await pair.client.login()
        await pair.client.command("PWD", "257")


@pytest.mark.asyncio
async def test_explicit_tls_certificate_verification_failure(pair_factory, Server, ssl_server_context):
    untrusted_context = ssl.create_default_context()
    async with pair_factory(
        server=Server(ssl=ssl_server_context, explicit_tls=True),
        logged=False,
        do_quit=False,
    ) as pair:
        with pytest.raises(ssl.SSLCertVerificationError):
            await pair.client.upgrade_to_tls(sslcontext=untrusted_context)
        pair.client.close()
        await wait_for_connections(pair.server)


@pytest.mark.asyncio
async def test_explicit_tls_login_requires_tls(
    pair_factory,
    Server,
    ssl_server_context,
    ssl_client_context,
    expect_codes_in_exception,
):
    async with pair_factory(
        server=Server(
            ssl=ssl_server_context,
            explicit_tls=True,
            require_tls_for_login=True,
        ),
        logged=False,
        do_quit=False,
    ) as pair:
        with expect_codes_in_exception("534"):
            await pair.client.login()
        await pair.client.upgrade_to_tls(sslcontext=ssl_client_context)
        await pair.client.login()


@pytest.mark.asyncio
async def test_explicit_tls_handshake_interruption_releases_connection(pair_factory, Server, ssl_server_context):
    async with pair_factory(
        server=Server(ssl=ssl_server_context, explicit_tls=True),
        connected=False,
        logged=False,
        do_quit=False,
    ) as pair:
        reader, writer = await asyncio.open_connection(
            pair.server.server_host,
            pair.server.server_port,
        )
        greeting = await reader.readline()
        assert greeting.startswith(b"220")
        writer.write(b"AUTH TLS\r\n")
        await writer.drain()
        response = await reader.readline()
        assert response.startswith(b"234")
        # interrupt the handshake: close instead of sending a ClientHello
        writer.close()
        await wait_for_connections(pair.server)
        # the server is still alive and accepts new connections
        reader, writer = await asyncio.open_connection(
            pair.server.server_host,
            pair.server.server_port,
        )
        greeting = await reader.readline()
        assert greeting.startswith(b"220")
        writer.close()


@pytest.mark.asyncio
async def test_explicit_tls_data_connection_handshake_interruption(
    pair_factory,
    Server,
    ssl_server_context,
    ssl_client_context,
):
    async with pair_factory(
        server=Server(ssl=ssl_server_context, explicit_tls=True),
        logged=False,
        do_quit=False,
    ) as pair:
        await pair.client.upgrade_to_tls(sslcontext=ssl_client_context)
        await pair.client.login()
        code, info = await pair.client.command("EPSV", "229")
        _, port = pair.client.parse_epsv_response(info[-1])
        reader, writer = await asyncio.open_connection(pair.client.server_host, port)
        writer.write(b"this is not a TLS handshake\r\n")
        await writer.drain()
        writer.close()
        await asyncio.sleep(0.1)
        # broken data connection is discarded, control connection stays usable
        await pair.client.command("PWD", "257")
        (connection,) = pair.server.connections.values()
        assert not connection.future.data_connection.done()
