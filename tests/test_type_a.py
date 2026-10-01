import pytest


async def stor(pair, name, data, conn_type="I"):
    async with pair.client.get_stream(
        "STOR " + name,
        "1xx",
        conn_type=conn_type,
    ) as stream:
        await stream.write(data)


async def retr(pair, name, conn_type="I"):
    async with pair.client.get_stream(
        "RETR " + name,
        "1xx",
        conn_type=conn_type,
    ) as stream:
        return await stream.read()


@pytest.mark.asyncio
async def test_type_a_stor_converts_nvt_to_local(pair_factory, Server):
    async with pair_factory(server=Server(block_size=1)) as pair:
        # CRLF, CR NUL and a trailing lone CR, all arriving byte by byte
        await stor(pair, "foo.txt", b"foo\r\nbar\r\x00baz\r", conn_type="A")
        assert await retr(pair, "foo.txt") == b"foo\nbar\rbaz\r"


@pytest.mark.asyncio
async def test_type_a_retr_converts_local_to_nvt(pair_factory, Server):
    async with pair_factory(server=Server(block_size=1)) as pair:
        # local newline is "\n", every local "\r" is data
        await stor(pair, "foo.txt", b"foo\nbar\rbaz\r")
        assert await retr(pair, "foo.txt", conn_type="A") == b"foo\r\nbar\r\x00baz\r\x00"


@pytest.mark.asyncio
async def test_type_a_retr_local_crlf(pair_factory, Server):
    async with pair_factory(server=Server(block_size=1)) as pair:
        # local "\r\n" is data CR followed by a newline
        await stor(pair, "foo.txt", b"a\r\nb")
        assert await retr(pair, "foo.txt", conn_type="A") == b"a\r\x00\r\nb"


@pytest.mark.asyncio
async def test_type_a_round_trip(pair_factory, Server):
    async with pair_factory(server=Server(block_size=1)) as pair:
        data = b"a\r\nb\r\x00c\r\n"
        await stor(pair, "foo.txt", data, conn_type="A")
        assert await retr(pair, "foo.txt", conn_type="A") == data


@pytest.mark.asyncio
async def test_type_a_empty_file(pair_factory, Server):
    async with pair_factory(server=Server(block_size=1)) as pair:
        await stor(pair, "foo.txt", b"", conn_type="A")
        assert await retr(pair, "foo.txt") == b""
        assert await retr(pair, "foo.txt", conn_type="A") == b""


@pytest.mark.asyncio
async def test_type_a_rest_rejected(pair_factory, expect_codes_in_exception):
    async with pair_factory() as pair:
        atom = b"foobar"
        await pair.make_server_files("foo.txt", size=1, atom=atom)
        await pair.client.command("TYPE A", "200")
        with expect_codes_in_exception("504"):
            await pair.client.command("REST 3", "350")
        # restart offset is not applied after the rejection
        await pair.client.command("TYPE I", "200")
        async with pair.client.download_stream("foo.txt") as stream:
            assert await stream.read() == atom
