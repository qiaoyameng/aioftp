import pytest

from aioftp.common import (
    LocalToNetworkNewlineTranslator,
    NetworkToLocalNewlineTranslator,
)


def _feed_byte_by_byte(translator, data):
    out = b"".join(translator.translate(bytes((byte,))) for byte in data)
    return out + translator.flush()


@pytest.mark.parametrize(
    ("wire", "local"),
    [
        pytest.param(b"", b"", id="empty"),
        pytest.param(b"abc", b"abc", id="no-newline"),
        pytest.param(b"a\r\nb\r\n", b"a\nb\n", id="crlf"),
        pytest.param(b"a\r\x00b", b"a\rb", id="cr-nul"),
        pytest.param(b"\r\n", b"\n", id="lone-crlf"),
        pytest.param(b"\r\x00", b"\r", id="lone-cr-nul"),
        pytest.param(b"abc\r", b"abc\r", id="trailing-lone-cr"),
        pytest.param(b"\r", b"\r", id="only-cr"),
        pytest.param(b"\r\r\n", b"\r\n", id="cr-then-crlf"),
        pytest.param(b"\r\x00\r\n", b"\r\n", id="cr-nul-then-crlf"),
        pytest.param(b"a\rb", b"a\rb", id="bare-cr-passed-through"),
    ],
)
def test_network_to_local_translator(wire, local):
    assert _feed_byte_by_byte(NetworkToLocalNewlineTranslator(), wire) == local


@pytest.mark.parametrize(
    ("local", "wire"),
    [
        pytest.param(b"", b"", id="empty"),
        pytest.param(b"abc", b"abc", id="no-newline"),
        pytest.param(b"a\nb\n", b"a\r\nb\r\n", id="lf"),
        pytest.param(b"a\r\nb", b"a\r\nb", id="already-crlf"),
        pytest.param(b"a\rb", b"a\r\x00b", id="bare-cr"),
        pytest.param(b"\n", b"\r\n", id="lone-lf"),
        pytest.param(b"\r", b"\r\x00", id="lone-cr"),
        pytest.param(b"abc\r", b"abc\r\x00", id="trailing-lone-cr"),
        pytest.param(b"\r\r\n", b"\r\x00\r\n", id="cr-then-crlf"),
        pytest.param(b"\n\n\r", b"\r\n\r\n\r\x00", id="mixed"),
    ],
)
def test_local_to_network_translator(local, wire):
    assert _feed_byte_by_byte(LocalToNetworkNewlineTranslator(), local) == wire


@pytest.mark.parametrize("split", [1, 2, 3, 7, 64])
def test_translators_arbitrary_chunk_sizes(split):
    wire = b"line1\r\nc\r\x00here\r\ntail\r\n"
    local = b"line1\nc\rhere\ntail\n"

    translator = NetworkToLocalNewlineTranslator()
    out = b"".join(translator.translate(chunk) for chunk in (wire[i : i + split] for i in range(0, len(wire), split)))
    assert out + translator.flush() == local

    translator = LocalToNetworkNewlineTranslator()
    out = b"".join(translator.translate(chunk) for chunk in (local[i : i + split] for i in range(0, len(local), split)))
    assert out + translator.flush() == wire


@pytest.mark.parametrize("block_size", [1, 2, 5, 8192])
@pytest.mark.asyncio
async def test_type_a_download(pair_factory, block_size):
    # wire/NVT content with CRLF, CR NUL and a trailing isolated CR
    wire = b"a\r\nb\r\x00c\rtail\r"
    expected = b"a\nb\rc\rtail\r"
    async with pair_factory() as pair:
        async with pair.client.upload_stream("nvt.txt", conn_type="I") as stream:
            await stream.write(wire)
        chunks = []
        async with pair.client.download_stream("nvt.txt", conn_type="A") as stream:
            async for chunk in stream.iter_by_block(block_size):
                chunks.append(chunk)
        assert b"".join(chunks) == expected


@pytest.mark.parametrize("block_size", [1, 2, 5, 8192])
@pytest.mark.asyncio
async def test_type_a_upload(pair_factory, block_size):
    local = b"a\nb\rc\rtail\r"
    expected = b"a\r\nb\r\x00c\r\x00tail\r\x00"
    async with pair_factory() as pair:
        async with pair.client.upload_stream("nvt.txt", conn_type="A") as stream:
            for i in range(0, len(local), block_size):
                await stream.write(local[i : i + block_size])
        async with pair.client.download_stream("nvt.txt", conn_type="I") as stream:
            assert await stream.read() == expected


@pytest.mark.asyncio
async def test_type_a_empty_file(pair_factory):
    async with pair_factory() as pair:
        async with pair.client.upload_stream("empty.txt", conn_type="A") as stream:
            pass
        async with pair.client.download_stream("empty.txt", conn_type="A") as stream:
            assert await stream.read() == b""


@pytest.mark.asyncio
async def test_type_a_vs_type_i_roundtrip(pair_factory):
    local = b"first\nsecond\rboth\r\nend\r"
    async with pair_factory() as pair:
        async with pair.client.upload_stream("a.txt", conn_type="A") as stream:
            await stream.write(local)
        async with pair.client.upload_stream("i.txt", conn_type="I") as stream:
            await stream.write(local)
        async with pair.client.download_stream("a.txt", conn_type="A") as stream:
            assert await stream.read() == local
        async with pair.client.download_stream("i.txt", conn_type="I") as stream:
            assert await stream.read() == local
        # the TYPE A wire bytes must really differ (translation happened)
        async with pair.client.download_stream("a.txt", conn_type="I") as stream:
            wire = await stream.read()
        assert wire != local
        assert b"\r\n" in wire


@pytest.mark.parametrize("method", ["upload_stream", "append_stream", "download_stream"])
@pytest.mark.asyncio
async def test_type_a_rest_rejected_client(pair_factory, method):
    async with pair_factory() as pair:
        stream_factory = getattr(pair.client, method)
        with pytest.raises(ValueError):
            await stream_factory("x.txt", conn_type="A", offset=3)


@pytest.mark.asyncio
async def test_type_a_rest_rejected_server(pair_factory, expect_codes_in_exception):
    async with pair_factory() as pair:
        await pair.client.command("TYPE A", "200")
        with expect_codes_in_exception("501"):
            await pair.client.command("REST 3", "350")
