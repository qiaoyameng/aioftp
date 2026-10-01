import asyncio
from pathlib import PurePosixPath

import pytest

import aioftp


@pytest.mark.asyncio
async def test_stou_upload_unique(pair_factory):
    async with pair_factory() as pair:
        await pair.make_client_files("foo.txt", size=1, atom=b"foobar")
        path = await pair.client.upload_unique("foo.txt")
        assert path == PurePosixPath("/foo.txt")
        async with pair.client.download_stream(path) as stream:
            data = await stream.read()
        assert data == b"foobar"


@pytest.mark.asyncio
async def test_stou_upload_unique_into_subdirectory(pair_factory):
    async with pair_factory() as pair:
        await pair.make_client_files("foo.txt", size=1, atom=b"foobar")
        path = await pair.client.upload_unique("foo.txt", "sub/dir/data.txt")
        assert path == PurePosixPath("/sub/dir/data.txt")
        async with pair.client.download_stream(path) as stream:
            data = await stream.read()
        assert data == b"foobar"


@pytest.mark.asyncio
async def test_stou_stream_without_hint(pair_factory):
    async with pair_factory() as pair:
        async with pair.client.upload_unique_stream() as stream:
            await stream.write(b"foobar")
        assert stream.unique_path == PurePosixPath("/stou")
        async with pair.client.download_stream("/stou") as stream:
            data = await stream.read()
        assert data == b"foobar"


@pytest.mark.asyncio
async def test_stou_same_hint_sequential(pair_factory):
    async with pair_factory() as pair:
        await pair.make_client_files("foo.txt", size=1, atom=b"a")
        first = await pair.client.upload_unique("foo.txt", "data.txt")
        await pair.make_client_files("foo.txt", size=1, atom=b"b")
        second = await pair.client.upload_unique("foo.txt", "data.txt")
        assert first != second
        assert await pair.server_paths_exists(first, second)
        contents = []
        for path in (first, second):
            async with pair.client.download_stream(path) as stream:
                contents.append(await stream.read())
        assert sorted(contents) == [b"a", b"b"]


@pytest.mark.asyncio
async def test_stou_concurrent_same_hint(pair_factory, Server, tmp_path):
    s = Server(
        [aioftp.User(base_path=tmp_path)],
        path_io_factory=aioftp.PathIO,
    )
    async with pair_factory(None, s) as pair:
        count = 8

        async def upload_one(i):
            client = aioftp.Client()
            await client.connect(pair.server.server_host, pair.server.server_port)
            await client.login()
            try:
                payload = f"content-{i}-".encode() * 100
                async with client.upload_unique_stream("data.txt") as stream:
                    await stream.write(payload)
                return stream.unique_path, payload
            finally:
                await client.quit()
                client.close()

        results = await asyncio.gather(*(upload_one(i) for i in range(count)))
        paths = [path for path, _ in results]
        # every upload got its own file, no upload was overwritten
        assert len(set(paths)) == count
        on_disk = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        assert len(on_disk) == count
        for path, payload in results:
            assert on_disk[path.name] == payload


@pytest.mark.asyncio
async def test_stou_permission_denied_nothing_created(
    pair_factory,
    Server,
    expect_codes_in_exception,
):
    s = Server(
        [
            aioftp.User(permissions=[aioftp.Permission(writable=False)]),
        ],
    )
    async with pair_factory(None, s) as pair:
        await pair.make_client_files("foo.txt", size=1, atom=b"foobar")
        with expect_codes_in_exception("550"):
            await pair.client.upload_unique("foo.txt", "data.txt")
        assert not await pair.client.exists("data.txt")
        assert await pair.client.list() == []


@pytest.mark.asyncio
async def test_stou_not_supported_by_path_io(
    pair_factory,
    Server,
    expect_codes_in_exception,
):
    class NoUniqueFilePathIO(aioftp.MemoryPathIO):
        supports_unique_file_creation = False

    s = Server(path_io_factory=NoUniqueFilePathIO)
    async with pair_factory(None, s) as pair:
        with expect_codes_in_exception("502"):
            async with pair.client.upload_unique_stream("data.txt"):
                pass
        assert await pair.client.list() == []


@pytest.mark.asyncio
async def test_stou_disk_write_failure_leaves_partial_file(pair_factory, Server):
    class FailingWritePathIO(aioftp.MemoryPathIO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.writes = 0

        @aioftp.pathio.universal_exception
        async def write(self, file, data):
            self.writes += 1
            if self.writes > 1:
                raise OSError("disk is full")
            return await super().write(file, data)

    s = Server(path_io_factory=FailingWritePathIO)
    async with pair_factory(None, s) as pair:
        await pair.make_client_files("foo.txt")
        with pytest.raises(aioftp.UploadIncompleteError) as exc_info:
            await pair.client.upload_unique("foo.txt", "data.txt")
        assert exc_info.value.path == PurePosixPath("/data.txt")
        # file was created, but only the first block made it to the disk
        assert await pair.client.exists("data.txt")
        async with pair.client.download_stream("data.txt") as stream:
            data = await stream.read()
        assert data == b"-" * aioftp.DEFAULT_BLOCK_SIZE


@pytest.mark.asyncio
async def test_stou_client_abort_leaves_partial_file(pair_factory):
    async with pair_factory() as pair:
        sent = b"-" * aioftp.DEFAULT_BLOCK_SIZE
        stream = await pair.client.upload_unique_stream("data.txt")
        await stream.write(sent)
        await asyncio.sleep(0.1)
        await pair.client.abort()
        stream.close()
        path = stream.unique_path
        # file was created before the transfer started and stays on the
        # server with the data received so far
        assert await pair.client.exists(path)
        async with pair.client.download_stream(path) as download:
            data = await download.read()
        assert sent[: len(data)] == data
        assert len(data) <= len(sent)


@pytest.mark.asyncio
async def test_stou_path_unreachable(pair_factory, expect_codes_in_exception):
    async with pair_factory() as pair:
        with expect_codes_in_exception("550"):
            async with pair.client.upload_unique_stream("foo/bar/data.txt"):
                pass
        assert await pair.client.list() == []
