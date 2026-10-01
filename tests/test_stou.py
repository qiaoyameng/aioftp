import asyncio
import pathlib
import tempfile

import pytest

import aioftp


@pytest.mark.asyncio
async def test_stou_basic(pair_factory):
    async with pair_factory() as pair:
        async with pair.client.stou_stream("foo") as stream:
            first_path = stream.server_path
            await stream.write(b"foobar")
        assert first_path == pathlib.PurePosixPath("/foo")
        async with pair.client.download_stream("foo") as stream:
            assert await stream.read() == b"foobar"


@pytest.mark.asyncio
async def test_stou_unique_names(pair_factory):
    async with pair_factory() as pair:
        paths = set()
        for _ in range(5):
            path = await pair.client.upload_unique(pathlib.Path(__file__), "foo")
            assert path not in paths
            paths.add(path)
        assert len(paths) == 5
        assert pathlib.PurePosixPath("/foo") in paths


@pytest.mark.asyncio
async def test_stou_content_preserved(pair_factory):
    async with pair_factory() as pair:
        payloads = {}
        for index in range(5):
            stream = await pair.client.stou_stream("foo")
            data = f"content-{index}".encode()
            async with stream:
                await stream.write(data)
                path = stream.server_path
            payloads[path] = data
        for path, data in payloads.items():
            async with pair.client.download_stream(str(path)) as stream:
                assert await stream.read() == data


@pytest.mark.asyncio
async def test_stou_concurrent_same_hint(pair_factory):
    async with pair_factory() as pair:
        count = 8

        async def upload(index):
            stream = await pair.client.stou_stream("same")
            data = f"payload-{index}".encode()
            async with stream:
                await stream.write(data)
                path = stream.server_path
            return path, data

        results = await asyncio.gather(*(upload(i) for i in range(count)))
        paths = [path for path, _ in results]
        assert len(set(paths)) == count
        for path, data in results:
            async with pair.client.download_stream(str(path)) as stream:
                assert await stream.read() == data


@pytest.mark.asyncio
async def test_stou_no_hint_uses_current_directory(pair_factory):
    async with pair_factory() as pair:
        async with pair.client.stou_stream() as stream:
            await stream.write(b"foobar")
            path = stream.server_path
        assert path.parent == pathlib.PurePosixPath("/")
        assert await pair.server_paths_exists(str(path))


@pytest.mark.asyncio
async def test_stou_directory_hint(pair_factory):
    async with pair_factory() as pair:
        await pair.client.make_directory("bar")
        async with pair.client.stou_stream("bar/foo") as stream:
            await stream.write(b"x")
            path = stream.server_path
        assert path.parent == pathlib.PurePosixPath("/bar")
        assert await pair.server_paths_exists(str(path))


@pytest.mark.asyncio
async def test_stou_write_failure_leaves_partial_file(
    pair_factory,
    Server,
    expect_codes_in_exception,
):
    class OsErrorPathIO(aioftp.MemoryPathIO):
        @aioftp.pathio.universal_exception
        async def write(self, file, data):
            await file.write(data)
            raise OSError("disk full")

    s = Server(path_io_factory=OsErrorPathIO)
    async with pair_factory(None, s) as pair:
        with expect_codes_in_exception("451"):
            async with pair.client.stou_stream("foo") as stream:
                created_path = stream.server_path
                await stream.write(b"partial")
        assert await pair.server_paths_exists(str(created_path)) is True


@pytest.mark.asyncio
async def test_stou_interrupt_leaves_partial_file(pair_factory):
    async with pair_factory() as pair:
        stream = await pair.client.stou_stream("foo")
        created_path = stream.server_path
        await stream.write(b"partial")
        with pytest.raises((ConnectionResetError, BrokenPipeError)):
            while True:
                await stream.write(b"-" * aioftp.DEFAULT_BLOCK_SIZE)
                await pair.client.abort()
        assert await pair.server_paths_exists(str(created_path)) is True


@pytest.mark.asyncio
async def test_stou_unreachable_directory(pair_factory, expect_codes_in_exception):
    async with pair_factory() as pair:
        with expect_codes_in_exception("550"):
            await pair.client.stou_stream("no/such/dir/foo")


@pytest.mark.asyncio
async def test_stou_not_supported(pair_factory, Server, expect_codes_in_exception):
    class NoUniquePathIO(aioftp.MemoryPathIO):
        supports_unique_file = False

        async def open_unique(self, path):
            raise NotImplementedError

    s = Server(path_io_factory=NoUniquePathIO)
    async with pair_factory(None, s) as pair:
        with expect_codes_in_exception("502"):
            await pair.client.stou_stream("foo")


@pytest.mark.asyncio
async def test_stou_permission_denied(Server, pair_factory, expect_codes_in_exception):
    user = aioftp.User(permissions=(aioftp.Permission("/", writable=False),))
    server = Server(users=(user,))
    async with pair_factory(None, server) as pair:
        with expect_codes_in_exception("550"):
            await pair.client.stou_stream("foo")


@pytest.mark.asyncio
async def test_stou_real_fs_concurrent(Client, Server, pair_factory):
    with tempfile.TemporaryDirectory() as root:
        user = aioftp.User(base_path=root)
        client = Client(path_io_factory=aioftp.PathIO)
        server = Server(users=(user,), path_io_factory=aioftp.PathIO)
        async with pair_factory(client, server) as pair:
            count = 8

            async def upload(index):
                path = await pair.client.upload_unique(pathlib.Path(__file__), "same")
                return path, f"ignored-{index}".encode()

            paths = await asyncio.gather(*(upload(i) for i in range(count)))
            assert len({path for path, _ in paths}) == count
            for path, _ in paths:
                real = pathlib.Path(root) / str(path).lstrip("/")
                assert real.is_file()
            assert (pathlib.Path(root) / "same").is_file()


@pytest.mark.asyncio
async def test_stou_real_fs_write_failure(Client, Server, pair_factory, expect_codes_in_exception):
    class OsErrorPathIO(aioftp.PathIO):
        @aioftp.pathio.universal_exception
        async def write(self, file, data):
            await file.write(data)
            raise OSError("disk full")

    with tempfile.TemporaryDirectory() as root:
        user = aioftp.User(base_path=root)
        client = Client(path_io_factory=aioftp.PathIO)
        server = Server(users=(user,), path_io_factory=OsErrorPathIO)
        async with pair_factory(client, server) as pair:
            with expect_codes_in_exception("451"):
                async with pair.client.stou_stream("foo") as stream:
                    created_path = stream.server_path
                    await stream.write(b"partial")
            real = pathlib.Path(root) / str(created_path).lstrip("/")
            assert real.is_file()
            assert real.read_bytes() == b"partial"
