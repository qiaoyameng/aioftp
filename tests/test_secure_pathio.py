import contextlib
import functools
import os
import threading

import pytest

import aioftp

pytestmark = pytest.mark.skipif(
    not aioftp.SecurePathIO.is_supported(),
    reason="SecurePathIO is not supported on this platform",
)


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "root"
    path.mkdir()
    return path


@pytest.fixture
def outside(tmp_path):
    path = tmp_path / "outside"
    path.mkdir()
    return path


@pytest.fixture
def sentinel(outside):
    path = outside / "sentinel.txt"
    path.write_text("TOP SECRET")
    return path


def test_capability_reporting():
    assert aioftp.SecurePathIO.is_supported()
    assert aioftp.SecurePathIO.missing_capabilities() == ()


def test_capability_reporting_unsupported_platform(monkeypatch):
    import aioftp.pathio

    monkeypatch.setattr(aioftp.pathio.sys, "platform", "darwin")
    assert not aioftp.SecurePathIO.is_supported()
    assert "linux" in aioftp.SecurePathIO.missing_capabilities()
    with pytest.raises(NotImplementedError):
        aioftp.SecurePathIO()


@pytest.mark.asyncio
async def test_basic_operations(root):
    path_io = aioftp.SecurePathIO(root=root)
    await path_io.mkdir(root / "sub" / "deep", parents=True)
    assert await path_io.is_dir(root / "sub")
    assert await path_io.is_dir(root / "sub" / "deep")
    async with path_io.open(root / "sub" / "file.txt", mode="wb") as file:
        await file.write(b"hello")
    assert await path_io.exists(root / "sub" / "file.txt")
    assert await path_io.is_file(root / "sub" / "file.txt")
    async with path_io.open(root / "sub" / "file.txt", mode="rb") as file:
        assert await file.read(100) == b"hello"
    async with path_io.open(root / "sub" / "file.txt", mode="ab") as file:
        await file.write(b"!")
    assert (root / "sub" / "file.txt").read_bytes() == b"hello!"
    info = await path_io.stat(root / "sub" / "file.txt")
    assert info.st_size == 6
    names = sorted([p.name async for p in path_io.list(root / "sub")])
    assert names == ["deep", "file.txt"]
    await path_io.rename(root / "sub" / "file.txt", root / "sub" / "renamed.txt")
    assert not await path_io.exists(root / "sub" / "file.txt")
    assert await path_io.exists(root / "sub" / "renamed.txt")
    await path_io.unlink(root / "sub" / "renamed.txt")
    await path_io.rmdir(root / "sub" / "deep")
    assert not await path_io.exists(root / "sub" / "deep")


@pytest.mark.asyncio
async def test_path_outside_root_rejected(root, sentinel):
    path_io = aioftp.SecurePathIO(root=root)
    with pytest.raises(aioftp.PathIOError):
        await path_io.stat(sentinel)
    with pytest.raises(aioftp.PathIOError):
        await path_io.unlink(sentinel)
    with pytest.raises(aioftp.PathIOError):
        async with path_io.open(sentinel, mode="rb"):
            pass
    assert sentinel.read_text() == "TOP SECRET"


@pytest.mark.asyncio
async def test_symlink_escape_rejected(root, outside, sentinel):
    path_io = aioftp.SecurePathIO(root=root)
    os.symlink(outside, root / "link")
    with pytest.raises(aioftp.PathIOError):
        async with path_io.open(root / "link" / "sentinel.txt", mode="rb"):
            pass
    with pytest.raises(aioftp.PathIOError):
        async with path_io.open(root / "link" / "sentinel.txt", mode="wb"):
            pass
    with pytest.raises(aioftp.PathIOError):
        await path_io.unlink(root / "link" / "sentinel.txt")
    with pytest.raises(aioftp.PathIOError):
        await path_io.stat(root / "link" / "sentinel.txt")
    assert sentinel.read_text() == "TOP SECRET"


@pytest.mark.asyncio
async def test_symlink_policy_explicit(root, sentinel):
    # symlinks are rejected by default ...
    (root / "target.txt").write_text("inner")
    os.symlink("target.txt", root / "inner_link")
    with pytest.raises(aioftp.PathIOError):
        async with aioftp.SecurePathIO(root=root).open(root / "inner_link", mode="rb"):
            pass
    # ... and allowed inside the root when requested ...
    path_io = aioftp.SecurePathIO(root=root, allow_symlinks=True)
    async with path_io.open(root / "inner_link", mode="rb") as file:
        assert await file.read(100) == b"inner"
    # ... but escaping symlinks are still rejected
    os.symlink(str(sentinel), root / "absolute_link")
    os.symlink("../outside/sentinel.txt", root / "relative_link")
    for name in ("absolute_link", "relative_link"):
        with pytest.raises(aioftp.PathIOError):
            async with path_io.open(root / name, mode="rb"):
                pass
        with pytest.raises(aioftp.PathIOError):
            await path_io.stat(root / name)
    assert sentinel.read_text() == "TOP SECRET"


@pytest.mark.asyncio
async def test_rename_source_unchanged(root):
    path_io = aioftp.SecurePathIO(root=root)
    (root / "a.txt").write_text("data")
    token = await path_io.acquire_rename_source(root / "a.txt")
    await path_io.rename(token, root / "b.txt")
    assert (root / "b.txt").read_text() == "data"


@pytest.mark.asyncio
async def test_rename_source_file_swap_detected(root):
    path_io = aioftp.SecurePathIO(root=root)
    (root / "a.txt").write_text("one")
    token = await path_io.acquire_rename_source(root / "a.txt")
    # replace the source object with another one at the same path
    (root / "a.txt").unlink()
    (root / "a.txt").write_text("two")
    with pytest.raises(aioftp.PathIOError):
        await path_io.rename(token, root / "b.txt")
    assert (root / "a.txt").read_text() == "two"
    assert not (root / "b.txt").exists()


@pytest.mark.asyncio
async def test_rename_source_directory_swap_detected(root):
    path_io = aioftp.SecurePathIO(root=root)
    (root / "dir").mkdir()
    token = await path_io.acquire_rename_source(root / "dir")
    (root / "dir").rmdir()
    (root / "dir").mkdir()
    with pytest.raises(aioftp.PathIOError):
        await path_io.rename(token, root / "other")
    assert not (root / "other").exists()


@pytest.mark.asyncio
async def test_rename_source_removed_detected(root):
    path_io = aioftp.SecurePathIO(root=root)
    (root / "a.txt").write_text("one")
    token = await path_io.acquire_rename_source(root / "a.txt")
    (root / "a.txt").unlink()
    with pytest.raises(aioftp.PathIOError):
        await path_io.rename(token, root / "b.txt")


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_symlinks", [False, True])
async def test_directory_symlink_swap_race(root, outside, sentinel, allow_symlinks):
    """
    Another task keeps replacing a directory inside the root with a symlink
    to an outside directory. No matter when the path is resolved, the
    outside sentinel file must never be read, written or removed.
    """
    path_io = aioftp.SecurePathIO(root=root, allow_symlinks=allow_symlinks)
    victim = root / "sub"
    held = root / "held"
    victim.mkdir()
    stop = threading.Event()
    swaps = 0

    def swapper():
        nonlocal swaps
        while not stop.is_set():
            with contextlib.suppress(OSError):
                os.rename(victim, held)
            with contextlib.suppress(OSError):
                os.symlink(outside, victim)
                swaps += 1
            with contextlib.suppress(OSError):
                os.unlink(victim)
            with contextlib.suppress(OSError):
                os.rename(held, victim)

    thread = threading.Thread(target=swapper)
    thread.start()
    try:
        for _ in range(300):
            target = victim / "sentinel.txt"
            try:
                async with path_io.open(target, mode="rb") as file:
                    data = await file.read(100)
                assert data != b"TOP SECRET"
            except aioftp.PathIOError:
                pass
            try:
                async with path_io.open(target, mode="wb") as file:
                    await file.write(b"attempt")
            except aioftp.PathIOError:
                pass
            try:
                await path_io.unlink(target)
            except aioftp.PathIOError:
                pass
            assert sentinel.read_text() == "TOP SECRET"
    finally:
        stop.set()
        thread.join()
    assert swaps > 0
    assert sentinel.read_text() == "TOP SECRET"


@pytest.mark.asyncio
async def test_server_file_operations(pair_factory, Server, tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    s = Server(
        [aioftp.User(base_path=base)],
        path_io_factory=functools.partial(aioftp.SecurePathIO, root=base),
    )
    async with pair_factory(None, s) as pair:
        await pair.client.make_directory("sub")
        async with pair.client.upload_stream("sub/file.txt") as stream:
            await stream.write(b"hello")
        async with pair.client.download_stream("sub/file.txt") as stream:
            assert await stream.read() == b"hello"
        assert await pair.client.exists("sub/file.txt")
        await pair.client.rename("sub/file.txt", "sub/renamed.txt")
        assert await pair.client.exists("sub/renamed.txt")
        await pair.client.remove_file("sub/renamed.txt")
        assert not await pair.client.exists("sub/renamed.txt")


@pytest.mark.asyncio
async def test_server_rnfr_rnto_source_swap(
    pair_factory,
    Server,
    expect_codes_in_exception,
    tmp_path,
):
    base = tmp_path / "base"
    base.mkdir()
    (base / "a.txt").write_text("one")
    s = Server(
        [aioftp.User(base_path=base)],
        path_io_factory=functools.partial(aioftp.SecurePathIO, root=base),
    )
    async with pair_factory(None, s) as pair:
        await pair.client.command("RNFR a.txt", "350")
        # the source object is replaced between RNFR and RNTO
        (base / "a.txt").unlink()
        (base / "a.txt").write_text("two")
        with expect_codes_in_exception("451"):
            await pair.client.command("RNTO b.txt", "250")
    assert (base / "a.txt").read_text() == "two"
    assert not (base / "b.txt").exists()
