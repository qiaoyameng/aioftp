import asyncio
import contextlib
import os
import stat
import sys
import threading
from pathlib import Path

import pytest

import aioftp

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="SecurePathIO is supported on Linux only",
)


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    return root


@pytest.fixture
def outside(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    return outside


@pytest.fixture
def sentinel(outside):
    sentinel = outside / "sentinel.txt"
    sentinel.write_bytes(b"secret")
    return sentinel


@pytest.fixture(params=[False, True], ids=["no_symlinks", "allow_symlinks"])
def secure_io(request, root):
    return aioftp.SecurePathIO(root=root, allow_symlinks=request.param)


def test_not_supported_on_other_platforms(monkeypatch, root):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(NotImplementedError):
        aioftp.SecurePathIO(root=root)


def test_not_supported_without_dir_fd(monkeypatch, root):
    monkeypatch.delattr(os, "O_PATH")
    with pytest.raises(NotImplementedError):
        aioftp.SecurePathIO(root=root)


@pytest.mark.asyncio
async def test_basic_file_operations(secure_io, root):
    target = root / "sub" / "dir" / "file.txt"
    await secure_io.mkdir(target.parent, parents=True)
    assert await secure_io.is_dir(target.parent)
    async with secure_io.open(target, mode="wb") as file:
        await file.write(b"hello")
    assert await secure_io.exists(target)
    assert await secure_io.is_file(target)
    async with secure_io.open(target, mode="ab") as file:
        await file.write(b" world")
    async with secure_io.open(target, mode="rb") as file:
        assert await file.read(100) == b"hello world"
    stats = await secure_io.stat(target)
    assert stats.st_size == len(b"hello world")
    assert stat.S_ISREG(stats.st_mode)
    names = {p.name async for p in secure_io.list(target.parent)}
    assert names == {"file.txt"}
    renamed = root / "sub" / "dir" / "renamed.txt"
    await secure_io.rename(target, renamed)
    assert not await secure_io.exists(target)
    assert await secure_io.exists(renamed)
    await secure_io.unlink(renamed)
    assert not await secure_io.exists(renamed)
    await secure_io.rmdir(renamed.parent)
    assert not await secure_io.exists(renamed.parent)


@pytest.mark.asyncio
async def test_mkdir_exist_ok(secure_io, root):
    target = root / "sub"
    await secure_io.mkdir(target)
    with pytest.raises(aioftp.PathIOError):
        await secure_io.mkdir(target)
    await secure_io.mkdir(target, exist_ok=True)
    await secure_io.mkdir(root / "a" / "b" / "c", parents=True, exist_ok=True)
    assert await secure_io.is_dir(root / "a" / "b" / "c")


@pytest.mark.asyncio
async def test_relative_path_resolution(secure_io, root):
    async with secure_io.open(Path("relative.txt"), mode="wb") as file:
        await file.write(b"data")
    assert (root / "relative.txt").read_bytes() == b"data"
    assert await secure_io.exists(Path("relative.txt"))


@pytest.mark.asyncio
async def test_dotdot_escape_rejected(secure_io, root, sentinel):
    with pytest.raises(aioftp.PathIOError):
        await secure_io.exists(root / ".." / "outside" / "sentinel.txt")
    with pytest.raises(aioftp.PathIOError):
        await secure_io.unlink(root / ".." / "outside" / "sentinel.txt")
    assert sentinel.read_bytes() == b"secret"


@pytest.mark.asyncio
async def test_root_itself(secure_io, root):
    assert await secure_io.exists(root)
    assert await secure_io.is_dir(root)
    assert not await secure_io.is_file(root)
    stats = await secure_io.stat(root)
    assert stat.S_ISDIR(stats.st_mode)
    with pytest.raises(aioftp.PathIOError):
        await secure_io.rmdir(root)
    with pytest.raises(aioftp.PathIOError):
        await secure_io.unlink(root)


@pytest.mark.asyncio
async def test_inroot_symlink_policy(root):
    (root / "target.txt").write_bytes(b"content")
    (root / "link").symlink_to("target.txt")
    restricted = aioftp.SecurePathIO(root=root, allow_symlinks=False)
    with pytest.raises(aioftp.PathIOError):
        async with restricted.open(root / "link", mode="rb"):
            pass
    # stat does not follow the symlink, so the link itself is visible
    stats = await restricted.stat(root / "link")
    assert stat.S_ISLNK(stats.st_mode)
    assert not await restricted.is_file(root / "link")

    allowed = aioftp.SecurePathIO(root=root, allow_symlinks=True)
    async with allowed.open(root / "link", mode="rb") as file:
        assert await file.read(100) == b"content"
    assert await allowed.is_file(root / "link")
    async with allowed.open(root / "link", mode="ab") as file:
        await file.write(b"!")
    assert (root / "target.txt").read_bytes() == b"content!"


@pytest.mark.asyncio
async def test_symlink_escape_read_write_unlink(secure_io, root, outside, sentinel):
    (root / "link").symlink_to(outside)
    escape = root / "link" / "sentinel.txt"
    assert not await secure_io.exists(escape)
    with pytest.raises(aioftp.PathIOError):
        async with secure_io.open(escape, mode="rb"):
            pass
    with pytest.raises(aioftp.PathIOError):
        async with secure_io.open(escape, mode="wb"):
            pass
    with pytest.raises(aioftp.PathIOError):
        await secure_io.unlink(escape)
    with pytest.raises(aioftp.PathIOError):
        await secure_io.stat(escape)
    with pytest.raises(aioftp.PathIOError):
        await secure_io.mkdir(escape)
    # renaming the symlink itself inside the root is safe and allowed,
    # but the moved link still can not be used to escape
    await secure_io.rename(root / "link", root / "moved")
    with pytest.raises(aioftp.PathIOError):
        async with secure_io.open(root / "moved" / "sentinel.txt", mode="rb"):
            pass
    assert sentinel.read_bytes() == b"secret"


@pytest.mark.asyncio
async def test_symlink_escape_final_component(secure_io, root, sentinel):
    (root / "link").symlink_to(sentinel)
    with pytest.raises(aioftp.PathIOError):
        async with secure_io.open(root / "link", mode="rb"):
            pass
    with pytest.raises(aioftp.PathIOError):
        async with secure_io.open(root / "link", mode="wb"):
            pass
    assert sentinel.read_bytes() == b"secret"


@pytest.mark.asyncio
async def test_symlink_dotdot_escape(root, outside, sentinel):
    (root / "sub").mkdir()
    (root / "sub" / "evil").symlink_to(os.path.join("..", "..", "outside"))
    secure_io = aioftp.SecurePathIO(root=root, allow_symlinks=True)
    with pytest.raises(aioftp.PathIOError):
        async with secure_io.open(root / "sub" / "evil" / "sentinel.txt", mode="rb"):
            pass
    assert sentinel.read_bytes() == b"secret"


@pytest.mark.asyncio
async def test_rename_source_token(secure_io, root):
    source = root / "file.txt"
    source.write_bytes(b"data")
    token = await secure_io.capture_rename_source(source)
    await secure_io.rename(token, root / "renamed.txt")
    assert (root / "renamed.txt").read_bytes() == b"data"


@pytest.mark.asyncio
async def test_rename_source_replacement_detected(secure_io, root, sentinel):
    source = root / "file.txt"
    source.write_bytes(b"data")
    token = await secure_io.capture_rename_source(source)
    # source object is replaced between RNFR and RNTO
    source.unlink()
    source.symlink_to(sentinel)
    with pytest.raises(aioftp.PathIOError):
        await secure_io.rename(token, root / "renamed.txt")
    assert sentinel.read_bytes() == b"secret"
    assert not (root / "renamed.txt").exists()
    # same for a replacement by a regular file
    source.unlink()
    source.write_bytes(b"data")
    token = await secure_io.capture_rename_source(source)
    source.unlink()
    source.write_bytes(b"another file with another inode")
    with pytest.raises(aioftp.PathIOError):
        await secure_io.rename(token, root / "renamed.txt")
    assert source.read_bytes() == b"another file with another inode"


@pytest.mark.asyncio
async def test_race_sentinel(secure_io, root, outside, sentinel):
    decoy = root / "real_dir"
    decoy.mkdir()
    (decoy / "sentinel.txt").write_bytes(b"public")
    (root / "regular.txt").write_bytes(b"public")
    link_dir = root / "dir"
    link_file = root / "file"
    stop = threading.Event()

    def swap_entry(path, targets):
        index = 0
        while not stop.is_set():
            with contextlib.suppress(OSError):
                if path.is_symlink() or path.is_file():
                    path.unlink()
                elif path.is_dir():
                    (path / "sentinel.txt").unlink(missing_ok=True)
                    path.rmdir()
                target = targets[index % len(targets)]
                if target is None:
                    path.mkdir()
                elif target != "missing":
                    os.symlink(target, path)
                index += 1

    swapper = threading.Thread(
        target=swap_entry,
        args=(link_dir, [outside, decoy, None, "missing"]),
        daemon=True,
    )
    file_swapper = threading.Thread(
        target=swap_entry,
        args=(link_file, [sentinel, root / "regular.txt", "missing"]),
        daemon=True,
    )
    swapper.start()
    file_swapper.start()
    try:
        for _ in range(300):
            for target in (link_dir / "sentinel.txt", link_file):
                with contextlib.suppress(aioftp.PathIOError):
                    async with secure_io.open(target, mode="rb") as file:
                        data = await file.read(64)
                        assert data != b"secret"
                with contextlib.suppress(aioftp.PathIOError):
                    async with secure_io.open(target, mode="wb") as file:
                        await file.write(b"x")
                with contextlib.suppress(aioftp.PathIOError):
                    await secure_io.unlink(target)
            await asyncio.sleep(0)
    finally:
        stop.set()
        swapper.join()
        file_swapper.join()
    assert sentinel.read_bytes() == b"secret"


@pytest.mark.asyncio
async def test_server_rename_flow(pair_factory, Server, root):
    (root / "file.txt").write_bytes(b"data")
    server = Server(
        [aioftp.User(base_path=root)],
        path_io_factory=aioftp.SecurePathIO,
    )
    async with pair_factory(None, server) as pair:
        await pair.client.rename("file.txt", "renamed.txt")
        assert await pair.client.exists("renamed.txt")
    assert (root / "renamed.txt").read_bytes() == b"data"


@pytest.mark.asyncio
async def test_server_rnfr_rnto_source_swap(
    pair_factory,
    Server,
    root,
    sentinel,
    expect_codes_in_exception,
):
    source = root / "file.txt"
    source.write_bytes(b"data")
    server = Server(
        [aioftp.User(base_path=root)],
        path_io_factory=aioftp.SecurePathIO,
    )
    async with pair_factory(None, server) as pair:
        await pair.client.command("RNFR file.txt", "350")
        # replace the source object between RNFR and RNTO
        source.unlink()
        source.symlink_to(sentinel)
        with expect_codes_in_exception("451"):
            await pair.client.command("RNTO renamed.txt", "250")
    assert sentinel.read_bytes() == b"secret"
    assert not (root / "renamed.txt").exists()
