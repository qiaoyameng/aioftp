import pathlib

import pytest

import aioftp


@pytest.mark.asyncio
async def test_feat_advertises_mlst_facts_and_defaults(pair_factory):
    async with pair_factory() as pair:
        info = await pair.client.feat()
        mlst_lines = [line for line in info if "mlst" in line.lower()]
        assert len(mlst_lines) == 1
        line = mlst_lines[0]
        for fact in ("Size", "Create", "Modify", "Type"):
            assert f"{fact}*;" in line


@pytest.mark.asyncio
async def test_opts_mlst_selection_shared_by_mlst_and_mlsd(pair_factory):
    async with pair_factory() as pair:
        await pair.make_server_files("foo")
        await pair.client.set_mlst_facts(["modify", "type"])
        info = await pair.client.stat("foo")
        assert set(info) == {"modify", "type"}
        [(path, info)] = await pair.client.list()
        assert path == pathlib.PurePosixPath("foo")
        assert set(info) == {"modify", "type"}


@pytest.mark.asyncio
async def test_opts_mlst_unknown_fact_rejected_and_selection_kept(
    pair_factory,
    expect_codes_in_exception,
):
    async with pair_factory() as pair:
        await pair.make_server_files("foo")
        with expect_codes_in_exception("501"):
            await pair.client.set_mlst_facts(["size", "nonsense"])
        info = await pair.client.stat("foo")
        assert {"size", "create", "modify", "type"} <= set(info)


@pytest.mark.asyncio
async def test_opts_mlst_empty_selection(pair_factory):
    async with pair_factory() as pair:
        await pair.make_server_files("foo")
        await pair.client.set_mlst_facts()
        info = await pair.client.stat("foo")
        assert info == {}
        [(path, info)] = await pair.client.list()
        assert path == pathlib.PurePosixPath("foo")
        assert info == {}


@pytest.mark.asyncio
async def test_opts_mlst_case_insensitive_and_duplicates(pair_factory):
    async with pair_factory() as pair:
        await pair.make_server_files("foo")
        await pair.client.set_mlst_facts(["sIzE", "SIZE", "modify"])
        code, info = await pair.client.command("MLST foo", "2xx")
        line = info[1]
        assert line.count("Size=") == 1
        assert line.count("Modify=") == 1
        assert "SIZE=" not in line
        assert "Type=" not in line


@pytest.mark.asyncio
async def test_opts_mlst_selection_is_per_connection(pair_factory):
    async with pair_factory() as pair:
        await pair.make_server_files("foo")
        other = aioftp.Client()
        await other.connect(
            pair.server.server_host,
            pair.server.server_port,
        )
        await other.login()
        try:
            await pair.client.set_mlst_facts(["size"])
            await other.set_mlst_facts(["type"])
            first = await pair.client.stat("foo")
            second = await other.stat("foo")
            assert set(first) == {"size"}
            assert set(second) == {"type"}
        finally:
            await other.quit()
            other.close()


@pytest.mark.asyncio
async def test_opts_mlst_selection_reset_on_relogin(pair_factory):
    async with pair_factory() as pair:
        await pair.make_server_files("foo")
        await pair.client.set_mlst_facts(["size"])
        info = await pair.client.stat("foo")
        assert set(info) == {"size"}
        await pair.client.login()
        info = await pair.client.stat("foo")
        assert {"size", "create", "modify", "type"} <= set(info)


@pytest.mark.asyncio
async def test_opts_mlst_with_real_path_io(pair_factory, Server, tmp_path):
    (tmp_path / "foo.txt").write_bytes(b"hello")
    s = Server(
        [aioftp.User(base_path=tmp_path)],
        path_io_factory=aioftp.PathIO,
    )
    async with pair_factory(None, s) as pair:
        await pair.client.set_mlst_facts(["size", "type"])
        info = await pair.client.stat("foo.txt")
        assert set(info) == {"size", "type"}
        assert info["size"] == "5"
        assert info["type"] == "file"
        [(path, info)] = await pair.client.list()
        assert path == pathlib.PurePosixPath("foo.txt")
        assert set(info) == {"size", "type"}
        assert info["size"] == "5"


@pytest.mark.asyncio
async def test_mlsx_unavailable_facts_not_fabricated(pair_factory, Server):
    class NoStatPathIO(aioftp.MemoryPathIO):
        @aioftp.pathio.universal_exception
        async def stat(self, path):
            raise PermissionError("stat is not allowed")

    s = Server(path_io_factory=NoStatPathIO)
    async with pair_factory(None, s) as pair:
        await pair.make_server_files("foo")
        info = await pair.client.stat("foo")
        assert "size" not in info
        assert "create" not in info
        assert "modify" not in info
        assert info["type"] == "file"
        [(path, info)] = await pair.client.list()
        assert "size" not in info
        assert info["type"] == "file"
