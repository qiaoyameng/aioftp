import types

import pytest

import aioftp


@pytest.fixture(params=["memory", "real"])
def any_path_io_server(request, Server, tmp_path):
    if request.param == "real":
        return Server(
            users=[aioftp.User(base_path=tmp_path)],
            path_io_factory=aioftp.PathIO,
        )
    return Server(path_io_factory=aioftp.MemoryPathIO)


@pytest.mark.asyncio
async def test_feat_advertises_mlst_facts(pair_factory, any_path_io_server):
    async with pair_factory(None, any_path_io_server) as pair:
        features = await pair.client.feat()
        assert "MLST Size*;Create*;Modify*;Type*;" in features


@pytest.mark.asyncio
async def test_opts_mlst_selects_facts_for_mlst_and_mlsd(
    pair_factory,
    any_path_io_server,
):
    async with pair_factory(None, any_path_io_server) as pair:
        await pair.make_server_files("foo")
        active = await pair.client.set_mlst_facts(["Size", "Modify"])
        assert active == ["Size", "Modify"]
        info = await pair.client.stat("foo")
        assert set(info) == {"size", "modify"}
        [(name, listed)] = await pair.client.list()
        assert name.name == "foo"
        assert set(listed) == {"size", "modify"}


@pytest.mark.asyncio
async def test_opts_mlst_unknown_fact_rejected_and_keeps_selection(
    pair_factory,
    any_path_io_server,
    expect_codes_in_exception,
):
    async with pair_factory(None, any_path_io_server) as pair:
        await pair.make_server_files("foo")
        await pair.client.set_mlst_facts(["Size"])
        with expect_codes_in_exception("501"):
            await pair.client.set_mlst_facts(["Size", "Nope"])
        info = await pair.client.stat("foo")
        assert set(info) == {"size"}


@pytest.mark.asyncio
async def test_opts_mlst_empty_selection_resets_to_defaults(
    pair_factory,
    any_path_io_server,
):
    async with pair_factory(None, any_path_io_server) as pair:
        await pair.make_server_files("foo")
        await pair.client.set_mlst_facts(["Size"])
        active = await pair.client.set_mlst_facts()
        assert active == ["Size", "Create", "Modify", "Type"]
        info = await pair.client.stat("foo")
        assert set(info) == {"size", "create", "modify", "type"}


@pytest.mark.asyncio
async def test_opts_mlst_case_insensitive_and_deduplicated(
    pair_factory,
    any_path_io_server,
):
    async with pair_factory(None, any_path_io_server) as pair:
        await pair.make_server_files("foo")
        active = await pair.client.set_mlst_facts(["size", "MODIFY", "Size"])
        assert active == ["Size", "Modify"]
        info = await pair.client.stat("foo")
        assert set(info) == {"size", "modify"}


@pytest.mark.asyncio
async def test_opts_mlst_selection_is_per_connection(
    pair_factory,
    any_path_io_server,
):
    async with pair_factory(None, any_path_io_server) as pair:
        await pair.make_server_files("foo")
        other = aioftp.Client()
        await other.connect(pair.server.server_host, pair.server.server_port)
        await other.login()
        try:
            await pair.client.set_mlst_facts(["Size"])
            await other.set_mlst_facts(["Modify"])
            assert set(await pair.client.stat("foo")) == {"size"}
            assert set(await other.stat("foo")) == {"modify"}
        finally:
            await other.quit()
            other.close()


@pytest.mark.asyncio
async def test_opts_mlst_selection_reset_on_relogin(
    pair_factory,
    any_path_io_server,
):
    async with pair_factory(None, any_path_io_server) as pair:
        await pair.make_server_files("foo")
        await pair.client.set_mlst_facts(["Size"])
        await pair.client.login()
        info = await pair.client.stat("foo")
        assert set(info) == {"size", "create", "modify", "type"}


@pytest.mark.asyncio
async def test_mlst_unavailable_facts_not_fabricated(pair_factory, Server):
    class PartialStatsPathIO(aioftp.MemoryPathIO):
        @aioftp.pathio.universal_exception
        async def stat(self, path):
            stats = await super().stat(path)
            return types.SimpleNamespace(
                st_size=stats.st_size,
                st_mtime=None,
            )

    s = Server(path_io_factory=PartialStatsPathIO)
    async with pair_factory(None, s) as pair:
        await pair.make_server_files("foo")
        info = await pair.client.stat("foo")
        assert "size" in info
        assert "modify" not in info
        assert "create" not in info


@pytest.mark.asyncio
async def test_mlst_permission_denied_not_fabricated(
    pair_factory,
    Server,
    expect_codes_in_exception,
):
    s = Server(
        [
            aioftp.User(
                permissions=[
                    aioftp.Permission("/"),
                    aioftp.Permission("/foo", readable=False),
                ],
            ),
        ],
    )
    async with pair_factory(None, s) as pair:
        await pair.make_server_files("foo")
        with expect_codes_in_exception("550"):
            await pair.client.command("MLST /foo", "2xx")
