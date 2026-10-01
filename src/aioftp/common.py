import abc
import asyncio
import collections
import functools
import locale
import socket
import ssl
import sys
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Iterable, MutableMapping
from contextlib import contextmanager
from typing import Any, Generic, Protocol, TypeVar, overload

if sys.version_info >= (3, 11):
    from typing import ParamSpec, Self
else:
    from typing_extensions import ParamSpec, Self


__all__ = (
    "with_timeout",
    "StreamIO",
    "Throttle",
    "StreamThrottle",
    "ThrottleStreamIO",
    "END_OF_LINE",
    "DEFAULT_BLOCK_SIZE",
    "LOCAL_LINE_ENDING",
    "NetworkToLocalNewlineTranslator",
    "LocalToNetworkNewlineTranslator",
    "wrap_with_container",
    "AsyncStreamIterator",
    "AbstractAsyncLister",
    "AsyncListerMixin",
    "async_enterable",
    "DEFAULT_PORT",
    "DEFAULT_USER",
    "DEFAULT_PASSWORD",
    "DEFAULT_ACCOUNT",
    "setlocale",
    "SSLSessionBoundContext",
    "Code",
    "wrap_into_codes",
    "Connection",
)

END_OF_LINE = "\r\n"
DEFAULT_BLOCK_SIZE = 8192
#: Newline convention used for *local* files during TYPE A (ASCII)
#: transfers. The wire representation is always NVT ASCII (CRLF, with bare
#: carriage returns escaped as ``CR NUL``); this constant defines what
#: line endings are converted to/from on the local side. Files are handled
#: as raw bytes, so this is independent of the host operating system.
LOCAL_LINE_ENDING = b"\n"

DEFAULT_PORT = 21
DEFAULT_USER = "anonymous"
DEFAULT_PASSWORD = "anon@"
DEFAULT_ACCOUNT = ""
HALF_OF_YEAR_IN_SECONDS = 15778476
TWO_YEARS_IN_SECONDS = ((365 * 3 + 366) * 24 * 60 * 60) / 2


def _now() -> float:
    return asyncio.get_running_loop().time()


class NetworkToLocalNewlineTranslator:
    """
    Streaming translator from NVT ASCII (the FTP TYPE A wire format) to
    the local newline convention (:data:`LOCAL_LINE_ENDING`).

    NVT uses ``CR LF`` for line ends and escapes a bare carriage return as
    ``CR NUL``. The translator is a pure byte state machine: feed it data in
    chunks of *any* size (even a single byte) and a carriage return split
    across two chunks is handled correctly. No text decoding is performed,
    so arbitrary byte content can never be corrupted by a multibyte
    character crossing a chunk boundary.

    On EOF, a trailing carriage return with neither ``LF`` nor ``NUL``
    following is delivered verbatim (a bare ``CR``), preserving the data
    instead of silently swallowing it.
    """

    __slots__ = ("_pending_cr",)

    def __init__(self) -> None:
        self._pending_cr = False

    def translate(self, data: bytes) -> bytes:
        """
        Translate a chunk of NVT ASCII bytes.

        :param data: bytes read from the network
        :rtype: :py:class:`bytes`
        """
        if not data:
            return b""
        out = bytearray()
        pending_cr = self._pending_cr
        for byte in data:
            if pending_cr:
                pending_cr = False
                if byte == 0x0A:
                    out += LOCAL_LINE_ENDING
                elif byte == 0x00:
                    out.append(0x0D)
                else:
                    out.append(0x0D)
                    if byte == 0x0D:
                        pending_cr = True
                    else:
                        out.append(byte)
            elif byte == 0x0D:
                pending_cr = True
            else:
                out.append(byte)
        self._pending_cr = pending_cr
        return bytes(out)

    def flush(self) -> bytes:
        """
        Emit any buffered state at end of stream.

        :rtype: :py:class:`bytes`
        """
        if self._pending_cr:
            self._pending_cr = False
            return b"\r"
        return b""


class LocalToNetworkNewlineTranslator:
    """
    Streaming translator from the local newline convention
    (:data:`LOCAL_LINE_ENDING`) to NVT ASCII (the FTP TYPE A wire format).

    Local line endings become ``CR LF`` and bare carriage returns (i.e.
    ``CR`` bytes not part of a local line ending) are escaped as ``CR NUL``
    to survive a round trip. Like :class:`NetworkToLocalNewlineTranslator`
    this is a byte state machine: a local line ending or a bare ``CR``
    split across two chunks is translated correctly without any decoding.
    """

    __slots__ = ("_pending_cr",)

    def __init__(self) -> None:
        self._pending_cr = False

    def translate(self, data: bytes) -> bytes:
        """
        Translate a chunk of local bytes to NVT ASCII.

        :param data: bytes read from a local file
        :rtype: :py:class:`bytes`
        """
        if not data:
            return b""
        out = bytearray()
        pending_cr = self._pending_cr
        for byte in data:
            if pending_cr:
                pending_cr = False
                if byte == 0x0A:
                    out += b"\r\n"
                else:
                    out += b"\r\x00"
                    if byte == 0x0D:
                        pending_cr = True
                    else:
                        out.append(byte)
            elif byte == 0x0D:
                pending_cr = True
            elif byte == 0x0A:
                out += b"\r\n"
            else:
                out.append(byte)
        self._pending_cr = pending_cr
        return bytes(out)

    def flush(self) -> bytes:
        """
        Emit any buffered state at end of stream.

        :rtype: :py:class:`bytes`
        """
        if self._pending_cr:
            self._pending_cr = False
            return b"\r\x00"
        return b""


WithTimeOutParamSpec = ParamSpec("WithTimeOutParamSpec")
WithTimeOutReturnType = TypeVar("WithTimeOutReturnType")


def _with_timeout(
    name: str,
) -> Callable[
    [Callable[WithTimeOutParamSpec, Awaitable[WithTimeOutReturnType]]],
    Callable[WithTimeOutParamSpec, Awaitable[WithTimeOutReturnType]],
]:
    def decorator(
        f: Callable[WithTimeOutParamSpec, Awaitable[WithTimeOutReturnType]],
    ) -> Callable[WithTimeOutParamSpec, Awaitable[WithTimeOutReturnType]]:
        @functools.wraps(f)
        async def wrapper(
            *args: WithTimeOutParamSpec.args,
            **kwargs: WithTimeOutParamSpec.kwargs,
        ) -> WithTimeOutReturnType:
            cls = args[0]
            timeout = getattr(cls, name)
            return await asyncio.wait_for(f(*args, **kwargs), timeout)

        return wrapper

    return decorator


@overload
def with_timeout(
    name: str,
) -> Callable[
    [Callable[WithTimeOutParamSpec, Awaitable[WithTimeOutReturnType]]],
    Callable[WithTimeOutParamSpec, Awaitable[WithTimeOutReturnType]],
]: ...


@overload
def with_timeout(
    name: Callable[WithTimeOutParamSpec, Awaitable[WithTimeOutReturnType]],
) -> Callable[WithTimeOutParamSpec, Awaitable[WithTimeOutReturnType]]: ...


def with_timeout(
    name: str | Callable[WithTimeOutParamSpec, Awaitable[WithTimeOutReturnType]],
) -> (
    Callable[
        [Callable[WithTimeOutParamSpec, Awaitable[WithTimeOutReturnType]]],
        Callable[WithTimeOutParamSpec, Awaitable[WithTimeOutReturnType]],
    ]
    | Callable[WithTimeOutParamSpec, Awaitable[WithTimeOutReturnType]]
):
    """
    Method decorator, wraps method with :py:func:`asyncio.wait_for`. `timeout`
    argument takes from `name` decorator argument or "timeout".

    :param name: name of timeout attribute
    :type name: :py:class:`str`

    :raises asyncio.TimeoutError: if coroutine does not finished in timeout

    Wait for `self.timeout`
    ::

        >>> def __init__(self, ...):
        ...
        ...     self.timeout = 1
        ...
        ... @with_timeout
        ... async def foo(self, ...):
        ...
        ...     pass

    Wait for custom timeout
    ::

        >>> def __init__(self, ...):
        ...
        ...     self.foo_timeout = 1
        ...
        ... @with_timeout("foo_timeout")
        ... async def foo(self, ...):
        ...
        ...     pass

    """

    if isinstance(name, str):
        return _with_timeout(name)
    else:
        return _with_timeout("timeout")(name)


class AsyncStreamIterator:
    def __init__(self, read_coro: Callable[[], Awaitable[bytes]]) -> None:
        self.read_coro = read_coro

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> bytes:
        data = await self.read_coro()
        if data:
            return data
        else:
            raise StopAsyncIteration


IterableType = TypeVar("IterableType")


class AsyncListerMixin(abc.ABC, Generic[IterableType]):
    """
    Add ability to `async for` context to collect data to list via await.

    ::

        >>> class Context(AsyncListerMixin):
        ...     ...
        >>> results = await Context(...)
    """

    async def _to_list(self) -> list[IterableType]:
        items = []
        async for item in self:
            items.append(item)
        return items

    def __await__(self) -> Generator[None, None, list[IterableType]]:
        return self._to_list().__await__()

    @abc.abstractmethod
    def __aiter__(self) -> AsyncIterator[IterableType]:
        """
        Must be implemented by subclasses to enable iteration using ``async for``.
        :rtype: :py:class:`typing.AsyncIterator`[:py:class:`IterableType`]
        """


class AbstractAsyncLister(AsyncListerMixin[IterableType], abc.ABC):
    """
    Abstract context with ability to collect all iterables into
    :py:class:`list` via `await` with optional timeout (via
    :py:func:`aioftp.with_timeout`)

    :param timeout: timeout for __anext__ operation
    :type timeout: :py:class:`None`, :py:class:`int` or :py:class:`float`

    ::

        >>> class Lister(AbstractAsyncLister):
        ...
        ...     @with_timeout
        ...     async def __anext__(self):
        ...         ...

    ::

        >>> async for block in Lister(...):
        ...     ...

    ::

        >>> result = await Lister(...)
        >>> result
        [block, block, block, ...]
    """

    def __init__(self, *, timeout: float | int | None = None) -> None:
        super().__init__()
        self.timeout = timeout

    def __aiter__(self) -> Self:
        return self

    @with_timeout
    @abc.abstractmethod
    async def __anext__(self) -> IterableType:
        """
        :py:func:`asyncio.coroutine`

        Abstract method
        """


class AsyncContextManager(Protocol):
    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, *args: Any, **kwargs: Any) -> None: ...


AsyncEnterableReturnType = TypeVar("AsyncEnterableReturnType", bound=AsyncContextManager, covariant=True)


class AsyncEnterableInstanceProtocol(Protocol[AsyncEnterableReturnType]):
    async def __aenter__(self) -> AsyncEnterableReturnType: ...
    async def __aexit__(self, *args: Any, **kwargs: Any) -> None: ...
    def __await__(self) -> Generator[None, None, AsyncEnterableReturnType]: ...


AsyncEnterableParamSpec = ParamSpec("AsyncEnterableParamSpec")


def async_enterable(
    f: Callable[AsyncEnterableParamSpec, Awaitable[AsyncEnterableReturnType]],
) -> Callable[AsyncEnterableParamSpec, AsyncEnterableInstanceProtocol[AsyncEnterableReturnType]]:
    """
    Decorator. Bring coroutine result up, so it can be used as async context

    ::

        >>> async def foo():
        ...
        ...     ...
        ...     return AsyncContextInstance(...)
        ...
        ... ctx = await foo()
        ... async with ctx:
        ...
        ...     # do

    ::

        >>> @async_enterable
        ... async def foo():
        ...
        ...     ...
        ...     return AsyncContextInstance(...)
        ...
        ... async with foo() as ctx:
        ...
        ...     # do
        ...
        ... ctx = await foo()
        ... async with ctx:
        ...
        ...     # do

    """

    @functools.wraps(f)
    def wrapper(
        *args: AsyncEnterableParamSpec.args,
        **kwargs: AsyncEnterableParamSpec.kwargs,
    ) -> AsyncEnterableInstanceProtocol[AsyncEnterableReturnType]:
        class AsyncEnterableInstance:
            async def __aenter__(self) -> AsyncEnterableReturnType:
                self.context = await f(*args, **kwargs)
                return await self.context.__aenter__()

            async def __aexit__(self, *args: Any, **kwargs: Any) -> None:
                await self.context.__aexit__(*args, **kwargs)

            def __await__(self) -> Generator[None, None, AsyncEnterableReturnType]:
                return f(*args, **kwargs).__await__()

        return AsyncEnterableInstance()

    return wrapper


class Code(str):
    """
    Representation of server status code.
    """

    def matches(self, mask: str) -> bool:
        """
        :param mask: Template for comparision. If mask symbol is not digit
            then it passes.
        :type mask: :py:class:`str`

        ::

            >>> Code("123").matches("1")
            True
            >>> Code("123").matches("1x3")
            True
        """
        return all(map(lambda m, c: not m.isdigit() or m == c, mask, self))


StrType = TypeVar("StrType", bound=str)
NotStrType = TypeVar("NotStrType")  # typing doesn't support ~T to exclude all subclasses of str


@overload
def wrap_with_container(o: StrType) -> tuple[StrType]: ...
@overload
def wrap_with_container(o: NotStrType) -> NotStrType: ...


def wrap_with_container(o: StrType | NotStrType) -> tuple[StrType] | NotStrType:
    if isinstance(o, str):
        return (o,)  # type: ignore[return-value]
    return o


def wrap_into_codes(o: Iterable[str]) -> tuple[Code, ...]:
    return tuple(Code(item) for item in o if not isinstance(item, Code))


class StreamIO:
    """
    Stream input/output wrapper with timeout.

    :param reader: stream reader
    :type reader: :py:class:`asyncio.StreamReader`

    :param writer: stream writer
    :type writer: :py:class:`asyncio.StreamWriter`

    :param timeout: socket timeout for read/write operations
    :type timeout: :py:class:`int`, :py:class:`float` or :py:class:`None`

    :param read_timeout: socket timeout for read operations, overrides
        `timeout`
    :type read_timeout: :py:class:`int`, :py:class:`float` or :py:class:`None`

    :param write_timeout: socket timeout for write operations, overrides
        `timeout`
    :type write_timeout: :py:class:`int`, :py:class:`float` or :py:class:`None`
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        timeout: float | int | None = None,
        read_timeout: float | int | None = None,
        write_timeout: float | int | None = None,
    ):
        self.reader = reader
        self.writer = writer
        self.read_timeout = read_timeout or timeout
        self.write_timeout = write_timeout or timeout

    @with_timeout("read_timeout")
    async def readline(self) -> bytes:
        """
        :py:func:`asyncio.coroutine`

        Proxy for :py:meth:`asyncio.StreamReader.readline`.
        """
        return await self.reader.readline()

    @with_timeout("read_timeout")
    async def read(self, count: int = -1) -> bytes:
        """
        :py:func:`asyncio.coroutine`

        Proxy for :py:meth:`asyncio.StreamReader.read`.

        :param count: block size for read operation
        :type count: :py:class:`int`
        """
        return await self.reader.read(count)

    @with_timeout("read_timeout")
    async def readexactly(self, count: int) -> bytes:
        """
        :py:func:`asyncio.coroutine`

        Proxy for :py:meth:`asyncio.StreamReader.readexactly`.

        :param count: block size for read operation
        :type count: :py:class:`int`
        """
        return await self.reader.readexactly(count)

    @with_timeout("write_timeout")
    async def write(self, data: bytes) -> None:
        """
        :py:func:`asyncio.coroutine`

        Combination of :py:meth:`asyncio.StreamWriter.write` and
        :py:meth:`asyncio.StreamWriter.drain`.

        :param data: data to write
        :type data: :py:class:`bytes`
        """
        self.writer.write(data)
        await self.writer.drain()

    def close(self) -> None:
        """
        Close connection.
        """
        self.writer.close()

    async def start_tls(self, sslcontext: ssl.SSLContext, server_hostname: str | None) -> None:
        """
        Upgrades the connection to TLS
        """
        await self.writer.start_tls(  # type: ignore[attr-defined,unused-ignore]  # py3.12+
            sslcontext=sslcontext,
            server_hostname=server_hostname,
            ssl_handshake_timeout=self.write_timeout,
        )


class Throttle:
    """
    Throttle for streams.

    :param limit: speed limit in bytes or :py:class:`None` for unlimited
    :type limit: :py:class:`int` or :py:class:`None`

    :param reset_rate: time in seconds for «round» throttle memory (to deal
        with float precision when divide)
    :type reset_rate: :py:class:`int` or :py:class:`float`
    """

    def __init__(self, *, limit: int | None = None, reset_rate: float | int = 10) -> None:
        self._limit = limit
        self.reset_rate = reset_rate
        self._start: float | None = None
        self._sum = 0

    async def wait(self) -> None:
        """
        :py:func:`asyncio.coroutine`

        Wait until can do IO
        """
        if self._limit is not None and self._limit > 0 and self._start is not None:
            now = _now()
            end = self._start + self._sum / self._limit
            await asyncio.sleep(max(0, end - now))

    def append(self, data: bytes, start: float) -> None:
        """
        Count `data` for throttle

        :param data: bytes of data for count
        :type data: :py:class:`bytes`

        :param start: start of read/write time from
            :py:meth:`asyncio.BaseEventLoop.time`
        :type start: :py:class:`float`
        """
        if self._limit is not None and self._limit > 0:
            if self._start is None:
                self._start = start
            if start - self._start > self.reset_rate:
                self._sum -= round((start - self._start) * self._limit)
                self._start = start
            self._sum += len(data)

    @property
    def limit(self) -> int | None:
        """
        Throttle limit
        """
        return self._limit

    @limit.setter
    def limit(self, value: int | None) -> None:
        """
        Set throttle limit

        :param value: bytes per second
        :type value: :py:class:`int` or :py:class:`None`
        """
        self._limit = value
        self._start = None
        self._sum = 0

    def clone(self) -> "Throttle":
        """
        Clone throttle without memory
        """
        return Throttle(limit=self._limit, reset_rate=self.reset_rate)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(limit={self._limit!r}, reset_rate={self.reset_rate!r})"


class StreamThrottle(collections.namedtuple("StreamThrottle", "read write")):
    """
    Stream throttle with `read` and `write` :py:class:`aioftp.Throttle`

    :param read: stream read throttle
    :type read: :py:class:`aioftp.Throttle`

    :param write: stream write throttle
    :type write: :py:class:`aioftp.Throttle`
    """

    def clone(self) -> "StreamThrottle":
        """
        Clone throttles without memory
        """
        return StreamThrottle(
            read=self.read.clone(),
            write=self.write.clone(),
        )

    @classmethod
    def from_limits(
        cls,
        read_speed_limit: int | None = None,
        write_speed_limit: int | None = None,
    ) -> "StreamThrottle":
        """
        Simple wrapper for creation :py:class:`aioftp.StreamThrottle`

        :param read_speed_limit: stream read speed limit in bytes or
            :py:class:`None` for unlimited
        :type read_speed_limit: :py:class:`int` or :py:class:`None`

        :param write_speed_limit: stream write speed limit in bytes or
            :py:class:`None` for unlimited
        :type write_speed_limit: :py:class:`int` or :py:class:`None`
        """
        return cls(
            read=Throttle(limit=read_speed_limit),
            write=Throttle(limit=write_speed_limit),
        )


class ThrottleStreamIO(StreamIO):
    """
    Throttled :py:class:`aioftp.StreamIO`. `ThrottleStreamIO` is subclass of
    :py:class:`aioftp.StreamIO`. `throttles` attribute is dictionary of `name`:
    :py:class:`aioftp.StreamThrottle` pairs

    :param *args: positional arguments for :py:class:`aioftp.StreamIO`
    :param **kwargs: keyword arguments for :py:class:`aioftp.StreamIO`

    :param throttles: dictionary of throttles
    :type throttles: :py:class:`dict` with :py:class:`aioftp.Throttle` values

    ::

        >>> self.stream = ThrottleStreamIO(
        ...     reader,
        ...     writer,
        ...     throttles={
        ...         "main": StreamThrottle(
        ...             read=Throttle(...),
        ...             write=Throttle(...)
        ...         )
        ...     },
        ...     timeout=timeout
        ... )
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        throttles: dict[str, StreamThrottle] = {},
        *,
        timeout: float | int | None = None,
        read_timeout: float | int | None = None,
        write_timeout: float | int | None = None,
        read_translator: NetworkToLocalNewlineTranslator | None = None,
        write_translator: LocalToNetworkNewlineTranslator | None = None,
    ):
        super().__init__(reader, writer, timeout=timeout, read_timeout=read_timeout, write_timeout=write_timeout)
        self.throttles = throttles
        # When set, bytes read from the wire pass through `read_translator`
        # and bytes written to the wire pass through `write_translator`.
        # This is how TYPE A (ASCII) transfers are streamed: translation
        # happens per byte and never via "decode each chunk and join", which
        # would corrupt data split across chunk boundaries.
        self.read_translator = read_translator
        self.write_translator = write_translator
        self._read_buffer = bytearray()
        self._read_eof = False

    async def wait(self, name: str) -> None:
        """
        :py:func:`asyncio.coroutine`

        Wait for all throttles

        :param name: name of throttle to acquire ("read" or "write")
        :type name: :py:class:`str`
        """
        tasks = []
        for throttle in self.throttles.values():
            curr_throttle = getattr(throttle, name)
            if curr_throttle.limit:
                tasks.append(asyncio.create_task(curr_throttle.wait()))
        if tasks:
            await asyncio.wait(tasks)

    def append(self, name: str, data: bytes, start: float) -> None:
        """
        Update timeout for all throttles

        :param name: name of throttle to append to ("read" or "write")
        :type name: :py:class:`str`

        :param data: bytes of data for count
        :type data: :py:class:`bytes`

        :param start: start of read/write time from
            :py:meth:`asyncio.BaseEventLoop.time`
        :type start: :py:class:`float`
        """
        for throttle in self.throttles.values():
            getattr(throttle, name).append(data, start)

    async def read(self, count: int = -1) -> bytes:
        """
        :py:func:`asyncio.coroutine`

        :py:meth:`aioftp.StreamIO.read` proxy. When a read translator is
        attached (TYPE A downloads), translation is streamed across reads:
        translation never happens by joining separately decoded chunks, so
        a carriage return (or any byte) landing on a chunk boundary is
        handled. `count` limits the number of *translated* bytes returned;
        underlying reads are repeated until enough output is available or
        the wire reaches EOF, at which point the translator is flushed
        (delivering a possible trailing bare ``CR``).
        """
        if self.read_translator is None:
            await self.wait("read")
            start = _now()
            data = await super().read(count)
            self.append("read", data, start)
            return data
        if not self._read_eof and (count < 0 or len(self._read_buffer) < count):
            while not self._read_eof and (count < 0 or len(self._read_buffer) < count):
                await self.wait("read")
                start = _now()
                raw = await StreamIO.read(self, count)
                self.append("read", raw, start)
                if raw:
                    self._read_buffer += self.read_translator.translate(raw)
                else:
                    self._read_buffer += self.read_translator.flush()
                    self._read_eof = True
        if count < 0:
            data = bytes(self._read_buffer)
            self._read_buffer.clear()
        else:
            data = bytes(self._read_buffer[:count])
            del self._read_buffer[:count]
        return data

    async def readline(self) -> bytes:
        """
        :py:func:`asyncio.coroutine`

        :py:meth:`aioftp.StreamIO.readline` proxy. With a read translator
        attached the line is read byte by byte so that line boundaries are
        defined by the *translated* stream rather than the wire bytes.
        """
        if self.read_translator is not None:
            out = bytearray()
            while True:
                chunk = await self.read(1)
                if not chunk:
                    break
                out += chunk
                if chunk == b"\n":
                    break
            return bytes(out)
        await self.wait("read")
        start = _now()
        data = await super().readline()
        self.append("read", data, start)
        return data

    async def write(self, data: bytes) -> None:
        """
        :py:func:`asyncio.coroutine`

        :py:meth:`aioftp.StreamIO.write` proxy. When a write translator is
        attached (TYPE A uploads), `data` is streamed through the
        translator before reaching the wire. Call :meth:`flush_writes`
        after the final write to emit buffered state (a bare trailing
        ``CR`` becomes ``CR NUL``).
        """
        if self.write_translator is not None:
            data = self.write_translator.translate(data)
        await self.wait("write")
        start = _now()
        await super().write(data)
        self.append("write", data, start)

    async def flush_writes(self) -> None:
        """
        :py:func:`asyncio.coroutine`

        Flush the write translator (TYPE A uploads), emitting any bytes
        buffered by a chunk-boundary state. The translator output is
        already wire data, so it bypasses translation (unlike
        :meth:`write`) but still passes through the throttles. No-op
        without a translator.
        """
        if self.write_translator is None:
            return
        tail = self.write_translator.flush()
        if tail:
            await self.wait("write")
            start = _now()
            await StreamIO.write(self, tail)
            self.append("write", tail, start)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.close()

    def iter_by_line(self) -> AsyncStreamIterator:
        """
        Read/iterate stream by line.

        :rtype: :py:class:`aioftp.AsyncStreamIterator`

        ::

            >>> async for line in stream.iter_by_line():
            ...     ...
        """
        return AsyncStreamIterator(self.readline)

    def iter_by_block(self, count: int = DEFAULT_BLOCK_SIZE) -> AsyncStreamIterator:
        """
        Read/iterate stream by block.

        :rtype: :py:class:`aioftp.AsyncStreamIterator`

        ::

            >>> async for block in stream.iter_by_block(block_size):
            ...     ...
        """
        return AsyncStreamIterator(lambda: self.read(count))


LOCALE_LOCK = threading.Lock()


@contextmanager
def setlocale(name: str) -> Generator[str, None, None]:
    """
    Context manager with threading lock for set locale on enter, and set it
    back to original state on exit.

    ::

        >>> with setlocale("C"):
        ...     ...
    """
    with LOCALE_LOCK:
        old_locale = locale.setlocale(locale.LC_ALL)
        try:
            yield locale.setlocale(locale.LC_ALL, name)
        finally:
            locale.setlocale(locale.LC_ALL, old_locale)


# class from https://github.com/python/cpython/issues/79152 (with some changes)
class SSLSessionBoundContext(ssl.SSLContext):
    """ssl.SSLContext bound to an existing SSL session.

    Actually asyncio doesn't support TLS session resumption, the loop.create_connection() API
    does not take any TLS session related argument. There is ongoing work to add support for this
    at https://github.com/python/cpython/issues/79152.

    The loop.create_connection() API takes a SSL context argument though, the SSLSessionBoundContext
    is used to wrap a SSL context and inject a SSL session on calls to
        - SSLSessionBoundContext.wrap_socket()
        - SSLSessionBoundContext.wrap_bio()

    This wrapper is compatible with any TLS application which calls only the methods above when
    making new TLS connections. This class is NOT a subclass of ssl.SSLContext, so it will be
    rejected by applications which ensure the SSL context is an instance of ssl.SSLContext. Not being
    a subclass of ssl.SSLContext makes this wrapper lightweight.
    """

    __slots__ = ("context", "session")

    def __init__(self, protocol: int, context: ssl.SSLContext, session: ssl.SSLSession):
        self.context = context
        self.session = session

    def wrap_socket(
        self,
        sock: socket.socket,
        server_side: bool = False,
        do_handshake_on_connect: bool = True,
        suppress_ragged_eofs: bool = True,
        server_hostname: str | bytes | None = None,
        session: ssl.SSLSession | None = None,
    ) -> ssl.SSLSocket:
        if session is not None:
            raise ValueError("expected session to be None")
        return self.context.wrap_socket(
            sock=sock,
            server_hostname=server_hostname,
            server_side=server_side,
            do_handshake_on_connect=do_handshake_on_connect,
            suppress_ragged_eofs=suppress_ragged_eofs,
            session=self.session,
        )

    def wrap_bio(
        self,
        incoming: ssl.MemoryBIO,
        outgoing: ssl.MemoryBIO,
        server_side: bool = False,
        server_hostname: str | bytes | None = None,
        session: ssl.SSLSession | None = None,
    ) -> ssl.SSLObject:
        if session is not None:
            raise ValueError("expected session to be None")
        return self.context.wrap_bio(
            incoming=incoming,
            outgoing=outgoing,
            server_hostname=server_hostname,
            server_side=server_side,
            session=self.session,
        )


class Connection(collections.defaultdict[str, asyncio.Future[Any]]):
    """
    Connection state container for transparent work with futures for async
    wait

    :param kwargs: initialization parameters

    Container based on :py:class:`collections.defaultdict`, which holds
    :py:class:`asyncio.Future` as default factory. There is two layers of
    abstraction:

    * Low level based on simple dictionary keys to attributes mapping and
        available at Connection.future.
    * High level based on futures result and dictionary keys to attributes
        mapping and available at Connection.

    To clarify, here is groups of equal expressions
    ::

        >>> connection.future.foo
        >>> connection["foo"]

        >>> connection.foo
        >>> connection["foo"].result()

        >>> del connection.future.foo
        >>> del connection.foo
        >>> del connection["foo"]
    """

    __slots__ = ("future",)

    class Container:
        def __init__(self, storage: MutableMapping[str, asyncio.Future[Any]]) -> None:
            self.storage = storage

        def __getattr__(self, name: str) -> asyncio.Future[Any]:
            return self.storage[name]

        def __delattr__(self, name: str) -> None:
            self.storage.pop(name)

    future: "Connection.Container"
    default_factory: Callable[[], asyncio.Future[Any]]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(asyncio.Future)
        self.future = Connection.Container(self)
        for k, v in kwargs.items():
            self[k].set_result(v)

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name].result()
        raise AttributeError(f"{name!r} not in storage")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in Connection.__slots__:
            super().__setattr__(name, value)
        else:
            if self[name].done():
                self[name] = self.default_factory()
            self[name].set_result(value)

    def __delattr__(self, name: str) -> None:
        if name in self:
            self.pop(name)
