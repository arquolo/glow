__all__ = ['Uid']

import math
import re
import string
from functools import lru_cache
from typing import Self, SupportsInt
from uuid import UUID, uuid4

ALPHABET = string.digits + string.ascii_letters
ALPHABET = ''.join(sorted({*ALPHABET} - {*'0O1Il'}))

_BASE = len(ALPHABET)  # 57
_LEN = math.ceil(128 / math.log2(_BASE))  # 22

_TABLE = ALPHABET.encode('ascii').ljust(256, b'\0')
_NUMBERS = {s: i for i, s in enumerate(ALPHABET)}
_REGEX = re.compile(f'^[{ALPHABET}]{{{_LEN}}}$')


@lru_cache  # Small performance optimization
def base57_encode(number: int) -> str:
    out = bytearray(_LEN)
    for i in range(_LEN - 1, -1, -1):
        number, out[i] = divmod(number, _BASE)
    return out.translate(_TABLE).decode('ascii')


@lru_cache  # Small performance optimization
def base57_decode(shortuuid: str) -> int:
    if not _REGEX.fullmatch(shortuuid):
        msg = 'invalid shortuuid format'
        raise ValueError(msg)
    out = 0
    for char in shortuuid:
        out = out * _BASE + _NUMBERS[char]
    return out


class Uid(UUID):
    """Subclass of UUID with support of short-uuid serialization format.

    Uses base57 instead of hex for serialization.

    base57 uses lowercase and uppercase letters and digits,
    excluding similar-looking characters such as l, 1, I, O and 0,
    and it doesn't use URL-unsafe +, /, = characters (opposed to base64).

    UUIDs encoded with base57 have length of 22 characters, while
    with hex (default) - 32 characters.

    Uid can be created directly from UUID:

        >>> u = UUID('3b1f8b40-222c-4a6e-b77e-779d5a94e21c')
        >>> Uid(u)
        Uid('CXc85b4rqinB7s5J52TRYb')
        >>> str(Uid(u))
        'CXc85b4rqinB7s5J52TRYb'

    Or from string representation of short-uuid:

        >>> Uid('CXc85b4rqinB7s5J52TRYb')
        Uid('CXc85b4rqinB7s5J52TRYb')

    Simplified and more optimized (2-3x faster on average) fork of
    [shortuuid](https://github.com/skorokithakis/shortuuid)
    """

    def __init__(self, obj: str | SupportsInt) -> None:
        """Create Uid from str (parsing it as short-uuid) or int-compatible."""
        if not isinstance(obj, str | SupportsInt):
            msg = f'Either int, string or UUID required. Got {type(obj)}'
            raise TypeError(msg)

        value = base57_decode(obj) if isinstance(obj, str) else int(obj)
        super().__init__(int=value)

    def __str__(self) -> str:
        return base57_encode(int(self))

    @classmethod  # Pydantic 1.x requirement
    def __get_validators__(cls):
        yield cls

    @classmethod  # Pydantic 2.x requirement
    def __get_pydantic_core_schema__(cls, _, handler):
        from pydantic_core import core_schema

        return core_schema.no_info_after_validator_function(
            cls,
            handler(str | UUID | int),
            serialization=core_schema.plain_serializer_function_ser_schema(
                str, info_arg=False, return_schema=core_schema.str_schema()
            ),
        )

    @classmethod  # Pydantic 1.x requirement for OpenAPI
    def __modify_schema__(cls, field_schema: dict) -> None:
        field_schema.update(
            examples=[str(cls.v4()) for _ in range(2)],
            type='string',
            format=None,
            pattern=_REGEX.pattern,
        )

    @classmethod  # Pydantic 2.x requirement for OpenAPI
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        field_schema = handler(core_schema)
        field_schema = handler.resolve_ref_schema(field_schema)
        field_schema.pop('anyOf', None)
        cls.__modify_schema__(field_schema)
        return field_schema

    @classmethod
    def v4(cls) -> Self:
        """Generate a random UUID. Alias for Uid(uuid.uuid4())."""
        return cls(uuid4())
