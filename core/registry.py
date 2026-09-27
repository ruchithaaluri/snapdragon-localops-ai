"""A small, explicit registry for replaceable components.

Every pluggable piece of the system (AI backends, retrieval backends, ... ) is
registered by name here. Registration is explicit: nothing is auto-discovered,
so swapping an implementation is a configuration change, not a code change.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from core.errors import ProviderNotFoundError

__all__ = ["ProviderDescriptor", "ProviderRegistry"]

P = TypeVar("P")


@dataclass(frozen=True, slots=True)
class ProviderDescriptor(Generic[P]):
    """A named factory for a component, plus metadata for the UI."""

    name: str
    kind: str
    factory: Callable[..., P]
    description: str = ""
    requires: tuple[str, ...] = field(default=())
    priority: int = 100

    def create(self, **kwargs: Any) -> P:
        """Instantiate the component through its factory."""
        return self.factory(**kwargs)


class ProviderRegistry(Generic[P]):
    """Name -> factory map with predictable errors and ordering.

    >>> registry: ProviderRegistry[str] = ProviderRegistry()
    >>> registry.register(ProviderDescriptor(name="stub", kind="demo",
    ...                                     factory=lambda: "value"))
    >>> registry.create("stub")
    'value'
    >>> registry.create("missing")
    Traceback (most recent call last):
        ...
    core.errors.ProviderNotFoundError: no provider named 'missing' ...
    """

    def __init__(self, kind: str = "provider") -> None:
        self._kind = kind
        self._items: dict[str, ProviderDescriptor[P]] = {}

    def register(self, descriptor: ProviderDescriptor[P], *, replace: bool = False) -> None:
        """Add a descriptor.

        Raises on a duplicate name unless ``replace=True``, so a name collision
        is a visible error rather than a silent override.
        """
        if descriptor.name in self._items and not replace:
            raise ValueError(
                f"{self._kind} already registered: {descriptor.name!r}"
            )
        self._items[descriptor.name] = descriptor

    def get(self, name: str) -> ProviderDescriptor[P]:
        """Return the descriptor for a name or raise."""
        try:
            return self._items[name]
        except KeyError as exc:
            raise ProviderNotFoundError(
                f"no {self._kind} named {name!r}",
                available=self.names(),
            ) from exc

    def create(self, name: str, **kwargs: Any) -> P:
        """Instantiate a registered component by name."""
        return self.get(name).create(**kwargs)

    def has(self, name: str) -> bool:
        return name in self._items

    def names(self) -> tuple[str, ...]:
        """Registered names, ordered by priority then name."""
        return tuple(
            d.name
            for d in sorted(self._items.values(), key=lambda d: (d.priority, d.name))
        )

    def descriptors(self) -> tuple[ProviderDescriptor[P], ...]:
        return tuple(self._items[d] for d in self.names())

    def unregister(self, name: str) -> None:
        """Remove a name; raises if it is not registered."""
        self.get(name)
        del self._items[name]

    def clear(self) -> None:
        self._items.clear()

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._items

    def __iter__(self) -> Iterator[ProviderDescriptor[P]]:
        return iter(self.descriptors())

    def __len__(self) -> int:
        return len(self._items)
