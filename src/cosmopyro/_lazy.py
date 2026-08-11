"""Lazy public re-exports for subpackage ``__init__`` files.

Eager re-exports deadlock: internal modules reach across subpackages (e.g.
``cosmology.flat_lambdacdm`` imports ``utils.jax_utils``), and importing a
submodule runs its parent ``__init__`` first, so the two cycle.  Resolving names
on first access avoids that and keeps subpackage imports cheap.  Editors still
see the names via the ``if TYPE_CHECKING:`` block in each ``__init__``.
"""

from importlib import import_module

__all__ = ["lazy_getattr"]


def lazy_getattr(namespace, exports):
    """Build a PEP 562 ``__getattr__`` resolving ``exports`` {name: module}.

    Pass the caller's ``globals()`` as ``namespace``; resolved names are cached
    there, so each is looked up at most once.
    """
    package = namespace["__name__"]

    def __getattr__(name):
        module = exports.get(name)
        if module is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(import_module(module, package), name)
        namespace[name] = value
        return value

    return __getattr__
