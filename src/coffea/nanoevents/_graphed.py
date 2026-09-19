"""The ``graphed`` arm of :class:`coffea.nanoevents.NanoEventsFactory`.

``mode="graphed"`` records an analysis into a ``graphed`` graph instead of materialising it. The
graph is built against the :class:`GraphedNanoArray` subclass installed here through the recording
backend, so every name a NanoEvents collection can be asked for takes exactly one route:

* a record field of the schema'd form, recorded as a ``field`` op;
* a ``no_dispatch`` descriptor, run eagerly on the record-time typetracer;
* a name graphed mode refuses, which raises :exc:`NotImplementedError` with a pointer;
* a :class:`coffea.util._DaskProperty`, whose ``.dask`` body runs with the graphed array in the
  deferred array's place;
* anything else, through ``graphed``'s own attribute dispatch.

Cross-references come out as ordinary graph edges with both collections as operands, so a worker
needs nothing but the recorded graph to replay them.

This is the only module in coffea that imports ``graphed``, and only ``mode="graphed"`` imports it.
"""

from inspect import getattr_static

import graphed
import uproot
from graphed.array import BoundMethod
from graphed.awkward import AwkwardBackend
from graphed.provenance import register_internal

from coffea.util import _DaskProperty

# a recorded op's provenance is the analyst's line, not the coffea frame that recorded it
register_internal("coffea")

#: names whose deferred meaning is a systematics axis, which graphed spells its own way
_REFUSED = frozenset({"_ensure_systematics", "add_systematic"})

_VARY_POINTER = "graphed mode does not carry coffea systematics; build the variations with graphed.vary(...)"


class _GraphedBoundMethod(BoundMethod):
    """A behavior method call on a graphed array, recorded as a ``method`` op."""

    def __call__(self, *args, **kwargs):
        if any(callable(arg) for arg in (*args, *kwargs.values())):
            raise NotImplementedError(
                f"{self._name}() was passed a callable, and graphed mode has no way to record "
                f"one; if it varies event data, spell the variation with graphed.vary(...)"
            )
        return super().__call__(*args, **kwargs)


class GraphedNanoArray(graphed.Array):
    """The deferred NanoEvents surface, mirroring ``dask_awkward.Array.__getattr__``."""

    __slots__ = ()

    def __getattr__(self, name):
        # The order is the contract, not an optimisation: a record field shadows a behavior of
        # the same name, and a descriptor that declined dispatch is answered before the arms
        # that record.
        if name.startswith("__"):
            raise AttributeError(name)
        session = self.session
        form = session.form(self)
        tracer = session.backend._with_behavior(form.tt)
        if name in tracer.fields:
            return session.record_op("field", [self], {"field": name})
        static = getattr_static(tracer, name, None)
        if getattr(static, "_no_dispatch", False):
            return static._dask_get(tracer, type(tracer), self)
        if name in _REFUSED:
            raise NotImplementedError(_VARY_POINTER)
        if isinstance(static, _DaskProperty):
            return static._dask_get(tracer, type(tracer), self)
        # everything left is graphed's own dispatch, reusing its method/property split; unlike
        # graphed.Array.__getattr__ it is reached for underscore names too, which the behavior
        # classes use for their internals
        if session.backend.attribute_kind(form, name) == "method":
            return _GraphedBoundMethod(self, name)
        return session.record_op("field", [self], {"field": name})


class GraphedNanoBackend(AwkwardBackend):
    """An ``AwkwardBackend`` whose sessions hand back :class:`GraphedNanoArray`."""

    def array_type(self):
        return GraphedNanoArray


def graphed_backend():
    """The worker-side recording backend, addressed as an import reference::

        graphed.aggregate_plan(..., backend="coffea.nanoevents._graphed:graphed_backend")

    It carries no behavior dict: a mapped chunk is assembled by ``awkward.from_buffers`` with the
    schema's behavior already applied, and that dict holds closures that would not pickle.
    """
    return GraphedNanoBackend()


def check_from_root(schemaclass, steps_per_file, uproot_options):
    """Refuse the ``from_root`` arguments and schemas graphed mode cannot honour."""
    if steps_per_file is not uproot._util.unset:
        raise NotImplementedError(
            "graphed mode does not partition files here; pass steps_per_file where the plan is "
            "built, graphed.aggregate_plan(..., steps_per_file=...)"
        )
    if "allow_read_errors_with_report" in uproot_options:
        raise NotImplementedError(
            "graphed mode has no report tuple; failed partitions are dead-lettered and retried "
            "by graphed.checkpoint.run_resumable"
        )
    if not isinstance(schemaclass, type) or not vars(schemaclass).get(
        "__graphed_capable__"
    ):
        raise NotImplementedError(
            f"graphed mode needs a schema declaring __graphed_capable__ = True in its own class "
            f"body, which {schemaclass!r} does not"
        )
