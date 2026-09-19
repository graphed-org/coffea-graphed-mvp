"""Every name a graphed-mode collection can be asked for takes exactly one known route."""

import os
import sys
import types
from inspect import getattr_static

import awkward as ak
import pytest

if sys.version_info < (3, 11):
    pytest.skip("graphed requires Python 3.11 or newer", allow_module_level=True)

graphed = pytest.importorskip("graphed")
pytest.importorskip("graphed.awkward")

from graphed.array import BoundMethod  # noqa: E402
from test_nanoevents_graphed import assert_same  # noqa: E402

from coffea.nanoevents import (  # noqa: E402
    NanoAODSchema,
    NanoEventsFactory,
    PFNanoAODSchema,
)
from coffea.nanoevents.methods.base import NanoCollection  # noqa: E402
from coffea.nanoevents.methods.edm4hep import edm4hep_nanocollection  # noqa: E402
from coffea.util import _DaskMethod, _DaskProperty  # noqa: E402

#: the routes of the graphed dispatch, in the order they are tried
FIELD, NO_DISPATCH, GRAPHED_ARM, REFUSED, DASK_ARM, FALLBACK = (
    "field",
    "no_dispatch",
    "graphed_arm",
    "refused",
    "dask_arm",
    "fallback",
)


def _graphed_arm(descriptor):
    """The arm a descriptor's ``.graphed`` registration slot holds, beside ``.dask``'s."""
    return getattr(descriptor, "_graphed_get", None)


def _behavior_names(schemaclass):
    """Every attribute coffea's own behavior classes add, found by walking the schema's
    behavior dict — descriptors (``no_dispatch`` ones included), plain properties and plain
    methods. Class constants and nested classes are not routed and are left out."""
    routed = (property, _DaskProperty, _DaskMethod, types.FunctionType)
    names = set()
    for value in schemaclass.behavior().values():
        if not isinstance(value, type):
            continue
        for klass in value.__mro__:
            if not klass.__module__.startswith("coffea.nanoevents.methods"):
                continue
            names.update(
                name
                for name, member in vars(klass).items()
                if not name.startswith("__") and isinstance(member, routed)
            )
    return names


def _tracer(array):
    session = array.session
    return session.backend._with_behavior(session.form(array).tt)


def _route(holder, eager_holder, name):
    """Which route ``holder.<name>`` takes, observed; ``None`` when nothing explains it."""
    tracer = _tracer(holder)
    if name in (tracer.fields or []):
        return FIELD
    static = getattr_static(tracer, name, None)
    if _graphed_arm(static) is not None:
        return GRAPHED_ARM
    failure = None
    try:
        got = getattr(holder, name)
    except NotImplementedError:
        return REFUSED
    except Exception as exc:
        got, failure = None, type(exc)
    if failure is not None:
        # an arm ran and its own body refused, exactly as the eager arm refuses
        with pytest.raises(failure):
            getattr(eager_holder, name)
    elif isinstance(got, BoundMethod):
        return FALLBACK
    elif not isinstance(got, graphed.Array):
        return NO_DISPATCH  # ran eagerly on the typetracer, recording nothing
    if isinstance(static, _DaskProperty) and getattr(static, "_dask_get", None):
        return DASK_ARM
    if failure is None:
        return FALLBACK
    return None


def _arms(tests_directory, sample, schemaclass):
    path = os.path.join(tests_directory, "samples", sample)
    both = []
    for mode in ("graphed", "eager"):
        both.append(
            NanoEventsFactory.from_root(
                {path: "Events"}, schemaclass=schemaclass, mode=mode
            ).events()
        )
    return tuple(both)


@pytest.fixture(scope="module")
def nanoaod(tests_directory):
    return _arms(tests_directory, "nano_dy.root", NanoAODSchema)


@pytest.fixture(scope="module")
def pfnano(tests_directory):
    return _arms(tests_directory, "pfnano.root", PFNanoAODSchema)


@pytest.fixture(
    params=[(NanoAODSchema, "nanoaod"), (PFNanoAODSchema, "pfnano")],
    ids=["nanoaod", "pfnano"],
)
def admitted(request, nanoaod, pfnano):
    schemaclass, fixture = request.param
    return (schemaclass, *request.getfixturevalue(fixture))


def test_every_name_takes_exactly_one_route(admitted):
    schemaclass, events, eager = admitted
    behaviors = _behavior_names(schemaclass)
    assert behaviors, "the behavior walk found nothing to route"

    holders = [("events", events, eager)]
    holders += [(f, events[f], eager[f]) for f in _tracer(events).fields]

    routes, unclassified = {}, []
    for collection, holder, eager_holder in holders:
        tracer = _tracer(holder)
        fields = set(tracer.fields or [])
        for name in sorted(behaviors | fields):
            if name not in fields and getattr_static(tracer, name, None) is None:
                continue  # this behavior is not attached to this collection
            route = _route(holder, eager_holder, name)
            if route is None:
                unclassified.append(f"{collection}.{name}")
            else:
                routes.setdefault(route, set()).add(name)

    assert unclassified == []
    assert set(routes) == {
        FIELD,
        NO_DISPATCH,
        GRAPHED_ARM,
        REFUSED,
        DASK_ARM,
        FALLBACK,
    }
    assert routes[GRAPHED_ARM] == {"_apply_global_index"}
    assert routes[REFUSED] == {"_ensure_systematics", "add_systematic"}
    assert {"_events", "_content", "_collection_name"} <= routes[NO_DISPATCH]
    assert {"matched_jet", "children", "distinctParent"} <= routes[DASK_ARM]
    assert {"delta_r", "metric_table", "isTight"} <= routes[FALLBACK]


def test_the_global_index_arms_are_registered_on_the_graphed_slot():
    assert callable(_DaskProperty.graphed) and callable(_DaskMethod.graphed)
    for owner, name in (
        (NanoCollection, "_apply_global_index"),
        (edm4hep_nanocollection, "_apply_nested_global_index"),
    ):
        descriptor = getattr_static(owner, name)
        assert _graphed_arm(descriptor) is not None
        assert getattr(descriptor, "_dask_get", None) is not None


# ---- one representative per route, asserted bit-for-bit against the eager arm ------------------
def r_field(events):
    return events.Muon.pt


def r_no_dispatch(events):
    return events.Muon._events().Jet.pt


def r_graphed_arm(events):
    return events.GenJet._apply_global_index(events.Jet.genJetIdxG).pt


def r_dask_arm(events):
    return events.Muon.matched_jet.pt


def r_fallback_method(events):
    return events.Muon.delta_r(events.Muon.matched_jet)


def r_fallback_property(events):
    return events.Jet.isTight


REPRESENTATIVES = {
    FIELD: (r_field, "Muon", "pt"),
    NO_DISPATCH: (r_no_dispatch, "Muon", "_events"),
    GRAPHED_ARM: (r_graphed_arm, "GenJet", "_apply_global_index"),
    DASK_ARM: (r_dask_arm, "Muon", "matched_jet"),
    "fallback_method": (r_fallback_method, "Muon", "delta_r"),
    "fallback_property": (r_fallback_property, "Jet", "isTight"),
}


@pytest.mark.parametrize("route", sorted(REPRESENTATIVES))
def test_route_representative_matches_eager(nanoaod, route):
    events, eager = nanoaod
    analysis, collection, name = REPRESENTATIVES[route]
    taken = _route(events[collection], eager[collection], name)
    assert taken == (FALLBACK if route.startswith("fallback") else route)
    assert_same(events.session.materialize(analysis(events)), analysis(eager))


def test_refused_route_representative_raises(nanoaod):
    events, _eager = nanoaod
    with pytest.raises(NotImplementedError):
        events.Muon._ensure_systematics()
    assert _route(events.Muon, _eager.Muon, "_ensure_systematics") == REFUSED


def test_eager_arms_are_untouched(tests_directory):
    """The same names on the eager array still answer with real values."""
    path = os.path.join(tests_directory, "samples", "nano_dy.root")
    eager = NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=NanoAODSchema, mode="eager"
    ).events()
    assert eager.Muon._collection_name() == "Muon"
    assert ak.all(eager.Muon.matched_jet.pt == eager.Jet[eager.Muon.jetIdx].pt)
