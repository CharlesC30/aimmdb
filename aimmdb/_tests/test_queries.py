import string

import numpy
import pandas
import pytest
from tiled.client import from_tree
from tiled.client.array import ArrayClient
from tiled.client.dataframe import DataFrameClient
from tiled.queries import Contains, In, Key, NotIn, Specs, StructureFamily

from aimmdb.adapters.aimm import AIMMCatalog


def write_data(client):
    # sandbox_0
    for letter, number in zip(list(string.ascii_lowercase), range(26)):
        client["uid"].write_array(
            number * numpy.ones(10),
            metadata={"dataset": "sandbox_0", "letter": letter, "number": number},
        )

    # sandbox_1
    client["uid"].write_array(
        numpy.random.rand(100),
        metadata={
            "dataset": "sandbox_1",
            "letters": list(string.ascii_lowercase),
            "name": "does_contain_z",
        },
    )

    client["uid"].write_array(
        numpy.random.rand(100),
        metadata={
            "dataset": "sandbox_1",
            "letters": list(string.ascii_lowercase)[:-1],
            "name": "does_not_contain_z",
        },
    )

    # sandbox_2
    client["uid"].write_array(
        numpy.random.rand(100), metadata={"dataset": "sandbox_2", "name": "array"}
    )
    client["uid"].write_dataframe(
        pandas.DataFrame({"a": numpy.random.rand(100), "b": numpy.random.rand(100)}),
        metadata={"dataset": "sandbox_2", "name": "dataframe"},
    )

    # sandbox_3
    client["uid"].write_array(
        numpy.random.rand(100),
        metadata={"dataset": "sandbox_3", "name": "no_specs"},
        specs=[],
    )
    client["uid"].write_array(
        numpy.random.rand(100),
        metadata={"dataset": "sandbox_3", "name": "specs_foo_bar"},
        specs=["foo", "bar"],
    )
    client["uid"].write_array(
        numpy.random.rand(100),
        metadata={"dataset": "sandbox_3", "name": "specs_foo_bar_baz"},
        specs=["foo", "bar", "baz"],
    )

    return client


api_key = "secret"


@pytest.fixture(scope="session")
def tree(tmp_path_factory):
    data_directory = tmp_path_factory.mktemp("data")
    t = AIMMCatalog.from_mongomock(data_directory)

    c = from_tree(
        t,
        api_key=api_key,
        authentication={"single_user_api_key": api_key},
    )

    write_data(c)

    return t


@pytest.fixture
def client(tree):
    return from_tree(
        tree,
        api_key=api_key,
        authentication={"single_user_api_key": api_key},
    )


def test_eq(client):
    c = client["dataset"]["sandbox_0"]["uid"]

    for letter in list(string.ascii_lowercase):
        results = c.search(Key("letter") == letter)
        assert len(results) == 1
        result = results.values().first()
        assert result.metadata["letter"] == letter


def test_noteq(client):
    c = client["dataset"]["sandbox_0"]["uid"]

    for letter in list(string.ascii_lowercase):
        results = c.search(Key("letter") != letter)
        for v in results.values():
            assert v.metadata["letter"] != letter


def test_comparison(client):
    c = client["dataset"]["sandbox_0"]["uid"]

    results = c.search(Key("number") >= 12)
    for v in results.values():
        assert v.metadata["number"] >= 12


def test_in(client):
    c = client["dataset"]["sandbox_0"]["uid"]
    results = c.search(In("letter", ["a", "k", "z"]))
    for v in results.values():
        assert v.metadata["letter"] in ["a", "k", "z"]


def test_notin(client):
    c = client["dataset"]["sandbox_0"]["uid"]
    results = c.search(NotIn("letter", ["a", "k", "z"]))
    for v in results.values():
        assert v.metadata["letter"] not in ["a", "k", "z"]


def test_contains(client):
    c = client["dataset"]["sandbox_1"]["uid"]

    results = c.search(Contains("letters", "z"))

    assert len(results) == 1
    assert results.values().first().metadata["name"] == "does_contain_z"


def test_structure_family(client):
    c = client["dataset"]["sandbox_2"]["uid"]

    results = c.search(StructureFamily("dataframe"))
    assert len(results) > 0
    for v in results.values():
        assert isinstance(v, DataFrameClient)

    results = c.search(StructureFamily("array"))
    assert len(results) > 0
    for v in results.values():
        assert isinstance(v, ArrayClient)


def test_specs(client):
    c = client["dataset"]["sandbox_3"]["uid"]

    results = c.search(Specs(include=["foo", "bar"]))
    assert len(results) > 0
    for v in results.values():
        assert "foo" in v.specs
        assert "bar" in v.specs

    results = c.search(Specs(include=["foo", "bar"], exclude=["baz"]))
    assert len(results) > 0
    for v in results.values():
        assert "foo" in v.specs
        assert "bar" in v.specs
        assert "baz" not in v.specs
