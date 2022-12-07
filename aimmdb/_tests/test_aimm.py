import copy

import numpy as np
import pandas as pd
from tiled.client import from_tree
from tiled.queries import Key
from tiled.validation_registration import ValidationRegistry

import aimmdb
from aimmdb.adapters.aimm import AIMMCatalog
from aimmdb.validation import validate_xas_transmission

from .utils import fail_with_status_code


def get_client(tmpdir, dataset_to_specs=None, validation_registry=None):
    dataset_to_specs = dataset_to_specs or {}
    validation_registry = validation_registry or ValidationRegistry()

    data_directory = tmpdir / "data"
    data_directory.mkdir()

    tree = AIMMCatalog.from_mongomock(
        data_directory,
        dataset_to_specs=dataset_to_specs,
    )

    api_key = "secret"
    c = from_tree(
        tree,
        api_key=api_key,
        authentication={"single_user_api_key": api_key},
        validation_registry=validation_registry,
    )

    return c


def test_basic(tmpdir):
    c = get_client(tmpdir)
    assert type(c) == aimmdb.client.AIMMCatalog


def test_write_array(tmpdir):
    c = get_client(tmpdir)

    x = np.random.rand(100, 100)
    metadata = {"foo": "bar"}

    # can't write without specifying dataset
    with fail_with_status_code(400):
        result = c["uid"].write_array(x, metadata=metadata)

    metadata.update(dataset="sandbox1", myid=1)
    c["uid"].write_array(x, metadata=metadata)

    results = c["uid"].search(Key("myid") == 1)
    result = results.values().first()
    result_array = result.read()

    np.testing.assert_equal(result_array, x)
    for k, v in metadata.items():
        assert result.metadata[k] == v


def test_write_dataframe(tmpdir):
    c = get_client(tmpdir)

    df = pd.DataFrame({"a": np.random.rand(100), "b": np.random.rand(100)})
    metadata = {"foo": "bar"}

    # can't write without specifying dataset
    with fail_with_status_code(400):
        result = c["uid"].write_dataframe(df, metadata=metadata)

    metadata.update(dataset="sandbox1", myid=1)
    c["uid"].write_dataframe(df, metadata=metadata)

    results = c["uid"].search(Key("myid") == 1)
    result = results.values().first()
    result_df = result.read()

    pd.testing.assert_frame_equal(result_df, df)

    for k, v in metadata.items():
        assert result.metadata[k] == v


def test_validation(tmpdir):
    validation_registry = ValidationRegistry()
    validation_registry.register("XAS_trans", validate_xas_transmission)
    dataset_to_specs = {"xas": ["XAS_trans"]}

    c = get_client(
        tmpdir,
        dataset_to_specs=dataset_to_specs,
        validation_registry=validation_registry,
    )

    metadata = {
        "dataset": "xas",
        "element": {"symbol": "Au", "edge": "K"},
        "facility": {"name": "ALS"},
        "beamline": {"name": "foo"},
        "myid": 1,
    }
    df = pd.DataFrame(
        {
            "energy": np.random.rand(100),
            "i0": np.random.rand(100),
            "itrans": np.random.rand(100),
        }
    )

    c["uid"].write_dataframe(df, metadata=metadata, specs=["XAS_trans"])

    results = c["uid"].search(Key("myid") == 1)
    result = results.values().first()
    result_df = result.read()

    pd.testing.assert_frame_equal(result_df, df)

    for k, v in metadata.items():
        assert result.metadata[k] == v

    # fail with wrong spec
    with fail_with_status_code(400):
        c["uid"].write_dataframe(df, metadata=metadata, specs=["FOO"])

    # fail without complete metadata
    metadata_incomplete = copy.deepcopy(metadata)
    metadata_incomplete.pop("element")
    with fail_with_status_code(400):
        c["uid"].write_dataframe(df, metadata=metadata_incomplete, specs=["XAS_trans"])

    # fail with missing columns
    df_missing_columns = pd.DataFrame(
        {
            "energy": np.random.rand(100),
            "i0": np.random.rand(100),
        }
    )
    with fail_with_status_code(400):
        c["uid"].write_dataframe(
            df_missing_columns, metadata=metadata, specs=["XAS_trans"]
        )
