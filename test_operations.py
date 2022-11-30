import matplotlib

matplotlib.use("TkAgg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aimmdb.postprocessing import operations, eli
import pytest
from scipy import stats
from larch import Group as xafsgroup
from larch.xafs import pre_edge

from glob import glob

TEST_DIR = "/home/charles/Desktop/test_data/aimmdb_testdir"
TEST_DATA = sorted(glob(TEST_DIR + "/*.dat"))


def add_tiled_uid(*data):
    """Add _tiled key to metadata `dict`. This makes it easier to work with
    postprocessing operators."""
    for d in data:
        md = d["metadata"]
        md.update(_tiled={"uid": ""})
        md["_tiled"]["uid"] = md["Scan.uid"]


def test_avg():
    # transmission data from first five scans
    trans_data = [eli.ingest(file)[0] for file in TEST_DATA[0:5]]
    for td in trans_data:
        md = td["metadata"]

        # operators expect uid in "_tiled" of metadata
        md.update(_tiled={"uid": ""})
        md["_tiled"]["uid"] = md["Scan.uid"]
    avg_data = operations.AverageData(y_column="mu_trans")
    result = avg_data(*trans_data)
    result_df = result["data"]

    plt.plot(np.array([td["data"]["mu_trans"] for td in trans_data]).T)
    plt.plot(result_df["mu_trans"], label="avg")
    plt.plot(result_df["mu_trans"] + result_df["stddev"], c="k")
    plt.plot(result_df["mu_trans"] - result_df["stddev"], c="k")
    plt.legend()
    plt.show()


def test_groupidentity():
    # transmission data from first five scans
    trans_data = [eli.ingest(file)[0] for file in TEST_DATA[0:5]]
    add_tiled_uid(*trans_data)
    gi = operations.GroupIdentity()
    result = gi(*trans_data)
    print(type(result))
    input_uids = [td["metadata"]["_tiled"]["uid"] for td in trans_data]
    for r in result:
        md = r["metadata"]
        np.testing.assert_equal(md["_post_processing"]["relatives"], input_uids)


def test_aligngrids():
    # transmission data from first five scans
    trans_data = [eli.ingest(file)[0] for file in TEST_DATA[0:5]]
    add_tiled_uid(*trans_data)

    trans_data[0]["data"]["energy"] += 25

    ag = operations.AlignGrids(y_columns=["mu_trans"])
    master_grid = trans_data[0]["data"]["energy"]
    result = ag(*trans_data)
    for r in result:
        grid = r["data"]["energy"].to_numpy()
        np.testing.assert_equal(grid, master_grid)


def outlier_rejection(*dfs, data_window=60, threshold=25):
    all_data = np.array([df["mu_trans"] for df in dfs])
    n_pts = all_data.shape[1]
    trim_fraction = (1 - data_window / 100) / 2
    trim_data = stats.trimboth(all_data, trim_fraction, axis=0)
    avg_data = np.mean(all_data, axis=0)

    trim_mean = np.mean(trim_data, axis=0)
    trim_std = np.std(trim_data, axis=0)
    deviation_from_mean = (
        np.sum(((all_data - trim_mean) / trim_std) ** 2, axis=1) / n_pts
    )
    print(deviation_from_mean**0.5)
    for df in dfs:
        data = df["mu_trans"]
        deviation_from_mean = (
            1 / n_pts * np.sum(((data - trim_mean) / trim_std) ** 2)
        )
        print(deviation_from_mean**0.5)


def test_outlier():
    # transmission data from first five scans
    trans_data = [eli.ingest(file)[0] for file in TEST_DATA[0:5]]
    add_tiled_uid(*trans_data)
    for td in trans_data:
        mu = td["data"]["mu_trans"]
        energy = td["data"]["energy"]
        plt.plot(energy, mu)
    plt.show()
    outlier_rejection(*[td["data"] for td in trans_data])

    # check_outliers = operations.CheckForOutliers(y_column="mu_trans")
    # result = check_outliers(*trans_data)
    # for r in result:
    #     print(r["data"])


if __name__ == "__main__":
    test_outlier()
