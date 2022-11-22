from aimmdb.ingest.eli import ingest, read_metadata_and_header
from pathlib import Path
import numpy as np
import pandas as pd


def test_read_md_and_header():
    test_path = Path("aimmdb/_tests/eli_test_file.dat")
    md, hdr = read_metadata_and_header(test_path)
    assert set(map(type, md.values())) == {str}  # assert all md values are of type str
    assert hdr.split() == [
        "energy",
        "i0",
        "it",
        "ir",
        "iff",
        "aux1",
        "aux2",
        "aux3",
        "aux4",
    ]


def test_ingest():
    test_path = Path("aimmdb/_tests/eli_test_file.dat")
    data = ingest(test_path, return_uid=True)
    test_channels = ["transmission", "fluorescence", "reference"]
    mu_channels = ["mu_trans", "mu_fluor", "mu_ref"]
    for channel_data, tst_ch, mu_ch in zip(data, test_channels, mu_channels):
        md = channel_data["metadata"]
        uid = channel_data["uid"]
        assert uid == md["_tiled"]["uid"] == "d66dda13-d69c-4ca6-8fb7-76290ad71073"
        channel = md["channel"]
        assert channel == tst_ch
        df = channel_data["data"]
        with open(test_path) as test_file:
            data_lines = [
                line for line in test_file.readlines() if not line.startswith("#")
            ]
            assert df.shape[0] == len(data_lines)
        assert mu_ch in list(df.columns)
