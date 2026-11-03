"""
Parity tests for the NFS-2023 Tuesday regenerated vs. published dataset.
"""

import os

import pandas as pd
import pytest


def label_diffs(mine: pd.Series, pub: pd.Series) -> pd.Series:
    """Percent difference per label: (mine - published) / published * 100."""
    counts = pd.DataFrame({"mine": mine, "pub": pub}).fillna(0)
    return (counts["mine"] - counts["pub"]) / counts["pub"] * 100


def same_columns(mine_cols, pub_cols) -> bool:
    return list(mine_cols) == list(pub_cols)


def test_within_one_percent_passes():
    mine = pd.Series({"BENIGN": 1003, "FTP-Patator": 100})
    pub = pd.Series({"BENIGN": 1000, "FTP-Patator": 100})
    assert (label_diffs(mine, pub).abs() <= 1).all()


def test_over_one_percent_is_caught():
    diffs = label_diffs(pd.Series({"SSH-Patator": 95}), pd.Series({"SSH-Patator": 100}))
    assert diffs["SSH-Patator"] == -5


def test_missing_label_counts_as_gap():
    diffs = label_diffs(pd.Series({"BENIGN": 10}), pd.Series({"BENIGN": 10, "SSH-Patator": 5}))
    assert diffs["SSH-Patator"] == -100


def test_columns_must_match_in_order():
    assert same_columns(["a", "b"], ["a", "b"])
    assert not same_columns(["a", "b"], ["b", "a"])


REGEN =r"C:\Users\Hossane\Desktop\test_pcap\NFS-2023-TE\datasets\Tuesday_Regen.csv"
PUBLISHED = r"C:\Users\Hossane\Desktop\test_pcap\NFS-2023-TE\datasets\CyberML-DataQuality-Tuesday.csv"


@pytest.mark.skipif(
    not (REGEN and PUBLISHED and os.path.isfile(REGEN) and os.path.isfile(PUBLISHED)),
    reason="regenerated and published CSVs not available",
)
def test_tuesday_matches_published():
    mine = pd.read_csv(REGEN, usecols=["label"])["label"].value_counts()
    pub = pd.read_csv(PUBLISHED, usecols=["label"])["label"].value_counts()
    assert (label_diffs(mine, pub).abs() <= 1).all()
    mine_cols = pd.read_csv(REGEN, nrows=0).columns
    pub_cols = pd.read_csv(PUBLISHED, nrows=0).columns
    assert same_columns(mine_cols, pub_cols)