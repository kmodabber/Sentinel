"""TC-06: every finished flow is scored and written to PostgreSQL.

Most tests use a fake connection so they run anywhere. The last one runs
against a real database when TEST_DATABASE_URL points at one that was
started from db/schema.sql.
"""
import os

import pandas as pd
import pytest

from sensor import pcap_flows, run, store
from sensor.package import load_package, score
from sentinel_features.features import FEATURES, LABELS
from sentinel_ml.dummy_package import build_dummy_package


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.last = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        if self.conn.fail_on == len(self.conn.log):
            raise RuntimeError("db down")
        self.conn.log.append((sql, params))
        self.last = sql

    def fetchone(self):
        self.conn.next_id += 1
        return (self.conn.next_id,)


class FakeConn:
    def __init__(self, fail_on=None):
        self.log = []
        self.next_id = 0
        self.fail_on = fail_on
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        pass

    def rows(self, sql):
        return [p for s, p in self.log if s is sql]


def make_flows(n=3):
    cols = list(dict.fromkeys(FEATURES + pcap_flows.METADATA_COLUMNS))
    df = pd.DataFrame({c: [1.0] * n for c in cols})
    df["src_ip"] = [f"10.0.0.{i + 1}" for i in range(n)]
    df["dst_ip"] = "10.0.0.99"
    df["src_port"] = [40000 + i for i in range(n)]
    df["dst_port"] = 443
    df["protocol"] = [6, 17, 1][:n] + [6] * max(0, n - 3)
    df["bidirectional_first_seen_ms"] = 1_500_000_000_000
    df["bidirectional_duration_ms"] = 12.5
    df["bidirectional_packets"] = 7
    df["bidirectional_bytes"] = 640
    return df


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    folder = build_dummy_package(tmp_path_factory.mktemp("pkg") / "pkg")
    return folder, load_package(folder)


def test_flow_row_maps_columns():
    row = store.flow_row(make_flows(1).iloc[0], "replay")
    seen_at, src, dst, sport, dport, proto, dur, pkts, nbytes, source = row
    assert seen_at.isoformat() == "2017-07-14T02:40:00+00:00"
    assert (src, dst, sport, dport) == ("10.0.0.1", "10.0.0.99", 40000, 443)
    assert (proto, dur, pkts, nbytes, source) == ("TCP", 12.5, 7, 640, "replay")


def test_unknown_protocol_number_is_kept_as_text():
    assert store.protocol_name(47) == "47"
    assert store.protocol_name(17.0) == "UDP"


def test_one_flow_and_one_detection_per_flow(package):
    _, pkg = package
    flows = make_flows(3)
    scores = score(pkg, flows)
    conn = FakeConn()

    assert store.store_scored_flows(conn, flows, scores) == 3

    assert len(conn.rows(store.INSERT_FLOW)) == 3
    detections = conn.rows(store.INSERT_DETECTION)
    assert len(detections) == 3
    for i, (flow_id, version, label, confidence) in enumerate(detections):
        assert flow_id == i + 1
        assert version == "sentinel-dummy-v0"
        assert label in LABELS
        assert 0 <= confidence <= 1
    assert conn.commits == 1 and conn.rollbacks == 0


def test_alert_flag_stays_false_until_threshold_exists():
    assert "FALSE" in store.INSERT_DETECTION


def test_confidence_is_clipped_into_range():
    flows = make_flows(1)
    scores = pd.DataFrame({
        "predicted_class": ["BENIGN"], "confidence": [1.0000001],
        "model_version": ["v"],
    })
    conn = FakeConn()
    store.store_scored_flows(conn, flows, scores)
    assert conn.rows(store.INSERT_DETECTION)[0][3] == 1.0


def test_error_rolls_back_and_raises():
    flows = make_flows(2)
    scores = pd.DataFrame({
        "predicted_class": ["BENIGN"] * 2, "confidence": [0.9] * 2,
        "model_version": ["v"] * 2,
    })
    conn = FakeConn(fail_on=3)
    with pytest.raises(RuntimeError):
        store.store_scored_flows(conn, flows, scores)
    assert conn.commits == 0 and conn.rollbacks == 1


def test_mismatched_scores_are_refused():
    with pytest.raises(ValueError):
        store.store_scored_flows(FakeConn(), make_flows(2), pd.DataFrame())


def test_missing_metadata_column_is_refused(package):
    _, pkg = package
    flows = make_flows(2).drop(columns=["src_ip"])
    with pytest.raises(KeyError, match="src_ip"):
        store.store_scored_flows(FakeConn(), flows, score(pkg, flows))


def test_register_model_runs_before_detections(package, monkeypatch):
    folder, _ = package
    monkeypatch.setattr(run, "flows_from_pcap", lambda p, keep_metadata: make_flows(2))
    conn = FakeConn()
    assert run.run("x.pcap", folder, conn) == 2
    assert conn.log[0][0] is store.INSERT_MODEL
    assert conn.log[0][1][0] == "sentinel-dummy-v0"


def test_connect_needs_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        store.connect()


TEST_DB = os.environ.get("TEST_DATABASE_URL")


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to a schema-initialised DB")
def test_real_postgres_round_trip(package, monkeypatch):
    folder, _ = package
    monkeypatch.setattr(run, "flows_from_pcap", lambda p, keep_metadata: make_flows(3))
    conn = store.connect(TEST_DB)
    try:
        before = _count(conn, "detections")
        assert run.run("x.pcap", folder, conn) == 3
        assert _count(conn, "detections") == before + 3
        with conn.cursor() as cur:
            cur.execute(
                "SELECT f.protocol, d.model_version, d.is_alert FROM detections d "
                "JOIN flows f ON f.id = d.flow_id ORDER BY d.id DESC LIMIT 3"
            )
            rows = cur.fetchall()
        assert {r[0] for r in rows} == {"TCP", "UDP", "ICMP"}
        assert {r[1] for r in rows} == {"sentinel-dummy-v0"}
        assert not any(r[2] for r in rows)
    finally:
        conn.close()


def _count(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        return cur.fetchone()[0]
