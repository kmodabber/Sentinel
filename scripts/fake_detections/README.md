# SEN-FR-20 — Fake detections generator (Sentinel C2 v0)

This implementation matches the **provided `flows`, `model_versions`, and `detections` schema**. It does not create tables and does not write `alert_messages` (reserved for Discord notification grouping).

**Important C2 limitation:** `flows.source` permits only `replay`, `live`, or `honeypot`, not `synthetic`. The generator uses `replay` as a **development-only placeholder** and sets `model_version='synthetic-fr20-v1'` so synthetic detections can be identified. Do **not** use it on real production data; confirm this convention with the team or extend the C2 contract to permit `synthetic`.

## Setup in Windows PowerShell

Run commands from the Sentinel repository root after copying this package's files into a `scripts/fake_detections/` folder.

```powershell
python -m pip install -r scripts/fake_detections/requirements.txt
$env:DATABASE_URL = "postgresql://YOUR_USER:YOUR_PASSWORD@localhost:5432/YOUR_DB"
python scripts/fake_detections/fake_detections.py --count 30 --interval 1
```

Replace database placeholders with your **actual local PostgreSQL values** from Compose or `.env`. Do not commit credentials. For a continuous stream:

```powershell
python scripts/fake_detections/fake_detections.py --interval 1
```

Press Ctrl+C to stop.

## Unit tests

```powershell
$env:PYTHONPATH = "scripts/fake_detections"
python -m unittest discover -s scripts/fake_detections/tests -v
```

## TC-09: database integration check

**Use an empty, disposable test database** initialized using the supplied C2 `db/schema.sql`, not a shared/team database. The generator never truncates or deletes rows.

Before running:

```sql
SELECT COUNT(*) AS flows_before FROM flows;
SELECT COUNT(*) AS detections_before FROM detections;
```

Both counts must be zero. Run 30 records (`--count 30 --interval 1`), then check:

```sql
SELECT COUNT(*) AS flows_after FROM flows;
SELECT COUNT(*) AS detections_after FROM detections;
SELECT predicted_class, COUNT(*) FROM detections GROUP BY predicted_class ORDER BY 2 DESC;
SELECT model_version, COUNT(*) FROM detections GROUP BY model_version;
SELECT COUNT(*) AS broken_foreign_keys
FROM detections d LEFT JOIN flows f ON f.id = d.flow_id WHERE f.id IS NULL;
SELECT MIN(seen_at), MAX(seen_at) FROM flows
WHERE seen_at > NOW() - INTERVAL '5 minutes';
SELECT id, seen_at, src_ip, dst_ip, dst_port, source FROM flows ORDER BY id DESC LIMIT 10;
```

Expected: **30 flows**, at least one detection for the default seeded run, `broken_foreign_keys = 0`, and `seen_at` spread across roughly 29 seconds. If the database is not empty, the total counts cannot be used without recording the baseline.

## Definition of Done

TC-09 must pass on an actual PostgreSQL instance using C2. Capture terminal and SQL outputs (plus Grafana if available); run tests and open a reviewed PR. This package has been unit-tested only, not database-integrated here. Do not merge data, credentials or generated records.
