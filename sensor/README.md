# sensor

## Score and store flows (SEN-FR-17, TC-06)

Runs a PCAP through the model and writes each flow and its detection to PostgreSQL.

```bash
docker compose up -d postgres            # applies db/schema.sql on first start
export DATABASE_URL=postgresql://sentinel:<password>@localhost:5432/sentinel
python -m sentinel_ml.dummy_package      # builds models/sentinel-dummy-v0
python -m sensor.run --pcap slice.pcap --model models/sentinel-dummy-v0
```

Use `localhost` in `DATABASE_URL` when running from the host. The `postgres` host name in `.env.example` only works inside the compose network.

Per flow it writes:

- `flows`: time, IPs, ports, protocol, duration, packets, bytes, `source = 'replay'`
- `detections`: `flow_id`, `predicted_class`, `confidence`, `model_version`, `is_alert = false`
- `model_versions`: the package version (added if missing, since detections reference it)

All rows for a run are written in one transaction, so a failure leaves nothing half-written. `is_alert` stays false until the threshold lands (SEN-FR-18).

Tests: `pytest tests/test_store.py`. Set `TEST_DATABASE_URL` to also run the round trip against a real database.
