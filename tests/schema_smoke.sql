-- Run after db/schema.sql in a disposable test database.
BEGIN;
INSERT INTO model_versions (version, metrics, active)
VALUES ('test-v0', '{"accuracy": 0.9}', TRUE);
INSERT INTO flows (seen_at, src_ip, dst_ip, src_port, dst_port, protocol, duration_ms, packets, bytes, source)
VALUES (NOW(), '192.0.2.1', '198.51.100.2', 12345, 443, 'TCP', 10.5, 5, 250, 'replay');
INSERT INTO detections (flow_id, model_version, predicted_class, confidence, is_alert, top_reasons)
VALUES (currval(pg_get_serial_sequence('flows', 'id')), 'test-v0', 'benign', 0.95, FALSE, '["test"]');
INSERT INTO alert_messages (alert_type, src_ip, first_seen, last_seen, count)
VALUES ('port_scan', '192.0.2.1', NOW(), NOW(), 1);
DO $$ BEGIN
  IF (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name IN ('flows','detections','model_versions','alert_messages')) <> 4 THEN
    RAISE EXCEPTION 'Expected four C2 tables';
  END IF;
END $$;
ROLLBACK;
