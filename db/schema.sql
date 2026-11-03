-- Sentinel Contract C2: proposed PostgreSQL schema v0
-- Review SQL types, nullability, and keys with the team before merging.
CREATE TABLE IF NOT EXISTS flows (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    seen_at TIMESTAMPTZ NOT NULL,
    src_ip INET NOT NULL,
    dst_ip INET NOT NULL,
    src_port INTEGER CHECK (src_port BETWEEN 0 AND 65535),
    dst_port INTEGER CHECK (dst_port BETWEEN 0 AND 65535),
    protocol TEXT NOT NULL,
    duration_ms DOUBLE PRECISION CHECK (duration_ms >= 0),
    packets BIGINT CHECK (packets >= 0),
    bytes BIGINT CHECK (bytes >= 0),
    source TEXT NOT NULL CHECK (source IN ('replay', 'live', 'honeypot'))
);

CREATE TABLE IF NOT EXISTS model_versions (
    version TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    active BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS detections (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    flow_id BIGINT NOT NULL REFERENCES flows(id),
    model_version TEXT NOT NULL REFERENCES model_versions(version),
    predicted_class TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    is_alert BOOLEAN NOT NULL DEFAULT FALSE,
    top_reasons JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS alert_messages (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    alert_type TEXT NOT NULL,
    src_ip INET NOT NULL,
    first_seen TIMESTAMPTZ NOT NULL,
    last_seen TIMESTAMPTZ NOT NULL,
    count BIGINT NOT NULL DEFAULT 1 CHECK (count >= 1),
    discord_sent_at TIMESTAMPTZ,
    CONSTRAINT alert_time_order CHECK (last_seen >= first_seen)
);

CREATE INDEX IF NOT EXISTS idx_flows_seen_at ON flows(seen_at);
CREATE INDEX IF NOT EXISTS idx_detections_flow_id ON detections(flow_id);
CREATE INDEX IF NOT EXISTS idx_detections_created_at ON detections(created_at);
CREATE INDEX IF NOT EXISTS idx_alert_messages_last_seen ON alert_messages(last_seen);
