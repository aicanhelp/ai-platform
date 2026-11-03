-- Copyright (c) 2026 Huawei Technologies Co., Ltd.
-- openFuyao is licensed under Mulan PSL v2.
-- Snapshot warm-up schema (design doc 0004: 沙箱快照预热).
-- Apply this to the PostgreSQL instance backing the warm subsystem.

-- 2.4.1 User request task table.
CREATE TABLE IF NOT EXISTS snapshot_warm_task (
    task_id              VARCHAR     PRIMARY KEY,
    snapshot_ids         JSONB       NOT NULL,
    state                SMALLINT    NOT NULL DEFAULT 0,  -- WarmTaskState
    total_snapshot_count BIGINT      NOT NULL DEFAULT 0,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_snapshot_warm_task_state
    ON snapshot_warm_task (state, created_at);

-- 2.4.2 SuperPod warm detail table.
-- superpod_id holds the JSONB array of planned target SuperPod IDs (one row per
-- snapshot), warmed_superpod_id the subset that has successfully warmed; state
-- becomes SUCCESS only when the two sets are equal.
CREATE TABLE IF NOT EXISTS superpod_warm_detail (
    snapshot_id        VARCHAR     PRIMARY KEY,
    category           SMALLINT    NOT NULL DEFAULT 0,  -- SnapshotCategory
    superpod_id        JSONB       NOT NULL DEFAULT '[]'::jsonb,
    warmed_superpod_id JSONB       NOT NULL DEFAULT '[]'::jsonb,
    state              SMALLINT    NOT NULL DEFAULT 0,  -- SnapshotWarmState
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_superpod_warm_detail_state
    ON superpod_warm_detail (state, updated_at DESC, snapshot_id ASC);
