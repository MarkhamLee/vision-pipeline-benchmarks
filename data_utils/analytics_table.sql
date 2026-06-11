CREATE TABLE IF NOT EXISTS vision_pipeline_benchmark_analytics (
    id              BIGSERIAL       PRIMARY KEY,
    recorded_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    source_id       VARCHAR(128)    NOT NULL,

    -- Model 1 detection pair (e.g. Person)
    model1_class    VARCHAR(64)     NOT NULL,
    model1_count    INTEGER         NOT NULL CHECK (model1_count >= 0),

    -- Model 2 detection pair (e.g. Car)
    model2_class    VARCHAR(64)     NOT NULL,
    model2_count    INTEGER         NOT NULL CHECK (model2_count >= 0),

    -- LLM scene narration (sparse — nullable)
    -- Populated every ~15 minutes by VLM; NULL on count-only writes
    narration       TEXT            NULL
);

-- Filter all rows for a given source across multi-day runs
CREATE INDEX IF NOT EXISTS idx_seq_analytics_source_id
    ON vision_pipeline_benchmark_analytics (source_id);

-- Time-range queries scoped to a source (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_seq_analytics_source_time
    ON vision_pipeline_benchmark_analytics (source_id, recorded_at DESC);

-- Efficiently retrieve only rows that have narration
CREATE INDEX IF NOT EXISTS idx_seq_analytics_narration
    ON vision_pipeline_benchmark_analytics (recorded_at DESC)
    WHERE narration IS NOT NULL