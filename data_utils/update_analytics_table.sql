-- Adds detection count, narration, and source tracking columns
-- to the existing vision_pipeline_benchmark_analytics table.

ALTER TABLE vision_pipeline_benchmark_analytics
    ADD COLUMN IF NOT EXISTS recorded_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS source_id       VARCHAR(128)    NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS model1_class    VARCHAR(64)     NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS model1_count    INTEGER         NOT NULL DEFAULT 0 CHECK (model1_count >= 0),
    ADD COLUMN IF NOT EXISTS model2_class    VARCHAR(64)     NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS model2_count    INTEGER         NOT NULL DEFAULT 0 CHECK (model2_count >= 0),
    ADD COLUMN IF NOT EXISTS narration       TEXT            NULL;

-- Remove defaults after adding (keeps schema clean for future inserts)
ALTER TABLE vision_pipeline_benchmark_analytics
    ALTER COLUMN source_id    DROP DEFAULT,
    ALTER COLUMN model1_class DROP DEFAULT,
    ALTER COLUMN model1_count DROP DEFAULT,
    ALTER COLUMN model2_class DROP DEFAULT,
    ALTER COLUMN model2_count DROP DEFAULT;

-- Filter all rows for a given source across multi-day runs
CREATE INDEX IF NOT EXISTS idx_vpba_source_id
    ON vision_pipeline_benchmark_analytics (source_id);

-- Time-range queries scoped to a source
CREATE INDEX IF NOT EXISTS idx_vpba_source_time
    ON vision_pipeline_benchmark_analytics (source_id, recorded_at DESC);

-- Efficiently retrieve only rows that have narration
CREATE INDEX IF NOT EXISTS idx_vpba_narration
    ON vision_pipeline_benchmark_analytics (recorded_at DESC)
    WHERE narration IS NOT NULL;