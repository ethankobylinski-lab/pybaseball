# Data Lineage

1. **Ingestion** – `etl/ingest.py` pulls raw JSON from the MLB StatsAPI.  Responses
   are cached in `data/raw` to support reproducible snapshots.
2. **Normalisation** – `etl/normalize.py` converts the raw payloads into canonical
   Polars tables (`games`, `pbp`, `box_team`, `box_batting`, `box_pitching`).
3. **Warehouse** – `etl/warehouse.sql` defines DuckDB views that transform the
   canonical tables into modelling datasets.  The SQL is shared between the
   analytical notebooks, models, and the interactive application.
4. **Features** – `features/pipeline.py` loads the canonical tables into DuckDB
   and materialises the feature views under the control of
   `features.registry.FeatureRegistry`.
5. **Models** – `models/training.py` fits logistic regression and LightGBM
   models, and `models/evaluation.py` reports metrics for backtests.
6. **Application** – `app/data.py` discovers season snapshots automatically and
   `app/main.py` serves the `Discoveries` and `Model Builder` experiences using
   Streamlit.

To retarget the pipeline for another league (college, youth), implement an
alternative ingestion module that populates the canonical tables with the same
schema.  The downstream SQL and model code will operate unchanged.

## Extending to Amateur Levels

1. Write a new ingestion client in `etl/ingest.py` (or a sibling module) that
   knows how to download the amateur feeds.  The output should be identical to
   the MLB StatsAPI JSON, or mapped to it.
2. Reuse `etl/normalize.py` to populate the canonical tables.  If extra fields
   are available (e.g., pitch types) they can be appended as optional columns.
3. Drop the resulting parquet snapshots into `data/warehouse` to power the
   Streamlit app, or pipe them straight into DuckDB via `features/pipeline.py`.
4. Update the docs to describe the data source and any assumptions unique to the
   new league.
