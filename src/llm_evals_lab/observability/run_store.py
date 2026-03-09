"""
Run store: persistence layer for pipeline run traces.

Saves RunRecord objects as individual JSON files under results/runs/
and maintains a rolling CSV summary under results/tables/.

Design
------
- Each run is stored as {run_id}.json for full trace fidelity.
- A summary CSV (run_summary.csv) is maintained with flat metric columns
  for quick analysis without loading individual traces.
- The store supports loading all runs, filtering by experiment ID,
  and exporting to pandas DataFrames.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from llm_evals_lab.schemas import RunRecord
from llm_evals_lab.utils import _json_default

logger = logging.getLogger(__name__)

_SUMMARY_FILENAME = "run_summary.csv"


class RunStore:
    """
    Persistent store for pipeline run traces.

    Parameters
    ----------
    runs_dir : Path
        Directory where individual JSON traces are written.
    tables_dir : Path, optional
        Directory where CSV summaries are written.
        Defaults to ``runs_dir.parent / "tables"``.
    """

    def __init__(
        self,
        runs_dir: Path,
        tables_dir: Optional[Path] = None,
    ) -> None:
        self.runs_dir = runs_dir
        self.tables_dir = tables_dir or runs_dir.parent / "tables"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    # ── Write ─────────────────────────────────────────────────────────────────

    def save(self, record: RunRecord) -> Path:
        """
        Persist a RunRecord as JSON and update the summary CSV.

        Returns the path to the written JSON file.
        """
        json_path = self.runs_dir / f"{record.run_id}.json"
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(
                record.model_dump(),
                fh,
                indent=2,
                default=_json_default,
                ensure_ascii=False,
            )

        self._append_to_summary(record)
        logger.debug("Saved run %s to %s", record.run_id, json_path)
        return json_path

    def _append_to_summary(self, record: RunRecord) -> None:
        """Append one row to the rolling summary CSV."""
        summary_path = self.tables_dir / _SUMMARY_FILENAME
        row = record.to_summary_dict()
        new_df = pd.DataFrame([row])

        if summary_path.exists():
            existing = pd.read_csv(summary_path)
            # Avoid duplicates (idempotent save)
            existing = existing[existing["run_id"] != record.run_id]
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined.to_csv(summary_path, index=False)

    # ── Read ──────────────────────────────────────────────────────────────────

    def load(self, run_id: str) -> Optional[RunRecord]:
        """Load a single RunRecord by run_id."""
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            logger.warning("Run not found: %s", run_id)
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return RunRecord(**data)

    def load_all(self) -> list[RunRecord]:
        """Load all stored RunRecords from the runs directory."""
        records: list[RunRecord] = []
        for path in sorted(self.runs_dir.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                records.append(RunRecord(**data))
            except Exception as exc:
                logger.warning("Failed to load run from %s: %s", path, exc)
        return records

    def load_summary(self) -> pd.DataFrame:
        """Load the rolling summary CSV as a DataFrame."""
        path = self.tables_dir / _SUMMARY_FILENAME
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def filter_by_experiment(self, experiment_id: str) -> list[RunRecord]:
        """Load all runs belonging to a specific experiment group."""
        return [r for r in self.load_all() if r.experiment_id == experiment_id]

    def list_experiments(self) -> list[str]:
        """Return unique experiment IDs across all stored runs."""
        df = self.load_summary()
        if df.empty or "experiment_id" not in df.columns:
            return []
        return df["experiment_id"].dropna().unique().tolist()

    def to_dataframe(self, experiment_id: Optional[str] = None) -> pd.DataFrame:
        """
        Return a flat DataFrame of all (or filtered) run summaries.

        This is the primary interface for analysis and dashboard use.
        """
        df = self.load_summary()
        if experiment_id and not df.empty and "experiment_id" in df.columns:
            df = df[df["experiment_id"] == experiment_id]
        return df

    # ── Maintenance ───────────────────────────────────────────────────────────

    def rebuild_summary(self) -> pd.DataFrame:
        """
        Regenerate the summary CSV from all stored JSON traces.

        Useful after manual trace edits or data migrations.
        """
        records = self.load_all()
        if not records:
            return pd.DataFrame()
        rows = [r.to_summary_dict() for r in records]
        df = pd.DataFrame(rows)
        summary_path = self.tables_dir / _SUMMARY_FILENAME
        df.to_csv(summary_path, index=False)
        logger.info("Rebuilt summary from %d runs → %s", len(records), summary_path)
        return df

    @property
    def run_count(self) -> int:
        return len(list(self.runs_dir.glob("*.json")))
