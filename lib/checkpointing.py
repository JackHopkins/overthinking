"""Checkpoint and resume utilities for long-running experiments."""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional


class ExperimentCheckpoint:
    """Manages experiment checkpoints for resumability.

    Provides:
    - CSV-based results storage with append capability
    - JSON sample storage per alpha/method combination
    - Progress tracking via completed key sets
    - Summary generation
    """

    def __init__(self, checkpoint_dir: str, experiment_name: str):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            experiment_name: Name of the experiment (used in filenames)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.results_file = self.checkpoint_dir / f"{experiment_name}_results.csv"
        self.config_file = self.checkpoint_dir / f"{experiment_name}_config.json"
        self.samples_dir = self.checkpoint_dir / f"{experiment_name}_samples"
        self.samples_dir.mkdir(exist_ok=True)

    def load_results(self) -> pd.DataFrame:
        """Load existing results or return empty DataFrame.

        Returns:
            DataFrame with cached results, or empty DataFrame
        """
        if self.results_file.exists():
            df = pd.read_csv(self.results_file)
            print(f"Loaded {len(df)} cached results from {self.results_file}")
            return df
        return pd.DataFrame()

    def save_results(self, df: pd.DataFrame):
        """Save results to checkpoint file.

        Args:
            df: DataFrame to save
        """
        df.to_csv(self.results_file, index=False)
        print(f"Saved {len(df)} results to {self.results_file}")

    def get_completed_keys(
        self,
        df: pd.DataFrame,
        key_columns: list[str],
    ) -> set[tuple]:
        """Get set of completed (alpha, method, sample_idx, ...) tuples.

        Args:
            df: DataFrame with results
            key_columns: Columns that form the unique key

        Returns:
            Set of tuples representing completed evaluations
        """
        if df.empty:
            return set()
        return set(tuple(row) for row in df[key_columns].values)

    def append_results(self, new_results: list[dict]) -> pd.DataFrame:
        """Append new results to existing checkpoint.

        Args:
            new_results: List of result dictionaries

        Returns:
            Combined DataFrame with all results
        """
        df = self.load_results()
        new_df = pd.DataFrame(new_results)
        combined = pd.concat([df, new_df], ignore_index=True)
        self.save_results(combined)
        return combined

    def save_samples(self, samples: list[dict], alpha: float, method: str):
        """Save generated samples for an alpha/method combination.

        Args:
            samples: List of sample dictionaries
            alpha: Alpha value
            method: Coefficient method name
        """
        filename = self.samples_dir / f"samples_a{alpha}_m_{method}.json"
        with open(filename, 'w') as f:
            json.dump(samples, f, indent=2)

    def load_samples(self, alpha: float, method: str) -> Optional[list[dict]]:
        """Load samples for an alpha/method combination if they exist.

        Args:
            alpha: Alpha value
            method: Coefficient method name

        Returns:
            List of sample dicts, or None if not found
        """
        filename = self.samples_dir / f"samples_a{alpha}_m_{method}.json"
        if filename.exists():
            with open(filename) as f:
                return json.load(f)
        return None

    def save_config(self, config: dict):
        """Save experiment configuration.

        Args:
            config: Configuration dictionary
        """
        config_copy = config.copy()
        config_copy["timestamp"] = datetime.now().isoformat()
        with open(self.config_file, 'w') as f:
            json.dump(config_copy, f, indent=2)

    def load_config(self) -> Optional[dict]:
        """Load experiment configuration if it exists.

        Returns:
            Configuration dict, or None if not found
        """
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        return None

    def generate_summary(
        self,
        df: pd.DataFrame,
        group_cols: list[str],
        metric_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Generate summary statistics grouped by specified columns.

        Args:
            df: DataFrame with results
            group_cols: Columns to group by (e.g., ['alpha', 'method'])
            metric_cols: Columns to aggregate (auto-detected if None)

        Returns:
            Summary DataFrame with mean, std, count per group
        """
        if df.empty:
            return pd.DataFrame()

        # Auto-detect numeric columns for aggregation
        if metric_cols is None:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            metric_cols = [c for c in numeric_cols if c not in group_cols]

        if not metric_cols:
            return pd.DataFrame()

        agg_dict = {col: ['mean', 'std', 'count'] for col in metric_cols}
        summary = df.groupby(group_cols).agg(agg_dict).round(3)

        return summary

    def get_progress(
        self,
        total_alphas: int,
        total_methods: int,
        total_samples: int,
    ) -> dict:
        """Get current progress summary.

        Args:
            total_alphas: Expected number of alpha values
            total_methods: Expected number of coefficient methods
            total_samples: Expected samples per alpha/method

        Returns:
            Dict with 'completed', 'total', 'percent' keys
        """
        df = self.load_results()
        completed = len(df) if not df.empty else 0
        total = total_alphas * total_methods * total_samples

        return {
            "completed": completed,
            "total": total,
            "percent": (completed / total * 100) if total > 0 else 0,
        }

    def clear(self, confirm: bool = False):
        """Clear all checkpoint data.

        Args:
            confirm: Must be True to actually clear
        """
        if not confirm:
            print("Pass confirm=True to clear checkpoint data")
            return

        if self.results_file.exists():
            self.results_file.unlink()
            print(f"Removed {self.results_file}")

        if self.config_file.exists():
            self.config_file.unlink()
            print(f"Removed {self.config_file}")

        for sample_file in self.samples_dir.glob("*.json"):
            sample_file.unlink()
            print(f"Removed {sample_file}")

        print("Checkpoint cleared")