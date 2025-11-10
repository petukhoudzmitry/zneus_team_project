"""
Utility functions for inspecting experiment logs and plotting results.

Includes:
- load_log: read experiments CSV
- top_k: return best runs by metric
- helpers to extract history from TensorBoard or artifact files
- plot_top_val_rmse: visualize validation RMSE for top runs
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

from runners import CSV_PATH


try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    _HAS_TB = True
except Exception:
    # If tensorboard is not available, TB-based extraction will be skipped
    _HAS_TB = False


def load_log() -> pd.DataFrame:
    """Load experiments CSV log, return empty DF if missing or unreadable."""
    if os.path.exists(CSV_PATH):
        try:
            return pd.read_csv(CSV_PATH)
        except Exception as e:
            # If reading fails, warn and return an empty DataFrame so callers can handle gracefully
            print(f"Warning: failed to read CSV at {CSV_PATH}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def top_k(k: int = 5) -> pd.DataFrame:
    """Return top-k runs by best_val_mse (or empty DF)."""
    df = load_log()
    if df.empty:
        print("No experiments yet")
        return df

    if 'best_val_mse' not in df.columns:
        # Provide a helpful warning if the expected metric column is missing
        print("Warning: 'best_val_mse' column missing in log; returning first rows instead.")
        return df.head(k)
    return df.sort_values('best_val_mse').head(k)


def _load_history_from_tb(tb_logdir: str, scalar_name_keyword: str = 'val_rmse') -> Optional[pd.DataFrame]:
    """
    Try to extract scalar series from a TensorBoard logdir. Returns DataFrame or None.

    This function looks for a scalar tag containing scalar_name_keyword and falls back to any 'val' scalar.
    Requires the tensorboard package to be importable.
    """
    if not _HAS_TB:
        return None
    if not tb_logdir or not os.path.exists(tb_logdir):
        return None
    try:
        # Load event files and query scalars
        ea = EventAccumulator(tb_logdir, size_guidance={EventAccumulator.SCALARS: 0})
        ea.Reload()
        scalars = ea.Tags().get('scalars', [])

        candidate = None
        # Pick a scalar that matches the requested keyword (case-insensitive)
        for s in scalars:
            if scalar_name_keyword.lower() in s.lower():
                candidate = s
                break
        if candidate is None:
            # Fallback: pick any scalar that contains 'val' to try to get validation metrics
            candidate = next((s for s in scalars if 'val' in s.lower()), None)
        if candidate is None:
            return None
        vals = ea.Scalars(candidate)
        # Convert TensorBoard scalar entries to a simple DataFrame
        epochs = [v.step for v in vals]
        values = [v.value for v in vals]
        return pd.DataFrame({'epoch': epochs, 'val_rmse': values})
    except Exception as e:
        # Fail gracefully: don't raise from the utility function; print a warning instead
        print(f"Warning: failed to read TensorBoard logs at {tb_logdir}: {e}")
        return None


def _load_history(hist_path: Optional[str], tb_logdir: Optional[str]) -> Optional[pd.DataFrame]:
    """Try to load history DataFrame from pickle/csv or from TB fallback."""
    # Prefer explicit history file if provided and exists
    if isinstance(hist_path, str) and hist_path and os.path.exists(hist_path):
        try:
            # Support both pickle and CSV history files
            if hist_path.endswith('.pkl') or hist_path.endswith('.pickle'):
                return pd.read_pickle(hist_path)
            else:
                return pd.read_csv(hist_path)
        except Exception as e:
            # If loading fails, fall through to try TB logs
            print(f"Warning: failed to load history from {hist_path}: {e}")

    # If history file loading failed, attempt to extract from tensorboard logs
    if isinstance(tb_logdir, str) and tb_logdir:
        tb_hist = _load_history_from_tb(tb_logdir)
        if tb_hist is not None:
            return tb_hist


    return None


def plot_top_val_rmse(k: int = 5, show_plot: bool = True) -> pd.DataFrame:
    """
    Plot validation RMSE curves for the top-k runs.

    Returns the DataFrame of top-k runs used (may be empty).
    """
    df = top_k(k)
    if df.empty:
        return df

    plt.figure(figsize=(10, 6))
    plotted_any = False

    for _, r in df.iterrows():
        # Extract tag and artifact pointers from the log row
        tag = r.get('tag', '<no-tag>')
        hist_path = r.get('history_path') if 'history_path' in r.index else None
        tb_logdir = r.get('tb_logdir') if 'tb_logdir' in r.index else None

        # Try to obtain a history DataFrame from either a saved file or tensorboard logs
        hist_df = _load_history(hist_path, tb_logdir)
        if hist_df is None:
            # Skip runs without retrievable history
            print(f"Skipping {tag}: no history found (checked history_path and tb_logdir).")
            continue

        # Ensure we have an 'epoch' column (reset index if needed)
        if 'epoch' not in hist_df.columns:
            hist_df = hist_df.reset_index().rename(columns={'index': 'epoch'})

        # Ensure we have val_rmse; if only val_mse available, convert it
        if 'val_rmse' not in hist_df.columns:
            if 'val_mse' in hist_df.columns:
                try:
                    hist_df['val_rmse'] = hist_df['val_mse'] ** 0.5
                except Exception:
                    # Last-resort conversion for unexpected types
                    hist_df['val_rmse'] = hist_df['val_mse'].apply(lambda x: float(x) ** 0.5 if pd.notna(x) else x)
            else:
                print(f"Skipping {tag}: neither 'val_rmse' nor 'val_mse' present in history.")
                continue

        # Sort by epoch to ensure a proper curve
        try:
            hist_df = hist_df.sort_values('epoch')
        except Exception:
            pass

        # Prepare a concise legend label
        label = f"{tag} | {r.get('best_val_mse', float('nan')):.5f}" if 'best_val_mse' in r.index else tag
        try:
            plt.plot(hist_df['epoch'], hist_df['val_rmse'], label=label)
            plotted_any = True
        except Exception as e:
            print(f"Warning: failed to plot {tag}: {e}")

    plt.xlabel('epoch')
    plt.ylabel('val_rmse')
    plt.title(f"Top {k} runs")

    # Only show legend if any labeled lines were drawn
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    else:
        print("Note: no labeled lines to show in legend (maybe all runs were skipped).")

    plt.grid(True)
    if show_plot:
        plt.show()

    return df
