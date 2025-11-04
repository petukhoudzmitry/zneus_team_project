import glob
import os
import json
import time
import traceback

import pandas as pd
from datetime import datetime

from experiments import run_experiment

EXPERIMENTS_DIR = "experiments"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(EXPERIMENTS_DIR, "experiments_log.csv")

def _exp_id():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def log_row(row):

    df = pd.DataFrame([row])
    if not os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, index=False)
    else:
        df.to_csv(CSV_PATH, mode='a', header=False, index=False)

def _discover_artifacts_for_tag(tag):
    """
    Search artifacts/ and runs/ for files related to tag.
    Returns (history_path_or_None, model_path_or_None, tb_logdir_or_None)
    """
    
    history_candidates = glob.glob(os.path.join("artifacts", f"history_{tag}_*.*"))
    model_candidates = glob.glob(os.path.join("artifacts", f"model_{tag}_*.*"))
    history_path = history_candidates[0] if history_candidates else None
    model_path = model_candidates[0] if model_candidates else None

    
    tb_dir = os.path.join("runs", tag)
    tb_logdir = tb_dir if os.path.isdir(tb_dir) else None

    
    if history_path is None:
        hist_prefix = os.path.join("artifacts", f"history_{tag.split('_')[0]}*")
        more = glob.glob(hist_prefix)
        history_path = more[0] if more else None
    if model_path is None:
        mod_prefix = os.path.join("artifacts", f"model_{tag.split('_')[0]}*")
        more = glob.glob(mod_prefix)
        model_path = more[0] if more else None

    return history_path, model_path, tb_logdir

def run_and_record(config, X_train, y_train, X_val, y_val, device='cpu', verbose=False):
    """
    Wrapper: run run_experiment, then write an experiments_log.csv row.
    This version will discover artifact files on disk if run_experiment did not return them.
    """
    
    if 'tag' not in config or not config['tag']:
        config['tag'] = f"exp_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    tag = config['tag']

    t0 = time.time()
    error_str = ""
    results = None
    try:
        model, results = run_experiment(X_train, y_train, X_val, y_val, config, device=device, verbose=verbose)
    except Exception as e:
        
        error_str = traceback.format_exc()
        if verbose:
            print("run_experiment error:", error_str)
        results = {'best_val_mse': float('nan'), 'history': pd.DataFrame(), 'config': config, 'duration_s': 0.0,
                   'model_path': None, 'history_path': None, 'tb_logdir': None}

    duration = time.time() - t0

    
    best_val_mse = results.get('best_val_mse', float('nan'))
    history_df = results.get('history', pd.DataFrame()) if results else pd.DataFrame()
    final_train_loss = float(history_df['train_loss'].iloc[-1]) if (not history_df.empty and 'train_loss' in history_df.columns) else float('nan')
    final_val_mse = float(history_df['val_mse'].iloc[-1]) if (not history_df.empty and 'val_mse' in history_df.columns) else float('nan')
    final_val_rmse = float(history_df['val_rmse'].iloc[-1]) if (not history_df.empty and 'val_rmse' in history_df.columns) else float('nan')

    
    history_path = results.get('history_path') if isinstance(results, dict) else None
    model_path = results.get('model_path') if isinstance(results, dict) else None
    tb_logdir = results.get('tb_logdir') if isinstance(results, dict) else None

    if not history_path or pd.isna(history_path):
        h, m, tb = _discover_artifacts_for_tag(tag)
        history_path = history_path or h
        model_path = model_path or m
        tb_logdir = tb_logdir or tb

    row = {
        "exp_id": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "tag": tag,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "duration_s": duration,
        "best_val_mse": best_val_mse,
        "final_train_loss": final_train_loss,
        "final_val_mse": final_val_mse,
        "final_val_rmse": final_val_rmse,
        "config": json.dumps(config, default=str),
        "tb_logdir": tb_logdir,
        "model_path": model_path,
        "history_path": history_path,
        "error": error_str
    }

    
    try:
        if not os.path.exists(CSV_PATH):
            pd.DataFrame([row]).to_csv(CSV_PATH, index=False)
        else:
            pd.DataFrame([row]).to_csv(CSV_PATH, mode='a', header=False, index=False)
    except Exception:
        print("Failed to write experiments_log.csv:", traceback.format_exc())

    return row



def run_plan_12(base_config, X_train, y_train, X_val, y_val, device='cpu'):
    plan = [{'tag': 'exp-01_baseline', **base_config},
            {'tag': 'exp-02_wd-1e-5', **{**base_config, 'weight_decay': 1e-5}},
            {'tag': 'exp-03_wd-1e-4', **{**base_config, 'weight_decay': 1e-4}},
            {'tag': 'exp-04_wd-1e-3', **{**base_config, 'weight_decay': 1e-3}},
            {'tag': 'exp-05_bn', **{**base_config, 'batchnorm': True}},
            {'tag': 'exp-06_do-0.05', **{**base_config, 'dropout': 0.05}},
            {'tag': 'exp-07_do-0.1', **{**base_config, 'dropout': 0.1}},
            {'tag': 'exp-08_bn_do-0.1', **{**base_config, 'batchnorm': True, 'dropout': 0.1}},
            {'tag': 'exp-09_smaller', **{**base_config, 'hidden_dims': [64, 32]}},
            {'tag': 'exp-10_bottleneck', **{**base_config, 'bottleneck': True}},
            {'tag': 'exp-11_skip', **{**base_config, 'skip': True}}, {'tag': 'exp-12_combined', **base_config}]

    rows = []
    for cfg in plan:
        print("Running:", cfg['tag'])
        row = run_and_record(cfg, X_train, y_train, X_val, y_val, device=device, verbose=False)
        rows.append(row)
    return pd.DataFrame(rows)
