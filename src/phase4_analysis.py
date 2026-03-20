"""
Phase 4: Correlation analysis between TDA features and circuit complexity.

Analyses:
  1. TDA features vs circuit complexity (at minimum size, delta=0)
  2. Phase transition: how topology changes as s increases past C(f)
  3. Predictive power: can TDA features predict circuit complexity?
  4. Wasserstein distances between persistence diagrams

Output: data/analysis/correlations.json
        data/analysis/phase_transition.json
        data/analysis/prediction.json
"""

import json
import os
from collections import Counter
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def load_features(data_dir):
    feat_path = os.path.join(data_dir, 'tda_features.json')
    with open(feat_path) as f:
        data = json.load(f)
    return data['features']


def analysis_1_correlations(features):
    """Correlate TDA features with circuit complexity at delta=0 (minimum size)."""
    print("=== Analysis 1: TDA vs Circuit Complexity (delta=0) ===\n", flush=True)

    # Filter to delta=0 instances
    d0 = [f for f in features if f['delta'] == 0]
    if not d0:
        print("  No delta=0 instances found!", flush=True)
        return {}

    circuit_sizes = np.array([f['circuit_size'] for f in d0])
    print(f"  {len(d0)} instances at delta=0", flush=True)
    print(f"  Circuit sizes: {sorted(set(circuit_sizes))}", flush=True)

    # TDA feature names
    tda_keys = [k for k in d0[0].keys()
                if k.startswith('H') and '_' in k and k[1].isdigit()]

    results = {}
    print(f"\n  {'Feature':<30} {'Spearman r':>10} {'p-value':>10}", flush=True)
    print(f"  {'-'*50}", flush=True)

    for key in sorted(tda_keys):
        vals = np.array([f[key] for f in d0], dtype=float)
        if vals.std() == 0:
            continue
        r, p = stats.spearmanr(circuit_sizes, vals)
        results[key] = {'spearman_r': float(r), 'p_value': float(p)}
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {key:<30} {r:>10.4f} {p:>10.4e} {sig}", flush=True)

    # Also correlate n_solutions with circuit complexity
    n_sols = np.array([f['n_solutions'] for f in d0], dtype=float)
    if n_sols.std() > 0:
        r, p = stats.spearmanr(circuit_sizes, n_sols)
        results['n_solutions'] = {'spearman_r': float(r), 'p_value': float(p)}
        print(f"\n  {'n_solutions':<30} {r:>10.4f} {p:>10.4e}", flush=True)

    return results


def analysis_2_phase_transition(features):
    """Track how TDA features change as s increases past C(f)."""
    print("\n=== Analysis 2: Phase Transition (topology vs delta) ===\n", flush=True)

    # Group by circuit_size and delta
    by_cs = {}
    for f in features:
        cs = f['circuit_size']
        by_cs.setdefault(cs, {}).setdefault(f['delta'], []).append(f)

    results = {}
    for cs in sorted(by_cs.keys()):
        print(f"  Circuit size C={cs}:", flush=True)
        cs_result = {}
        for delta in sorted(by_cs[cs].keys()):
            subset = by_cs[cs][delta]
            h1_tp = np.mean([f['H1_total_persistence'] for f in subset])
            h1_bars = np.mean([f['H1_n_bars'] for f in subset])
            n_sol = np.mean([f['n_solutions'] for f in subset])
            cs_result[delta] = {
                'n_instances': len(subset),
                'mean_H1_total_persistence': float(h1_tp),
                'mean_H1_n_bars': float(h1_bars),
                'mean_n_solutions': float(n_sol),
            }
            print(f"    delta={delta}: n={len(subset)}, "
                  f"H1_persist={h1_tp:.2f}, "
                  f"H1_bars={h1_bars:.1f}, "
                  f"n_sol={n_sol:.0f}", flush=True)
        results[str(cs)] = cs_result

    return results


def analysis_3_prediction(features):
    """Can TDA features predict circuit complexity?"""
    print("\n=== Analysis 3: Predictive Power ===\n", flush=True)

    d0 = [f for f in features if f['delta'] == 0]
    if len(d0) < 10:
        print("  Too few instances for prediction", flush=True)
        return {}

    # Build feature matrix
    tda_keys = sorted([k for k in d0[0].keys()
                       if k.startswith('H') and '_' in k and k[1].isdigit()])
    X = np.array([[f[k] for k in tda_keys] for f in d0], dtype=float)
    y = np.array([f['circuit_size'] for f in d0])

    # Also add n_solutions as a feature
    n_sol = np.array([f['n_solutions'] for f in d0], dtype=float).reshape(-1, 1)
    X_full = np.hstack([X, n_sol])
    tda_keys_full = tda_keys + ['n_solutions']

    # Handle NaN/inf
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    results = {}

    # Random Forest (robust, handles non-linear)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = min(5, len(np.unique(y)))
    if cv >= 2 and len(y) >= cv * 2:
        scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy')
        results['rf_accuracy'] = {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'cv_folds': cv,
        }
        print(f"  Random Forest {cv}-fold CV accuracy: "
              f"{scores.mean():.3f} +/- {scores.std():.3f}", flush=True)

        # Feature importance
        rf.fit(X_scaled, y)
        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        print(f"  Top features:", flush=True)
        for idx in top_idx:
            if importances[idx] > 0.01:
                print(f"    {tda_keys_full[idx]}: {importances[idx]:.4f}", flush=True)
        results['feature_importances'] = {
            tda_keys_full[i]: float(importances[i])
            for i in top_idx if importances[i] > 0.01
        }

    # Baseline: always predict most common class
    most_common_count = Counter(y).most_common(1)[0][1]
    baseline = most_common_count / len(y)
    results['baseline_accuracy'] = float(baseline)
    print(f"  Baseline (majority class): {baseline:.3f}", flush=True)

    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    analysis_dir = os.path.join(data_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    features = load_features(data_dir)
    print(f"Loaded {len(features)} feature vectors\n", flush=True)

    # Run analyses
    corr = analysis_1_correlations(features)
    phase = analysis_2_phase_transition(features)
    pred = analysis_3_prediction(features)

    # Save
    with open(os.path.join(analysis_dir, 'correlations.json'), 'w') as f:
        json.dump(corr, f, indent=2)
    with open(os.path.join(analysis_dir, 'phase_transition.json'), 'w') as f:
        json.dump(phase, f, indent=2)
    with open(os.path.join(analysis_dir, 'prediction.json'), 'w') as f:
        json.dump(pred, f, indent=2)

    print(f"\nResults saved to {analysis_dir}", flush=True)


if __name__ == '__main__':
    main()
