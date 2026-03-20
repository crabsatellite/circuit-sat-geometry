"""
Phase 3: Compute persistent homology on circuit-SAT solution spaces.

For each sampled solution matrix (Phase 2 output):
  - Compute pairwise Hamming distance matrix
  - Run Ripser to get persistence diagrams (H0, H1, H2)
  - Extract topological features: Betti curves, total persistence, etc.

Output: data/persistence/<tt_hex>_s<size>_pd.json  (persistence diagrams)
        data/tda_features.json                     (extracted features for all instances)
"""

import json
import time
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
from ripser import ripser


def hamming_distance_matrix(solutions):
    """Compute pairwise Hamming distance matrix."""
    # pdist with 'hamming' gives fraction; multiply by dim to get integer Hamming distance
    if len(solutions) < 2:
        return np.zeros((len(solutions), len(solutions)))
    dists = pdist(solutions, metric='hamming') * solutions.shape[1]
    return squareform(dists)


def extract_features(diagrams, max_dim=2):
    """Extract topological features from persistence diagrams."""
    features = {}
    for dim in range(max_dim + 1):
        if dim < len(diagrams):
            dgm = diagrams[dim]
            # Remove infinite persistence bars for feature extraction
            finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else np.empty((0, 2))
            lifetimes = finite[:, 1] - finite[:, 0] if len(finite) > 0 else np.array([])

            features[f'H{dim}_n_bars'] = len(dgm)
            features[f'H{dim}_n_finite'] = len(finite)
            features[f'H{dim}_total_persistence'] = float(lifetimes.sum()) if len(lifetimes) > 0 else 0.0
            features[f'H{dim}_max_persistence'] = float(lifetimes.max()) if len(lifetimes) > 0 else 0.0
            features[f'H{dim}_mean_persistence'] = float(lifetimes.mean()) if len(lifetimes) > 0 else 0.0
            features[f'H{dim}_std_persistence'] = float(lifetimes.std()) if len(lifetimes) > 0 else 0.0

            # Betti number at midpoint of persistence range
            if len(finite) > 0:
                all_births = finite[:, 0]
                all_deaths = finite[:, 1]
                mid = (all_births.min() + all_deaths.max()) / 2
                betti_mid = np.sum((finite[:, 0] <= mid) & (finite[:, 1] > mid))
                features[f'H{dim}_betti_mid'] = int(betti_mid)
            else:
                features[f'H{dim}_betti_mid'] = 0

            # Persistence entropy
            if len(lifetimes) > 0 and lifetimes.sum() > 0:
                probs = lifetimes / lifetimes.sum()
                probs = probs[probs > 0]
                features[f'H{dim}_entropy'] = float(-np.sum(probs * np.log(probs)))
            else:
                features[f'H{dim}_entropy'] = 0.0
        else:
            for suffix in ['n_bars', 'n_finite', 'total_persistence',
                           'max_persistence', 'mean_persistence',
                           'std_persistence', 'betti_mid', 'entropy']:
                features[f'H{dim}_{suffix}'] = 0

    return features


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    sol_dir = os.path.join(data_dir, 'solutions')
    pd_dir = os.path.join(data_dir, 'persistence')
    os.makedirs(pd_dir, exist_ok=True)

    config_path = os.path.join(script_dir, '..', 'configs', 'default.json')
    with open(config_path) as f:
        config = json.load(f)

    max_dim = config['max_homology_dim']
    subsample = config['tda_subsample']

    # Load Phase 2 metadata
    meta_path = os.path.join(data_dir, 'solutions_meta.json')
    with open(meta_path) as f:
        sol_meta = json.load(f)

    instances = sol_meta['instances']
    print(f"Processing {len(instances)} instances (max_dim={max_dim}, "
          f"subsample={subsample})\n", flush=True)

    all_features = []
    skipped = 0
    t_start = time.time()

    for i, inst in enumerate(instances):
        tt_hex = inst['tt_hex']
        s = inst['target_s']
        n_sol = inst['n_solutions']

        if n_sol < 3:
            # Need at least 3 points for meaningful homology
            skipped += 1
            continue

        # Resume support: check if persistence diagram already exists
        pd_path = os.path.join(pd_dir, f'{tt_hex}_s{s}_pd.json')
        if os.path.exists(pd_path):
            # Load existing results and extract features
            with open(pd_path) as f:
                pd_data = json.load(f)
            dgms = []
            for dim in range(max_dim + 1):
                key = f'H{dim}'
                if key in pd_data and pd_data[key]:
                    dgms.append(np.array(pd_data[key]))
                else:
                    dgms.append(np.empty((0, 2)))
            features = extract_features(dgms, max_dim)
            features['tt_hex'] = tt_hex
            features['tt'] = inst['tt']
            features['circuit_size'] = inst['circuit_size']
            features['target_s'] = s
            features['delta'] = inst['delta']
            features['n_solutions'] = n_sol
            features['n_used'] = min(n_sol, subsample)
            features['tda_time_s'] = 0.0
            all_features.append(features)
            continue

        # Load solutions
        sol_path = os.path.join(sol_dir, inst['file'])
        solutions = np.load(sol_path)

        # Subsample if needed
        if len(solutions) > subsample:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(solutions), subsample, replace=False)
            solutions = solutions[idx]

        # Compute distance matrix
        t0 = time.time()
        dist_matrix = hamming_distance_matrix(solutions)

        # Run Ripser
        result = ripser(dist_matrix, maxdim=max_dim, distance_matrix=True)
        tda_time = time.time() - t0

        # Extract features
        features = extract_features(result['dgms'], max_dim)
        features['tt_hex'] = tt_hex
        features['tt'] = inst['tt']
        features['circuit_size'] = inst['circuit_size']
        features['target_s'] = s
        features['delta'] = inst['delta']
        features['n_solutions'] = n_sol
        features['n_used'] = len(solutions)
        features['tda_time_s'] = round(tda_time, 4)
        all_features.append(features)

        # Save persistence diagram
        pd_data = {}
        for dim in range(max_dim + 1):
            if dim < len(result['dgms']):
                dgm = result['dgms'][dim]
                pd_data[f'H{dim}'] = dgm.tolist()
            else:
                pd_data[f'H{dim}'] = []
        with open(pd_path, 'w') as f:
            json.dump(pd_data, f)

        if (i + 1) % 25 == 0 or tda_time > 5:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(instances)}] {tt_hex} s={s}: "
                  f"H0={features['H0_n_bars']}, H1={features['H1_n_bars']}, "
                  f"H2={features['H2_n_bars']} "
                  f"({tda_time:.2f}s, {n_sol}pts) [{elapsed:.0f}s elapsed]",
                  flush=True)

    # Save all features
    feat_path = os.path.join(data_dir, 'tda_features.json')
    with open(feat_path, 'w') as f:
        json.dump({
            'n_instances': len(all_features),
            'n_skipped': skipped,
            'max_dim': max_dim,
            'features': all_features,
        }, f, indent=2)
    print(f"\nSaved {len(all_features)} feature vectors to {feat_path}", flush=True)
    print(f"Skipped {skipped} instances (< 3 solutions)", flush=True)

    # Quick summary
    if all_features:
        print("\n=== Quick TDA Summary ===", flush=True)
        for delta in range(4):
            subset = [f for f in all_features if f['delta'] == delta]
            if not subset:
                continue
            h0 = np.mean([f['H0_total_persistence'] for f in subset])
            h1 = np.mean([f['H1_total_persistence'] for f in subset])
            h1_bars = np.mean([f['H1_n_bars'] for f in subset])
            print(f"  delta={delta}: n={len(subset)}, "
                  f"mean H0_persist={h0:.1f}, "
                  f"mean H1_persist={h1:.1f}, "
                  f"mean H1_bars={h1_bars:.1f}", flush=True)


if __name__ == '__main__':
    main()
