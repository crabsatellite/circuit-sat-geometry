"""
Phase 6: Robustness Validation Experiments

Three experiments to validate that the topology-complexity relationship
is a structural property of circuit-SAT solution spaces, not an encoding artifact.

R1: Encoding Invariance — re-sample WITHOUT symmetry breaking
    If the signal survives, it's not an artifact of the left<=right constraint.

R2: Metric Robustness — alternative distance metrics on existing solution data
    (a) connection-only Hamming  (b) function-only Hamming
    (c) binary connection encoding  (d) weighted Hamming (vary conn/func ratio)
    (e) random projection to low-dim Euclidean
    If the signal survives, it's not a Hamming/one-hot artifact.

R3: Null Models — controls to rule out spurious correlation
    (a) fixed sample size (N=50) — controls for solution count confound
    (b) column-shuffled null — destroys circuit structure, preserves marginals
    (c) label permutation test (1000 permutations) — statistical significance

Output: data/robustness/summary.json
"""

import json
import time
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from ripser import ripser

from phase1_circuit_sizes import CircuitSATSkeleton
from phase2_sample_solutions import enumerate_solutions
from phase3_persistent_homology import hamming_distance_matrix, extract_features


def load_experiment_data(data_dir):
    with open(os.path.join(data_dir, 'npn4_circuit_sizes.json')) as f:
        p1 = json.load(f)
    with open(os.path.join(data_dir, 'tda_features.json')) as f:
        tda = json.load(f)
    return p1['classes'], tda['features']


def tda_from_dist(dist_matrix, max_dim=2):
    result = ripser(dist_matrix, maxdim=max_dim, distance_matrix=True)
    return extract_features(result['dgms'], max_dim)


def spearman(cs, vals, label):
    r, p = stats.spearmanr(cs, vals)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {label:<35} r={r:.4f}  p={p:.2e} {sig}", flush=True)
    return {'spearman_r': round(float(r), 5), 'p_value': float(p)}


def parse_structural_layout(n, s):
    """Return per-gate offset ranges for left/right connections and function bits."""
    layout = []
    offset = 0
    for g in range(s):
        nw = n + g
        left = (offset, offset + nw);     offset += nw
        right = (offset, offset + nw);    offset += nw
        func = (offset, offset + 4);      offset += 4
        layout.append({'left': left, 'right': right, 'func': func})
    return layout


# ====================================================================
# R1: Encoding Invariance — No Symmetry Breaking
# ====================================================================

def r1_no_symmetry_breaking(data_dir, classes, orig_features, n=4):
    print("\n" + "=" * 65, flush=True)
    print("R1: ENCODING INVARIANCE — Re-sample WITHOUT symmetry breaking", flush=True)
    print("=" * 65 + "\n", flush=True)

    orig_d0 = {f['tt_hex']: f for f in orig_features if f['delta'] == 0}

    # Select representative subset: C in {3,4,5,6,7}
    by_c = {}
    for cls in classes:
        c = cls['circuit_size']
        if c in [3, 4, 5, 6, 7]:
            by_c.setdefault(c, []).append(cls)

    selected = []
    rng = np.random.default_rng(42)
    for c in sorted(by_c):
        group = by_c[c]
        if len(group) > 15:
            idx = rng.choice(len(group), 15, replace=False)
            group = [group[i] for i in idx]
        selected.extend(group)

    print(f"  Selected {len(selected)} instances: " +
          ", ".join(f"C={c}:{len(g)}" for c, g in sorted(by_c.items())),
          flush=True)

    results = []
    for i, cls in enumerate(selected):
        tt = cls['canonical_tt']
        tt_hex = cls['tt_hex']
        c = cls['circuit_size']
        s = max(c, 1)  # delta=0

        skeleton = CircuitSATSkeleton(n, s, symmetry_breaking=False)
        struct_vars = skeleton.get_structural_var_ids()

        time_limit = 30.0 if s >= 6 else 60.0
        solutions = enumerate_solutions(
            skeleton, tt, 1000, struct_vars, time_limit=time_limit)

        n_sol = solutions.shape[0]
        if n_sol < 3:
            continue

        if n_sol > 300:
            sub_rng = np.random.default_rng(42)
            idx = sub_rng.choice(n_sol, 300, replace=False)
            solutions = solutions[idx]

        dm = hamming_distance_matrix(solutions)
        feat = tda_from_dist(dm)
        feat['tt_hex'] = tt_hex
        feat['circuit_size'] = c
        feat['n_solutions_nosym'] = n_sol
        results.append(feat)

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(selected)}] {tt_hex} C={c} s={s}: "
                  f"{n_sol} solutions", flush=True)

    print(f"\n  No-symmetry-breaking ({len(results)} instances):", flush=True)
    cs = np.array([r['circuit_size'] for r in results])
    corr = {}
    for key in ['H0_max_persistence', 'H0_std_persistence',
                'H0_mean_persistence', 'H1_total_persistence']:
        vals = np.array([r[key] for r in results])
        corr[key] = spearman(cs, vals, f"NoSym  {key}")

    # Original correlations on same subset for direct comparison
    print(f"\n  Original (with symmetry breaking, same instances):", flush=True)
    matched = [orig_d0[r['tt_hex']] for r in results if r['tt_hex'] in orig_d0]
    cs_m = np.array([f['circuit_size'] for f in matched])
    orig_corr = {}
    for key in ['H0_max_persistence', 'H0_std_persistence']:
        vals = np.array([f[key] for f in matched])
        orig_corr[key] = spearman(cs_m, vals, f"Orig   {key}")

    return {'n_instances': len(results), 'nosym_correlations': corr,
            'orig_correlations_matched': orig_corr}


# ====================================================================
# R2: Metric Robustness
# ====================================================================

def r2_metric_robustness(data_dir, orig_features, n=4):
    print("\n" + "=" * 65, flush=True)
    print("R2: METRIC ROBUSTNESS — Alternative Distance Metrics", flush=True)
    print("=" * 65 + "\n", flush=True)

    sol_dir = os.path.join(data_dir, 'solutions')
    d0 = [f for f in orig_features if f['delta'] == 0 and f['n_solutions'] >= 10]
    print(f"  {len(d0)} delta=0 instances with >=10 solutions\n", flush=True)

    variants = {
        'conn_only': [],       # connection bits only
        'func_only': [],       # gate function bits only
        'binary_conn': [],     # one-hot → binary encoding for connections
        'weighted_conn2x': [], # connections weighted 2x vs functions
        'weighted_func2x': [], # functions weighted 2x vs connections
        'random_proj_30': [],  # random projection → 30-dim Euclidean
    }

    for i, inst in enumerate(d0):
        tt_hex = inst['tt_hex']
        s = inst['target_s']
        c = inst['circuit_size']

        sol_path = os.path.join(sol_dir, f"{tt_hex}_s{s}.npy")
        if not os.path.exists(sol_path):
            continue
        solutions = np.load(sol_path)

        if len(solutions) > 300:
            sub_rng = np.random.default_rng(42)
            idx = sub_rng.choice(len(solutions), 300, replace=False)
            solutions = solutions[idx]
        if len(solutions) < 3:
            continue

        layout = parse_structural_layout(n, s)

        # Gather index sets
        conn_idx = []
        func_idx = []
        for g_info in layout:
            conn_idx.extend(range(*g_info['left']))
            conn_idx.extend(range(*g_info['right']))
            func_idx.extend(range(*g_info['func']))

        # --- conn_only ---
        dm = hamming_distance_matrix(solutions[:, conn_idx])
        feat = tda_from_dist(dm)
        feat['circuit_size'] = c; feat['tt_hex'] = tt_hex
        variants['conn_only'].append(feat)

        # --- func_only ---
        dm = hamming_distance_matrix(solutions[:, func_idx])
        feat = tda_from_dist(dm)
        feat['circuit_size'] = c; feat['tt_hex'] = tt_hex
        variants['func_only'].append(feat)

        # --- binary_conn: one-hot → binary encoding ---
        binary_rows = []
        for row in solutions:
            bits = []
            for g_info in layout:
                for side in ['left', 'right']:
                    start, end = g_info[side]
                    wire = int(np.argmax(row[start:end]))
                    n_bits = max(1, int(np.ceil(np.log2(max(end - start, 2)))))
                    for b in range(n_bits):
                        bits.append((wire >> b) & 1)
                f_s, f_e = g_info['func']
                bits.extend(row[f_s:f_e].tolist())
            binary_rows.append(bits)
        sol_bin = np.array(binary_rows, dtype=np.uint8)
        dm = hamming_distance_matrix(sol_bin)
        feat = tda_from_dist(dm)
        feat['circuit_size'] = c; feat['tt_hex'] = tt_hex
        variants['binary_conn'].append(feat)

        # --- weighted Hamming: connections 2x ---
        sol_f = solutions.astype(np.float64)
        weights = np.ones(sol_f.shape[1])
        weights[conn_idx] = 2.0
        dm = squareform(pdist(sol_f * weights, metric='cityblock'))
        feat = tda_from_dist(dm)
        feat['circuit_size'] = c; feat['tt_hex'] = tt_hex
        variants['weighted_conn2x'].append(feat)

        # --- weighted Hamming: functions 2x ---
        weights2 = np.ones(sol_f.shape[1])
        weights2[func_idx] = 2.0
        dm = squareform(pdist(sol_f * weights2, metric='cityblock'))
        feat = tda_from_dist(dm)
        feat['circuit_size'] = c; feat['tt_hex'] = tt_hex
        variants['weighted_func2x'].append(feat)

        # --- random projection ---
        proj_rng = np.random.default_rng(42)
        d_proj = 30
        P = proj_rng.standard_normal((solutions.shape[1], d_proj)) / np.sqrt(d_proj)
        sol_proj = solutions.astype(np.float64) @ P
        dm = squareform(pdist(sol_proj, metric='euclidean'))
        feat = tda_from_dist(dm)
        feat['circuit_size'] = c; feat['tt_hex'] = tt_hex
        variants['random_proj_30'].append(feat)

        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(d0)}]", flush=True)

    all_corr = {}
    for name, feats in variants.items():
        if len(feats) < 5:
            continue
        print(f"\n  {name} ({len(feats)} instances):", flush=True)
        cs = np.array([f['circuit_size'] for f in feats])
        corr = {}
        for key in ['H0_max_persistence', 'H0_std_persistence',
                    'H1_total_persistence']:
            vals = np.array([f[key] for f in feats])
            if np.std(vals) == 0:
                corr[key] = {'spearman_r': 0.0, 'p_value': 1.0}
                print(f"    {key:<35} constant (no variance)", flush=True)
            else:
                corr[key] = spearman(cs, vals, key)
        all_corr[name] = corr

    # Original for comparison
    print(f"\n  original_hamming ({len(d0)} instances):", flush=True)
    cs_orig = np.array([f['circuit_size'] for f in d0])
    orig_c = {}
    for key in ['H0_max_persistence', 'H0_std_persistence',
                'H1_total_persistence']:
        vals = np.array([f[key] for f in d0])
        orig_c[key] = spearman(cs_orig, vals, key)
    all_corr['original_hamming'] = orig_c

    return all_corr


# ====================================================================
# R3: Null Models
# ====================================================================

def r3_null_models(data_dir, orig_features, n=4):
    print("\n" + "=" * 65, flush=True)
    print("R3: NULL MODELS — Control Experiments", flush=True)
    print("=" * 65 + "\n", flush=True)

    sol_dir = os.path.join(data_dir, 'solutions')

    # --- R3a: Fixed sample size N=50 ---
    d0_50 = [f for f in orig_features
             if f['delta'] == 0 and f['n_solutions'] >= 50]
    print(f"  R3a: Fixed sample size N=50 ({len(d0_50)} instances)\n",
          flush=True)

    fixed_results = []
    for inst in d0_50:
        sol_path = os.path.join(sol_dir,
                                f"{inst['tt_hex']}_s{inst['target_s']}.npy")
        if not os.path.exists(sol_path):
            continue
        solutions = np.load(sol_path)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(solutions), 50, replace=False)
        solutions = solutions[idx]

        dm = hamming_distance_matrix(solutions)
        feat = tda_from_dist(dm)
        feat['circuit_size'] = inst['circuit_size']
        feat['tt_hex'] = inst['tt_hex']
        fixed_results.append(feat)

    cs = np.array([f['circuit_size'] for f in fixed_results])
    r3a = {}
    for key in ['H0_max_persistence', 'H0_std_persistence']:
        vals = np.array([f[key] for f in fixed_results])
        r3a[key] = spearman(cs, vals, f"FixedN50 {key}")
    r3a['n_instances'] = len(fixed_results)

    # --- R3b: Column-shuffled null model ---
    d0_10 = [f for f in orig_features
             if f['delta'] == 0 and f['n_solutions'] >= 10]
    print(f"\n  R3b: Column-shuffled null model ({len(d0_10)} instances)\n",
          flush=True)

    shuffled_results = []
    for inst in d0_10:
        sol_path = os.path.join(sol_dir,
                                f"{inst['tt_hex']}_s{inst['target_s']}.npy")
        if not os.path.exists(sol_path):
            continue
        solutions = np.load(sol_path)

        if len(solutions) > 300:
            sub_rng = np.random.default_rng(42)
            idx = sub_rng.choice(len(solutions), 300, replace=False)
            solutions = solutions[idx]

        # Independently shuffle each column → destroys row structure
        rng = np.random.default_rng(123)
        shuffled = solutions.copy()
        for col in range(shuffled.shape[1]):
            rng.shuffle(shuffled[:, col])

        dm = hamming_distance_matrix(shuffled)
        feat = tda_from_dist(dm)
        feat['circuit_size'] = inst['circuit_size']
        feat['tt_hex'] = inst['tt_hex']
        shuffled_results.append(feat)

    cs = np.array([f['circuit_size'] for f in shuffled_results])
    r3b = {}
    for key in ['H0_max_persistence', 'H0_std_persistence']:
        vals = np.array([f[key] for f in shuffled_results])
        r3b[key] = spearman(cs, vals, f"Shuffled {key}")
    r3b['n_instances'] = len(shuffled_results)

    # --- R3c: Label permutation test ---
    n_perm = 1000
    print(f"\n  R3c: Label permutation test ({n_perm} permutations)\n",
          flush=True)

    d0_all = [f for f in orig_features if f['delta'] == 0]
    cs_all = np.array([f['circuit_size'] for f in d0_all])
    h0_max = np.array([f['H0_max_persistence'] for f in d0_all])

    obs_r, obs_p = stats.spearmanr(cs_all, h0_max)

    rng = np.random.default_rng(42)
    null_rs = np.array([
        stats.spearmanr(rng.permutation(cs_all), h0_max)[0]
        for _ in range(n_perm)
    ])

    p_perm = float(np.mean(np.abs(null_rs) >= np.abs(obs_r)))
    print(f"    Observed r = {obs_r:.4f}", flush=True)
    print(f"    Null: mean={null_rs.mean():.4f}, std={null_rs.std():.4f}, "
          f"max|r|={np.abs(null_rs).max():.4f}", flush=True)
    print(f"    Permutation p < {max(p_perm, 1/n_perm):.4f} "
          f"(0/{n_perm} exceeded)" if p_perm == 0
          else f"    Permutation p = {p_perm:.4f}", flush=True)

    r3c = {
        'observed_r': round(float(obs_r), 5),
        'null_mean': round(float(null_rs.mean()), 5),
        'null_std': round(float(null_rs.std()), 5),
        'null_max_abs_r': round(float(np.abs(null_rs).max()), 5),
        'permutation_p': float(p_perm),
        'n_permutations': n_perm,
    }

    return {'r3a_fixed_sample': r3a, 'r3b_shuffled': r3b,
            'r3c_label_permutation': r3c}


# ====================================================================
# Summary table
# ====================================================================

def print_summary_table(r1, r2, r3):
    print("\n" + "=" * 65, flush=True)
    print("ROBUSTNESS SUMMARY — H0_max_persistence vs circuit complexity",
          flush=True)
    print("=" * 65, flush=True)
    key = 'H0_max_persistence'
    rows = []

    # Original
    if 'original_hamming' in r2:
        c = r2['original_hamming'].get(key, {})
        rows.append(('Original (Hamming + SymBreak)', c.get('spearman_r', '?'),
                      c.get('p_value', '?')))
    # R1
    if 'nosym_correlations' in r1:
        c = r1['nosym_correlations'].get(key, {})
        rows.append(('R1: No symmetry breaking', c.get('spearman_r', '?'),
                      c.get('p_value', '?')))
    # R2 variants
    for name in ['conn_only', 'func_only', 'binary_conn',
                 'weighted_conn2x', 'weighted_func2x', 'random_proj_30']:
        if name in r2:
            c = r2[name].get(key, {})
            rows.append((f'R2: {name}', c.get('spearman_r', '?'),
                          c.get('p_value', '?')))
    # R3a
    c = r3.get('r3a_fixed_sample', {}).get(key, {})
    if c:
        rows.append(('R3a: Fixed N=50', c.get('spearman_r', '?'),
                      c.get('p_value', '?')))
    # R3b
    c = r3.get('r3b_shuffled', {}).get(key, {})
    if c:
        rows.append(('R3b: Column-shuffled null', c.get('spearman_r', '?'),
                      c.get('p_value', '?')))
    # R3c
    r3c = r3.get('r3c_label_permutation', {})
    if r3c:
        rows.append(('R3c: Label permutation (1000x)',
                      r3c.get('observed_r', '?'),
                      f"p<{max(r3c.get('permutation_p', 1), 0.001):.3f}"))

    print(f"\n  {'Variant':<40} {'r':>8} {'p-value':>12}", flush=True)
    print(f"  {'-'*60}", flush=True)
    for name, r, p in rows:
        r_str = f"{r:.4f}" if isinstance(r, float) else str(r)
        p_str = f"{p:.2e}" if isinstance(p, float) else str(p)
        print(f"  {name:<40} {r_str:>8} {p_str:>12}", flush=True)


# ====================================================================
# Main
# ====================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    robust_dir = os.path.join(data_dir, 'robustness')
    os.makedirs(robust_dir, exist_ok=True)

    classes, features = load_experiment_data(data_dir)
    print(f"Loaded {len(classes)} NPN4 classes, {len(features)} TDA features\n",
          flush=True)

    t_start = time.time()

    # R2 + R3 first (fast, reuse existing data)
    r2 = r2_metric_robustness(data_dir, features)
    r3 = r3_null_models(data_dir, features)

    # R1 last (requires SAT solving, slower)
    r1 = r1_no_symmetry_breaking(data_dir, classes, features)

    total_time = time.time() - t_start

    print_summary_table(r1, r2, r3)

    summary = {
        'r1_encoding_invariance': r1,
        'r2_metric_robustness': r2,
        'r3_null_models': r3,
        'total_time_s': round(total_time, 2),
    }
    with open(os.path.join(robust_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll robustness experiments complete in {total_time:.0f}s",
          flush=True)
    print(f"Results saved to {robust_dir}/summary.json", flush=True)


if __name__ == '__main__':
    main()
