"""
Phase 9: Scaling Laws + Connectivity Threshold Formula

Part A: Extend BFS to delta=2 for Phase 7 functions (see curve shape along s)
Part B: Test connectivity threshold formula  s_c(f; tau) = C(f) + 1
Part C: Fit C(f) vs D(f) scaling models:
        linear  C ~ a*D + b
        log     C ~ a*log(D) + b
        exp     D ~ alpha * exp(beta * C)

Output: data/scaling/threshold.json
        data/scaling/fits.json
"""

import json
import time
import os
import numpy as np
from scipy import stats as sp_stats
from phase7_reachability import parse_layout, find_components


# ====================================================================
# Part A: Extend BFS to delta=2
# ====================================================================

def extend_bfs_delta2(data_dir, phase7_results, n=4):
    """Run BFS at delta=2 for Phase 7 functions."""
    print("=" * 65, flush=True)
    print("PART A: BFS at delta=2", flush=True)
    print("=" * 65 + "\n", flush=True)

    sol_dir = os.path.join(data_dir, 'solutions')

    # Get unique functions from Phase 7
    seen = set()
    functions = []
    for r in phase7_results:
        if r['tt_hex'] not in seen:
            seen.add(r['tt_hex'])
            functions.append({
                'tt_hex': r['tt_hex'],
                'tt': None,  # loaded from Phase 1
                'circuit_size': r['circuit_size'],
            })

    # Load truth table values from Phase 1
    with open(os.path.join(data_dir, 'npn4_circuit_sizes.json')) as f:
        p1 = json.load(f)
    tt_map = {cls['tt_hex']: cls['canonical_tt'] for cls in p1['classes']}
    for func in functions:
        func['tt'] = tt_map[func['tt_hex']]

    delta2_results = []
    for i, func in enumerate(functions):
        c = func['circuit_size']
        s = c + 2
        tt_hex = func['tt_hex']
        tt = func['tt']

        sol_path = os.path.join(sol_dir, f"{tt_hex}_s{s}.npy")
        if not os.path.exists(sol_path):
            continue
        solutions = np.load(sol_path)
        if len(solutions) < 2:
            continue

        layout = parse_layout(n, s)
        t0 = time.time()
        components, assigned, total = find_components(
            solutions, n, s, tt, layout,
            max_nodes_per_comp=5000, max_total=20000)
        elapsed = time.time() - t0

        n_sol = len(solutions)
        n_comp = len(components)
        largest = max(c_['total_reachable'] for c_ in components) if components else 0
        max_dist = max(c_['max_distance'] for c_ in components) if components else 0
        all_exh = all(c_['exhaustive'] for c_ in components)

        result = {
            'tt_hex': tt_hex,
            'circuit_size': c,
            'delta': 2,
            's': s,
            'n_phase2_solutions': n_sol,
            'n_components': n_comp,
            'largest_component': largest,
            'max_graph_distance': max_dist,
            'total_bfs_explored': total,
            'all_exhaustive': all_exh,
            'component_details': components,
        }
        delta2_results.append(result)

        tag = "FULL" if all_exh else "CAPPED"
        print(f"  [{i+1}/{len(functions)}] {tt_hex} C={c} s={s}: "
              f"{n_comp} comp, largest={largest}, [{tag}] ({elapsed:.1f}s)",
              flush=True)

    return delta2_results


# ====================================================================
# Part B: Connectivity threshold formula
# ====================================================================

def test_threshold_formula(all_results):
    """Test s_c(f; tau) ≈ C(f) + 1 for various tau."""
    print(f"\n{'='*65}", flush=True)
    print("PART B: CONNECTIVITY THRESHOLD FORMULA", flush=True)
    print(f"{'='*65}\n", flush=True)

    # Group by function
    by_func = {}
    for r in all_results:
        key = r['tt_hex']
        by_func.setdefault(key, {})[r['delta']] = r

    # For each tau, find s_c(f; tau) and compare with C(f)+1
    taus = [0.25, 0.50, 0.75, 0.90]

    print(f"  {'tau':<8} {'s_c = C+1':<12} {'s_c = C':<10} {'s_c = C+2':<10} "
          f"{'other':<8} {'n':<5}", flush=True)
    print(f"  {'-'*53}", flush=True)

    threshold_data = {}

    for tau in taus:
        counts = {-1: 0, 0: 0, 1: 0, 2: 0, 'other': 0}
        details = []

        for tt_hex, deltas in by_func.items():
            if 0 not in deltas:
                continue
            c = deltas[0]['circuit_size']

            # Find s_c: minimum delta where giant component ratio >= tau
            s_c_delta = None
            for d in sorted(deltas.keys()):
                r = deltas[d]
                total = r['total_bfs_explored']
                largest = r['largest_component']
                exh = r['all_exhaustive']

                # Giant component ratio
                if total > 0:
                    gcr = largest / total
                else:
                    gcr = 0

                # Capped BFS with largest >= 5000: component was still growing
                # when we stopped. This IS a giant component regardless of tau.
                capped_giant = (not exh and largest >= 5000)
                if gcr >= tau or capped_giant:
                    s_c_delta = d
                    break

            if s_c_delta is not None:
                gap = s_c_delta  # gap = s_c - C(f) = delta
                if gap in counts:
                    counts[gap] += 1
                else:
                    counts['other'] += 1
                details.append({'tt_hex': tt_hex, 'C': c, 's_c_delta': gap})
            else:
                counts[-1] += 1  # never reached threshold

        n_tested = sum(v for v in counts.values())
        print(f"  {tau:<8.2f} {counts[1]:<12} {counts[0]:<10} {counts[2]:<10} "
              f"{counts['other'] + counts[-1]:<8} {n_tested:<5}", flush=True)

        threshold_data[str(tau)] = {
            'counts': {str(k): v for k, v in counts.items()},
            'details': details,
        }

    # Strongest statement
    print(f"\n  For tau=0.25: s_c(f;0.25) - C(f) = 1 holds for "
          f"{threshold_data['0.25']['counts'].get('1', 0)}/{sum(threshold_data['0.25']['counts'].values())} "
          f"functions", flush=True)

    return threshold_data


# ====================================================================
# Part C: C(f) vs D(f) scaling models
# ====================================================================

def fit_scaling_models(data_dir):
    """Fit linear/log/exp models to C(f) vs H0_max_persistence."""
    print(f"\n{'='*65}", flush=True)
    print("PART C: C(f) vs D(f) SCALING MODELS", flush=True)
    print(f"{'='*65}\n", flush=True)

    # Load TDA features (all 216 delta=0 instances)
    with open(os.path.join(data_dir, 'tda_features.json')) as f:
        tda = json.load(f)

    d0 = [f for f in tda['features']
          if f['delta'] == 0 and f['H0_max_persistence'] > 0]
    print(f"  {len(d0)} delta=0 instances with D > 0\n", flush=True)

    C = np.array([f['circuit_size'] for f in d0], dtype=float)
    D = np.array([f['H0_max_persistence'] for f in d0], dtype=float)

    results = {}

    # --- Model 1: Linear  C = a*D + b ---
    slope, intercept, r, p, se = sp_stats.linregress(D, C)
    C_pred = slope * D + intercept
    ss_res = np.sum((C - C_pred) ** 2)
    ss_tot = np.sum((C - C.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    n = len(C)
    k = 2
    aic = n * np.log(ss_res / n) + 2 * k

    results['linear'] = {
        'formula': f'C = {slope:.4f} * D + {intercept:.4f}',
        'R2': round(r2, 4),
        'AIC': round(aic, 2),
        'residual_std': round(np.std(C - C_pred), 4),
        'params': {'a': round(slope, 6), 'b': round(intercept, 4)},
    }
    print(f"  Linear:  C = {slope:.4f}*D + {intercept:.2f}  "
          f"R2={r2:.4f}  AIC={aic:.1f}", flush=True)

    # --- Model 2: Log  C = a*log(D) + b ---
    logD = np.log(D)
    slope2, intercept2, r2_, p2, se2 = sp_stats.linregress(logD, C)
    C_pred2 = slope2 * logD + intercept2
    ss_res2 = np.sum((C - C_pred2) ** 2)
    r2_log = 1 - ss_res2 / ss_tot
    aic2 = n * np.log(ss_res2 / n) + 2 * k

    results['logarithmic'] = {
        'formula': f'C = {slope2:.4f} * ln(D) + {intercept2:.4f}',
        'R2': round(r2_log, 4),
        'AIC': round(aic2, 2),
        'residual_std': round(np.std(C - C_pred2), 4),
        'params': {'a': round(slope2, 6), 'b': round(intercept2, 4)},
    }
    print(f"  Log:     C = {slope2:.4f}*ln(D) + {intercept2:.2f}  "
          f"R2={r2_log:.4f}  AIC={aic2:.1f}", flush=True)

    # --- Model 3: Exponential  D = alpha * exp(beta * C) ---
    # Equivalently: ln(D) = ln(alpha) + beta*C
    slope3, intercept3, r3, p3, se3 = sp_stats.linregress(C, logD)
    D_pred3 = np.exp(intercept3 + slope3 * C)
    # R2 in D-space
    ss_res3_D = np.sum((D - D_pred3) ** 2)
    ss_tot_D = np.sum((D - D.mean()) ** 2)
    r2_exp_D = 1 - ss_res3_D / ss_tot_D
    # R2 in log-space (where the fit was actually done)
    logD_pred3 = intercept3 + slope3 * C
    ss_res3_log = np.sum((logD - logD_pred3) ** 2)
    ss_tot_log = np.sum((logD - logD.mean()) ** 2)
    r2_exp_log = 1 - ss_res3_log / ss_tot_log
    aic3 = n * np.log(ss_res3_log / n) + 2 * k

    alpha = np.exp(intercept3)
    beta = slope3
    results['exponential'] = {
        'formula': f'D = {alpha:.4f} * exp({beta:.4f} * C)',
        'R2_logspace': round(r2_exp_log, 4),
        'R2_Dspace': round(r2_exp_D, 4),
        'AIC_logspace': round(aic3, 2),
        'residual_std_D': round(np.std(D - D_pred3), 4),
        'params': {'alpha': round(alpha, 6), 'beta': round(beta, 6)},
    }
    print(f"  Exp:     D = {alpha:.2f}*exp({beta:.4f}*C)  "
          f"R2(log)={r2_exp_log:.4f}  R2(D)={r2_exp_D:.4f}  AIC={aic3:.1f}",
          flush=True)

    # --- Model comparison ---
    print(f"\n  Model comparison (lower AIC = better):", flush=True)
    for name in ['linear', 'logarithmic', 'exponential']:
        r = results[name]
        r2_val = r.get('R2', r.get('R2_logspace', 0))
        aic_val = r.get('AIC', r.get('AIC_logspace', 0))
        print(f"    {name:<15} R2={r2_val:.4f}  AIC={aic_val:.1f}"
              f"  {r['formula']}", flush=True)

    # --- Per-complexity-level summary ---
    print(f"\n  Per-C(f) summary (mean D):", flush=True)
    for c in sorted(set(int(x) for x in C)):
        mask = C == c
        D_c = D[mask]
        print(f"    C={c}: n={sum(mask)}, mean D={D_c.mean():.1f}, "
              f"std={D_c.std():.1f}, range=[{D_c.min():.0f}, {D_c.max():.0f}]",
              flush=True)

    return results


# ====================================================================
# Part D: s-profile for representative functions
# ====================================================================

def s_profile(all_results):
    """Show how metrics evolve along the s direction for each function."""
    print(f"\n{'='*65}", flush=True)
    print("PART D: METRIC PROFILE ALONG s (delta=0,1,2)", flush=True)
    print(f"{'='*65}\n", flush=True)

    by_func = {}
    for r in all_results:
        by_func.setdefault(r['tt_hex'], {})[r['delta']] = r

    print(f"  {'func':<8} {'C':<4}", end='', flush=True)
    for d in [0, 1, 2]:
        print(f" | comp(d={d})", end='')
        print(f"  lrg(d={d})", end='')
    print(flush=True)
    print(f"  {'-'*80}", flush=True)

    for tt_hex in sorted(by_func.keys()):
        deltas = by_func[tt_hex]
        c = deltas[min(deltas.keys())]['circuit_size']
        print(f"  {tt_hex:<8} {c:<4}", end='', flush=True)
        for d in [0, 1, 2]:
            if d in deltas:
                r = deltas[d]
                print(f" | {r['n_components']:>5}   {r['largest_component']:>6}", end='')
            else:
                print(f" |     ?        ?", end='')
        print(flush=True)

    # Aggregate by complexity
    print(f"\n  Aggregate (mean):", flush=True)
    print(f"  {'C(f)':<6}", end='', flush=True)
    for d in [0, 1, 2]:
        print(f" | comp(d={d})  lrg(d={d})", end='')
    print(flush=True)
    print(f"  {'-'*70}", flush=True)

    for c in sorted(set(r['circuit_size'] for r in all_results)):
        print(f"  C={c:<4}", end='', flush=True)
        for d in [0, 1, 2]:
            sub = [r for r in all_results
                   if r['circuit_size'] == c and r['delta'] == d]
            if sub:
                mc = np.mean([r['n_components'] for r in sub])
                ml = np.mean([r['largest_component'] for r in sub])
                print(f" | {mc:>7.1f}  {ml:>8.0f}", end='')
            else:
                print(f" |       ?         ?", end='')
        print(flush=True)


# ====================================================================
# Main
# ====================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    out_dir = os.path.join(data_dir, 'scaling')
    os.makedirs(out_dir, exist_ok=True)

    # Load Phase 7 results
    with open(os.path.join(data_dir, 'reachability', 'summary.json')) as f:
        phase7 = json.load(f)
    phase7_results = phase7['results']

    t_start = time.time()

    # Part A: Extend to delta=2
    delta2 = extend_bfs_delta2(data_dir, phase7_results)

    # Merge all results
    all_results = phase7_results + delta2

    # Part B: Threshold formula
    threshold = test_threshold_formula(all_results)

    # Part C: Scaling models
    fits = fit_scaling_models(data_dir)

    # Part D: s-profile
    s_profile(all_results)

    total_time = time.time() - t_start

    # Save
    with open(os.path.join(out_dir, 'threshold.json'), 'w') as f:
        json.dump(threshold, f, indent=2)
    with open(os.path.join(out_dir, 'fits.json'), 'w') as f:
        json.dump(fits, f, indent=2)

    print(f"\nTotal time: {total_time:.0f}s", flush=True)
    print(f"Saved to {out_dir}/", flush=True)


if __name__ == '__main__':
    main()
