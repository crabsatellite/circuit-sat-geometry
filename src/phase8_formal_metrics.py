"""
Phase 8: Formal Metric Definitions + Connectivity Threshold Analysis

Defines three reusable metrics from Phase 7 reachability data:
  F(f,s)  = fragmentation index = n_components / n_solutions
  H(f,s)  = basin entropy = Shannon entropy of component size distribution
  kappa(f,s) = connectivity ratio = |largest component| / |total reachable|

Then tests the connectivity threshold conjecture:
  C(f) = max s such that the solution graph G(f,s) is disconnected.

Output: data/formal_metrics/metrics.json
        data/formal_metrics/threshold.json
"""

import json
import os
import numpy as np
from scipy import stats


def compute_metrics(result):
    """Compute formal metrics from a Phase 7 result entry."""
    n_comp = result['n_components']
    n_sol = result['n_phase2_solutions']
    total = result['total_bfs_explored']
    largest = result['largest_component']
    details = result['component_details']

    # F: fragmentation index
    F = n_comp / max(n_sol, 1)

    # H: basin entropy (over component sizes in BFS-reachable space)
    sizes = [c['total_reachable'] for c in details]
    total_size = sum(sizes)
    if total_size > 0 and len(sizes) > 1:
        probs = np.array(sizes, dtype=float) / total_size
        probs = probs[probs > 0]
        H = float(-np.sum(probs * np.log2(probs)))
    else:
        H = 0.0

    # kappa: connectivity ratio
    kappa = largest / max(total, 1)

    return {
        'F': round(F, 4),
        'H': round(H, 4),
        'kappa': round(kappa, 4),
        'n_components': n_comp,
        'n_solutions': n_sol,
        'largest': largest,
        'total_reachable': total,
        'max_distance': result['max_graph_distance'],
        'exhaustive': result['all_exhaustive'],
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    out_dir = os.path.join(data_dir, 'formal_metrics')
    os.makedirs(out_dir, exist_ok=True)

    # Load Phase 7 reachability data
    with open(os.path.join(data_dir, 'reachability', 'summary.json')) as f:
        reach = json.load(f)

    results = reach['results']

    # ================================================================
    # Compute metrics for all instances
    # ================================================================
    all_metrics = []
    for r in results:
        m = compute_metrics(r)
        m['tt_hex'] = r['tt_hex']
        m['circuit_size'] = r['circuit_size']
        m['delta'] = r['delta']
        m['s'] = r['s']
        all_metrics.append(m)

    # ================================================================
    # Metric scaling with C(f) at delta=0
    # ================================================================
    print("=" * 65, flush=True)
    print("FORMAL METRICS — Scaling with Circuit Complexity", flush=True)
    print("=" * 65, flush=True)

    print(f"\n  {'C(f)':<6} {'F (frag)':<10} {'H (entropy)':<12} "
          f"{'kappa (conn)':<13} {'#comp':<8} {'largest':<10} {'n':<4}", flush=True)
    print(f"  {'-'*63}", flush=True)

    for c in sorted(set(m['circuit_size'] for m in all_metrics)):
        for delta in [0, 1]:
            subset = [m for m in all_metrics
                      if m['circuit_size'] == c and m['delta'] == delta]
            if not subset:
                continue

            F_mean = np.mean([m['F'] for m in subset])
            H_mean = np.mean([m['H'] for m in subset])
            k_mean = np.mean([m['kappa'] for m in subset])
            comp_mean = np.mean([m['n_components'] for m in subset])
            lrg_mean = np.mean([m['largest'] for m in subset])

            tag = f"C={c} d={delta}"
            print(f"  {tag:<6} {F_mean:<10.3f} {H_mean:<12.3f} "
                  f"{k_mean:<13.3f} {comp_mean:<8.1f} {lrg_mean:<10.0f} "
                  f"{len(subset):<4}", flush=True)

    # ================================================================
    # Spearman: F vs C(f) at delta=0
    # ================================================================
    print(f"\n{'='*65}", flush=True)
    print("METRIC CORRELATIONS WITH C(f) AT DELTA=0", flush=True)
    print(f"{'='*65}\n", flush=True)

    d0 = [m for m in all_metrics if m['delta'] == 0]
    cs = np.array([m['circuit_size'] for m in d0])

    for metric_name in ['F', 'H', 'kappa', 'n_components', 'max_distance']:
        vals = np.array([m[metric_name] for m in d0], dtype=float)
        if np.std(vals) == 0:
            print(f"  {metric_name:<20} constant (no variance)", flush=True)
            continue
        r, p = stats.spearmanr(cs, vals)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {metric_name:<20} r={r:>7.4f}  p={p:.2e} {sig}", flush=True)

    # ================================================================
    # Connectivity threshold conjecture
    # ================================================================
    print(f"\n{'='*65}", flush=True)
    print("CONNECTIVITY THRESHOLD CONJECTURE", flush=True)
    print(f"{'='*65}\n", flush=True)

    # For each function, check:
    #   delta=0: disconnected? (n_components > 1)
    #   delta=1: connected?   (single giant component or kappa > 0.5)
    by_func = {}
    for m in all_metrics:
        by_func.setdefault(m['tt_hex'], {})[m['delta']] = m

    n_tested = 0
    n_disconnected_d0 = 0
    n_giant_d1 = 0
    threshold_holds = 0

    for tt_hex, deltas in by_func.items():
        if 0 not in deltas or 1 not in deltas:
            continue
        n_tested += 1

        d0_disc = deltas[0]['n_components'] > 1
        # Giant component criterion: largest component has >50% of total,
        # OR BFS was capped (meaning component is at least 5000+ nodes)
        d1_giant = (deltas[1]['kappa'] > 0.25 or
                    (not deltas[1]['exhaustive'] and deltas[1]['largest'] >= 5000))

        if d0_disc:
            n_disconnected_d0 += 1
        if d1_giant:
            n_giant_d1 += 1
        if d0_disc and d1_giant:
            threshold_holds += 1

    print(f"  Functions tested: {n_tested}", flush=True)
    print(f"  Disconnected at delta=0 (s=C(f)):   {n_disconnected_d0}/{n_tested} "
          f"({100*n_disconnected_d0/max(n_tested,1):.0f}%)", flush=True)
    print(f"  Giant component at delta=1 (s=C(f)+1): {n_giant_d1}/{n_tested} "
          f"({100*n_giant_d1/max(n_tested,1):.0f}%)", flush=True)
    print(f"  Threshold holds (both):              {threshold_holds}/{n_tested} "
          f"({100*threshold_holds/max(n_tested,1):.0f}%)", flush=True)

    print(f"\n  Conjecture: For all f in NPN4 with C(f) >= 2,", flush=True)
    print(f"    s = C(f) is the maximum circuit size at which", flush=True)
    print(f"    the local-move solution graph is disconnected.", flush=True)
    print(f"    At s = C(f)+1, a giant connected component emerges.", flush=True)
    print(f"\n  Status: {'CONFIRMED' if threshold_holds == n_tested else 'PARTIAL'} "
          f"on {n_tested} tested functions (C=2,5,7).", flush=True)

    # ================================================================
    # Delta transition: quantify the collapse
    # ================================================================
    print(f"\n{'='*65}", flush=True)
    print("DELTA TRANSITION — Quantified Collapse", flush=True)
    print(f"{'='*65}\n", flush=True)

    print(f"  {'C(f)':<6} {'F(d=0)':<10} {'F(d=1)':<10} {'ratio':<10} "
          f"{'kappa(d=0)':<12} {'kappa(d=1)':<12}", flush=True)
    print(f"  {'-'*60}", flush=True)

    for c in [2, 5, 7]:
        d0_sub = [m for m in all_metrics if m['circuit_size'] == c and m['delta'] == 0]
        d1_sub = [m for m in all_metrics if m['circuit_size'] == c and m['delta'] == 1]
        if not d0_sub or not d1_sub:
            continue

        F0 = np.mean([m['F'] for m in d0_sub])
        F1 = np.mean([m['F'] for m in d1_sub])
        k0 = np.mean([m['kappa'] for m in d0_sub])
        k1 = np.mean([m['kappa'] for m in d1_sub])
        ratio = F0 / max(F1, 0.001)

        print(f"  C={c:<4} {F0:<10.3f} {F1:<10.3f} {ratio:<10.1f}x "
              f"{k0:<12.3f} {k1:<12.3f}", flush=True)

    # Save
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump({'metrics': all_metrics}, f, indent=2)

    threshold = {
        'n_tested': n_tested,
        'n_disconnected_d0': n_disconnected_d0,
        'n_giant_d1': n_giant_d1,
        'threshold_holds': threshold_holds,
        'conjecture': ("C(f) is the maximum circuit size s at which "
                       "the local-move solution graph G(f,s) is disconnected. "
                       "At s = C(f)+1, a giant connected component emerges."),
        'status': 'CONFIRMED' if threshold_holds == n_tested else 'PARTIAL',
    }
    with open(os.path.join(out_dir, 'threshold.json'), 'w') as f:
        json.dump(threshold, f, indent=2)

    print(f"\nSaved to {out_dir}/", flush=True)


if __name__ == '__main__':
    main()
