"""Generate connectivity transition figure and D/d normalization analysis."""

import json
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')

    # ================================================================
    # Part 1: Connectivity Figure from reachability data
    # ================================================================

    with open(os.path.join(data_dir, 'reachability', 'summary.json')) as f:
        reach = json.load(f)

    results = reach['results']

    agg = {}
    for r in results:
        key = (r['circuit_size'], r['delta'])
        agg.setdefault(key, []).append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    complexities = [2, 5, 7]
    bar_width = 0.35
    x = np.arange(len(complexities))

    # Panel A: Number of components
    comp_d0 = []
    comp_d1 = []
    for c in complexities:
        d0 = agg.get((c, 0), [])
        d1 = agg.get((c, 1), [])
        comp_d0.append(np.mean([r['n_components'] for r in d0]) if d0 else 0)
        comp_d1.append(np.mean([r['n_components'] for r in d1]) if d1 else 0)

    bars1 = ax1.bar(x - bar_width/2, comp_d0, bar_width,
                    label=r'$\delta=0$ (min size)', color='#d62728', alpha=0.85)
    bars2 = ax1.bar(x + bar_width/2, comp_d1, bar_width,
                    label=r'$\delta=1$ (+1 gate)', color='#2ca02c', alpha=0.85)

    ax1.set_yscale('log')
    ax1.set_ylabel('Number of components (log scale)', fontsize=11)
    ax1.set_xlabel('Circuit complexity $C(f)$', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'$C={c}$' for c in complexities])
    ax1.legend(fontsize=9)
    ax1.set_title('(a) Component count', fontsize=12, fontweight='bold')

    for bar, val in zip(bars1, comp_d0):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, comp_d1):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Panel B: Largest component size
    lrg_d0 = []
    lrg_d1 = []
    lrg_d1_capped = []
    for c in complexities:
        d0 = agg.get((c, 0), [])
        d1 = agg.get((c, 1), [])
        lrg_d0.append(np.mean([r['largest_component'] for r in d0]) if d0 else 0)
        mean_lrg = np.mean([r['largest_component'] for r in d1]) if d1 else 0
        lrg_d1.append(mean_lrg)
        lrg_d1_capped.append(any(not r['all_exhaustive'] for r in d1))

    bars3 = ax2.bar(x - bar_width/2, lrg_d0, bar_width,
                    label=r'$\delta=0$ (min size)', color='#d62728', alpha=0.85)
    bars4 = ax2.bar(x + bar_width/2, lrg_d1, bar_width,
                    label=r'$\delta=1$ (+1 gate)', color='#2ca02c', alpha=0.85)

    ax2.set_yscale('log')
    ax2.set_ylabel('Largest component size (log scale)', fontsize=11)
    ax2.set_xlabel('Circuit complexity $C(f)$', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'$C={c}$' for c in complexities])
    ax2.legend(fontsize=9)
    ax2.set_title('(b) Largest component', fontsize=12, fontweight='bold')

    for bar, val in zip(bars3, lrg_d0):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    for i, (bar, val) in enumerate(zip(bars4, lrg_d1)):
        suffix = '+' if lrg_d1_capped[i] else ''
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                 f'{val:.0f}{suffix}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    fig_dir = os.path.join(script_dir, '..', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, 'fig_phase_transition.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved connectivity figure to {out_path}")
    plt.close()

    # ================================================================
    # Part 2: D/d normalization analysis
    # ================================================================

    with open(os.path.join(data_dir, 'tda_features.json')) as f:
        tda = json.load(f)

    features = tda['features']
    d0 = [f for f in features if f['delta'] == 0]

    print(f"\n{'='*65}")
    print("DIMENSIONAL CONFOUND ANALYSIS")
    print(f"{'='*65}")
    print(f"\nInstances at delta=0: {len(d0)}")

    for f in d0:
        s = f['target_s']
        f['dim'] = s * s + 11 * s
        f['D_over_d'] = f['H0_max_persistence'] / f['dim'] if f['dim'] > 0 else 0

    by_c = {}
    for f in d0:
        by_c.setdefault(f['circuit_size'], []).append(f)

    print(f"\n  {'C(f)':<6} {'N':<5} {'d(4,C)':<8} {'D mean':<10} {'D/d mean':<10} {'D/d std':<10}")
    print(f"  {'-'*49}")
    for c in sorted(by_c.keys()):
        group = by_c[c]
        d_val = group[0]['dim']
        D_mean = np.mean([f['H0_max_persistence'] for f in group])
        Dd_mean = np.mean([f['D_over_d'] for f in group])
        Dd_std = np.std([f['D_over_d'] for f in group])
        print(f"  {c:<6} {len(group):<5} {d_val:<8} {D_mean:<10.2f} {Dd_mean:<10.4f} {Dd_std:<10.4f}")

    cs = np.array([f['circuit_size'] for f in d0])
    ds = np.array([f['H0_max_persistence'] for f in d0])
    dims = np.array([f['dim'] for f in d0])
    d_over_d = np.array([f['D_over_d'] for f in d0])

    print(f"\n  Spearman correlations:")
    r1, p1 = stats.spearmanr(cs, ds)
    print(f"    C(f) vs D(f):          r={r1:.4f}, p={p1:.2e}")

    r2, p2 = stats.spearmanr(cs, d_over_d)
    print(f"    C(f) vs D(f)/d(n,s):   r={r2:.4f}, p={p2:.2e}")

    r3, p3 = stats.spearmanr(dims, ds)
    print(f"    d(n,s) vs D(f):        r={r3:.4f}, p={p3:.2e}")

    r4, p4 = stats.spearmanr(cs, dims)
    print(f"    C(f) vs d(n,s):        r={r4:.4f}, p={p4:.2e}")

    slope_dc, intercept_dc = np.polyfit(dims, cs, 1)
    cs_resid = cs - (slope_dc * dims + intercept_dc)
    slope_dd, intercept_dd = np.polyfit(dims, ds, 1)
    ds_resid = ds - (slope_dd * dims + intercept_dd)

    r_partial, p_partial = stats.spearmanr(cs_resid, ds_resid)
    print(f"\n  Partial correlation (D vs C | d, linear residuals):")
    print(f"    Spearman r={r_partial:.4f}, p={p_partial:.2e}")

    print(f"\n  Summary:")
    print(f"    Raw correlation (C vs D): r={r1:.3f}")
    print(f"    Normalized (C vs D/d):    r={r2:.3f}")
    print(f"    The dimensional effect explains "
          f"~{(1 - r2**2/r1**2)*100:.0f}% of the raw correlation's r_s^2")
    print(f"    Residual signal after normalization: "
          f"{'significant' if p2 < 0.001 else 'not significant'}")


if __name__ == '__main__':
    main()
