"""Generate H0 max-persistence vs C(f) scatter plot with exponential fit."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')

    with open(os.path.join(data_dir, 'tda_features.json')) as f:
        tda = json.load(f)

    features = tda['features']
    d0 = [f for f in features if f['delta'] == 0]

    print("Solution count distribution at delta=0:")
    for f in sorted(d0, key=lambda x: x['n_solutions']):
        if f['n_solutions'] <= 5:
            print(f"  {f['tt_hex']} C={f['circuit_size']} n_sol={f['n_solutions']}")
    print(f"\nTotal instances at delta=0: {len(d0)}")
    print(f"With n_solutions >= 2: {sum(1 for f in d0 if f['n_solutions'] >= 2)}")
    print(f"With n_solutions >= 3: {sum(1 for f in d0 if f['n_solutions'] >= 3)}")

    d0_analysis = [f for f in d0 if f['n_solutions'] >= 3]

    cs = np.array([f['circuit_size'] for f in d0_analysis])
    ds = np.array([f['H0_max_persistence'] for f in d0_analysis])

    by_c = {}
    for f in d0_analysis:
        by_c.setdefault(f['circuit_size'], []).append(f['H0_max_persistence'])

    fig, ax = plt.subplots(figsize=(7, 5))

    np.random.seed(42)
    jitter = np.random.uniform(-0.15, 0.15, len(cs))
    ax.scatter(cs + jitter, ds, alpha=0.4, s=20, c='#1f77b4', edgecolors='none')

    positions = sorted(by_c.keys())
    bp_data = [by_c[c] for c in positions]
    ax.boxplot(bp_data, positions=positions, widths=0.3,
               patch_artist=True, zorder=3,
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=1.5),
               whiskerprops=dict(color='gray'),
               capprops=dict(color='gray'),
               flierprops=dict(marker='.', markersize=3, alpha=0.3))

    try:
        log_ds = np.log(ds)
        slope, intercept = np.polyfit(cs, log_ds, 1)
        a_fit = np.exp(intercept)
        b_fit = slope
        x_fit = np.linspace(min(cs) - 0.5, max(cs) + 0.5, 100)
        ax.plot(x_fit, a_fit * np.exp(b_fit * x_fit), 'r--', linewidth=2, alpha=0.8,
                label=f'$D \\approx {a_fit:.2f} \\cdot e^{{{b_fit:.3f}\\,C}}$')
        ax.legend(fontsize=11, loc='upper left')
        print(f"Log-space OLS fit: a={a_fit:.4f}, b={b_fit:.4f}")
    except Exception as e:
        print(f"Fit failed: {e}")

    ax.set_xlabel('Circuit complexity $C(f)$', fontsize=12)
    ax.set_ylabel('Minimax linkage distance (MST bottleneck)', fontsize=12)
    ax.set_xticks(positions)

    plt.tight_layout()

    fig_dir = os.path.join(script_dir, '..', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, 'fig_tda_vs_complexity.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved scatter plot to {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
