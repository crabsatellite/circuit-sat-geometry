"""
Phase 5: Generate publication-quality figures.

Figures:
  1. Circuit size distribution (NPN4 classes)
  2. Persistence diagrams: easy vs hard functions
  3. TDA features vs circuit complexity (scatter/violin plots)
  4. Phase transition: topology as s increases past C(f)
  5. Solution count vs circuit complexity
  6. Feature importance (from prediction)
"""

import json
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Publication style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def fig1_circuit_size_distribution(data_dir, fig_dir):
    """Bar chart of circuit complexity distribution over NPN4 classes."""
    with open(os.path.join(data_dir, 'npn4_circuit_sizes.json')) as f:
        p1 = json.load(f)

    sizes = [c['circuit_size'] for c in p1['classes']]
    unique_sizes = sorted(set(sizes))
    counts = [sizes.count(s) for s in unique_sizes]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(unique_sizes, counts, color='steelblue', edgecolor='white', linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Circuit complexity $C(f)$')
    ax.set_ylabel('Number of NPN4 classes')
    ax.set_title('Circuit Complexity Distribution (4-input Boolean Functions)')
    ax.set_xticks(unique_sizes)
    fig.savefig(os.path.join(fig_dir, 'fig1_circuit_size_dist.png'))
    plt.close(fig)
    print("  fig1_circuit_size_dist.png", flush=True)


def fig2_persistence_diagrams(data_dir, fig_dir):
    """Side-by-side persistence diagrams for easy vs hard functions at delta=0."""
    with open(os.path.join(data_dir, 'tda_features.json')) as f:
        tda = json.load(f)

    d0 = [f for f in tda['features'] if f['delta'] == 0]
    if not d0:
        return

    # Find easy (low C, enough solutions) and hard (high C, enough solutions)
    d0_with_sols = [f for f in d0 if f['n_solutions'] >= 20]
    if len(d0_with_sols) < 2:
        d0_with_sols = d0
    d0_with_sols.sort(key=lambda x: x['circuit_size'])
    easy = d0_with_sols[0]
    hard = d0_with_sols[-1]

    pd_dir = os.path.join(data_dir, 'persistence')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, inst, label in [(axes[0], easy, 'Easy'), (axes[1], hard, 'Hard')]:
        pd_path = os.path.join(pd_dir, f"{inst['tt_hex']}_s{inst['target_s']}_pd.json")
        if not os.path.exists(pd_path):
            continue
        with open(pd_path) as f:
            pd_data = json.load(f)

        colors = ['tab:blue', 'tab:orange', 'tab:green']
        labels = ['$H_0$', '$H_1$', '$H_2$']
        max_death = 0
        for dim in range(3):
            key = f'H{dim}'
            if key in pd_data and pd_data[key]:
                dgm = np.array(pd_data[key])
                finite = dgm[np.isfinite(dgm[:, 1])]
                if len(finite) > 0:
                    ax.scatter(finite[:, 0], finite[:, 1],
                               s=15, alpha=0.6, color=colors[dim], label=labels[dim])
                    max_death = max(max_death, finite[:, 1].max())

        if max_death > 0:
            lim = max_death * 1.1
            ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=0.5)
            ax.set_xlim(-0.5, lim)
            ax.set_ylim(-0.5, lim)
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.set_title(f'{label}: C(f)={inst["circuit_size"]}, s={inst["target_s"]}\n'
                      f'TT=0x{inst["tt_hex"]}, {inst["n_solutions"]} solutions')
        ax.legend()

    fig.suptitle('Persistence Diagrams: Easy vs Hard Functions (delta=0)', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig2_persistence_easy_vs_hard.png'))
    plt.close(fig)
    print("  fig2_persistence_easy_vs_hard.png", flush=True)


def fig3_tda_vs_complexity(data_dir, fig_dir):
    """Scatter/violin: key TDA features vs circuit complexity at delta=0."""
    with open(os.path.join(data_dir, 'tda_features.json')) as f:
        tda = json.load(f)

    d0 = [f for f in tda['features'] if f['delta'] == 0]
    if not d0:
        return

    cs = np.array([f['circuit_size'] for f in d0])
    unique_cs = sorted(set(cs))

    feature_keys = [
        ('H0_max_persistence', 'H0 Max Persistence'),
        ('H0_std_persistence', 'H0 Persistence Std Dev'),
        ('H0_mean_persistence', 'H0 Mean Persistence'),
        ('H1_mean_persistence', 'H1 Mean Persistence'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (key, title) in zip(axes.flat, feature_keys):
        vals = np.array([f[key] for f in d0], dtype=float)

        # Box plot by circuit size
        groups = [vals[cs == c] for c in unique_cs]
        bp = ax.boxplot(groups, positions=unique_cs, widths=0.6,
                        patch_artist=True, showfliers=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        # Overlay scatter
        for c in unique_cs:
            mask = cs == c
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, mask.sum())
            ax.scatter(c + jitter, vals[mask], s=8, alpha=0.4, color='steelblue')

        ax.set_xlabel('Circuit complexity $C(f)$')
        ax.set_ylabel(title)
        ax.set_xticks(unique_cs)

        # Add Spearman r
        r, p = stats.spearmanr(cs, vals)
        ax.set_title(f'{title}\n(Spearman r={r:.3f}, p={p:.2e})')

    fig.suptitle('TDA Features vs Circuit Complexity (delta=0)', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig3_tda_vs_complexity.png'))
    plt.close(fig)
    print("  fig3_tda_vs_complexity.png", flush=True)


def fig4_phase_transition(data_dir, fig_dir):
    """Line plots: how topology changes as delta increases."""
    with open(os.path.join(data_dir, 'tda_features.json')) as f:
        tda = json.load(f)

    features = tda['features']
    if not features:
        return

    # Group by circuit_size
    by_cs = {}
    for f in features:
        by_cs.setdefault(f['circuit_size'], []).append(f)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ('H1_total_persistence', 'H1 Total Persistence'),
        ('H1_n_bars', 'H1 Bar Count'),
        ('n_solutions', 'Solution Count'),
    ]

    for ax, (key, ylabel) in zip(axes, metrics):
        for cs in sorted(by_cs.keys()):
            by_delta = {}
            for f in by_cs[cs]:
                by_delta.setdefault(f['delta'], []).append(f[key])
            deltas = sorted(by_delta.keys())
            means = [np.mean(by_delta[d]) for d in deltas]
            stds = [np.std(by_delta[d]) for d in deltas]
            ax.errorbar(deltas, means, yerr=stds, marker='o', markersize=4,
                        label=f'C={cs}', capsize=3, linewidth=1.5)
        ax.set_xlabel(r'$\delta = s - C(f)$')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle('Phase Transition: Topology vs Circuit Size Slack', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig4_phase_transition.png'))
    plt.close(fig)
    print("  fig4_phase_transition.png", flush=True)


def fig5_solution_count_heatmap(data_dir, fig_dir):
    """Heatmap: solution count by circuit complexity x delta."""
    with open(os.path.join(data_dir, 'tda_features.json')) as f:
        tda = json.load(f)

    features = tda['features']
    if not features:
        return

    cs_vals = sorted(set(f['circuit_size'] for f in features))
    delta_vals = sorted(set(f['delta'] for f in features))

    matrix = np.full((len(cs_vals), len(delta_vals)), np.nan)
    counts = np.zeros((len(cs_vals), len(delta_vals)))
    sums = np.zeros((len(cs_vals), len(delta_vals)))
    for f in features:
        i = cs_vals.index(f['circuit_size'])
        j = delta_vals.index(f['delta'])
        sums[i, j] += f['n_solutions']
        counts[i, j] += 1
    mask = counts > 0
    matrix[mask] = sums[mask] / counts[mask]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(np.log10(matrix + 1), aspect='auto', cmap='YlOrRd',
                   origin='lower')
    ax.set_xticks(range(len(delta_vals)))
    ax.set_xticklabels(delta_vals)
    ax.set_yticks(range(len(cs_vals)))
    ax.set_yticklabels(cs_vals)
    ax.set_xlabel(r'$\delta = s - C(f)$')
    ax.set_ylabel('Circuit complexity $C(f)$')
    ax.set_title('Solution Count (log10)')
    fig.colorbar(im, ax=ax, label='$\\log_{10}$(solutions + 1)')
    fig.savefig(os.path.join(fig_dir, 'fig5_solution_heatmap.png'))
    plt.close(fig)
    print("  fig5_solution_heatmap.png", flush=True)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    fig_dir = os.path.join(script_dir, '..', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print("Generating figures...\n", flush=True)
    fig1_circuit_size_distribution(data_dir, fig_dir)
    fig2_persistence_diagrams(data_dir, fig_dir)
    fig3_tda_vs_complexity(data_dir, fig_dir)
    fig4_phase_transition(data_dir, fig_dir)
    fig5_solution_count_heatmap(data_dir, fig_dir)
    print("\nAll figures saved.", flush=True)


if __name__ == '__main__':
    main()
