"""
Phase 7: Solution Space Connectivity — Local Reachability Analysis

For representative functions at delta=0 and delta=1, explore the solution
space graph where:
  - Nodes = structural assignments that compute f with s gates
  - Edges = single local move (reconnect one gate input OR flip one function bit)

Measures:
  - BFS reachable set size from a seed solution
  - Connected components among known solutions
  - Graph diameter (max BFS distance)
  - Reachability fraction: what portion of known solutions are locally connected?
  - Delta=0 vs delta=1 comparison (fragmentation → collapse)

This tests the mechanism behind the diameter-complexity coupling:
complex functions at minimum size have fragmented solution spaces with
barrier regions; adding one gate of slack collapses these barriers.

Output: data/reachability/summary.json
"""

import json
import time
import os
import numpy as np
from collections import deque


# ====================================================================
# Circuit evaluation (no SAT — direct computation)
# ====================================================================

def evaluate_circuit(struct_vec, n, s, target_tt, layout):
    """Check if struct_vec encodes a circuit computing target_tt.

    Direct evaluation: O(s * 2^n) — trivially fast for n=4.
    """
    num_assigns = 1 << n
    wire_val = np.zeros((n + s, num_assigns), dtype=np.uint8)

    for i in range(n):
        for x in range(num_assigns):
            wire_val[i, x] = (x >> i) & 1

    for g in range(s):
        l_s, l_e = layout[g]['left']
        r_s, r_e = layout[g]['right']
        f_s, _ = layout[g]['func']

        left_wire = np.argmax(struct_vec[l_s:l_e])
        right_wire = np.argmax(struct_vec[r_s:r_e])

        a_vals = wire_val[left_wire]
        b_vals = wire_val[right_wire]
        idx = 2 * a_vals + b_vals
        wire_val[n + g] = np.array([struct_vec[f_s + idx[x]] for x in range(num_assigns)],
                                    dtype=np.uint8)

    computed = 0
    for x in range(num_assigns):
        computed |= int(wire_val[n + s - 1, x]) << x
    return computed == target_tt


def parse_layout(n, s):
    """Per-gate offset ranges for structural variables."""
    layout = []
    offset = 0
    for g in range(s):
        nw = n + g
        left = (offset, offset + nw);   offset += nw
        right = (offset, offset + nw);  offset += nw
        func = (offset, offset + 4);    offset += 4
        layout.append({'left': left, 'right': right, 'func': func})
    return layout


# ====================================================================
# Neighbor generation (single local move)
# ====================================================================

def get_neighbors(struct_vec, n, s, layout):
    """Generate all valid single-edit neighbors.

    Local moves:
      - Reconnect one gate's left/right input to a different wire
      - Flip one function bit of one gate
    """
    neighbors = []
    for g in range(s):
        nw = n + g
        l_s, l_e = layout[g]['left']
        r_s, r_e = layout[g]['right']
        f_s, f_e = layout[g]['func']

        # Left reconnection: try every other wire
        cur_left = np.argmax(struct_vec[l_s:l_e])
        for w in range(nw):
            if w != cur_left:
                nb = struct_vec.copy()
                nb[l_s:l_e] = 0
                nb[l_s + w] = 1
                neighbors.append(nb)

        # Right reconnection
        cur_right = np.argmax(struct_vec[r_s:r_e])
        for w in range(nw):
            if w != cur_right:
                nb = struct_vec.copy()
                nb[r_s:r_e] = 0
                nb[r_s + w] = 1
                neighbors.append(nb)

        # Function bit flip
        for b in range(4):
            nb = struct_vec.copy()
            nb[f_s + b] = 1 - nb[f_s + b]
            neighbors.append(nb)

    return neighbors


# ====================================================================
# BFS on local-move graph
# ====================================================================

def bfs_component(seed, n, s, target_tt, layout, max_nodes=10000):
    """BFS from seed. Returns (visited_dict: key→distance, exhaustive: bool)."""
    seed_key = seed.tobytes()
    visited = {seed_key: 0}
    queue = deque([(seed, 0)])

    while queue and len(visited) < max_nodes:
        node, dist = queue.popleft()
        for nb in get_neighbors(node, n, s, layout):
            key = nb.tobytes()
            if key not in visited:
                if evaluate_circuit(nb, n, s, target_tt, layout):
                    visited[key] = dist + 1
                    queue.append((nb, dist + 1))

    exhaustive = len(queue) == 0
    return visited, exhaustive


def find_components(solutions, n, s, target_tt, layout, max_nodes_per_comp=10000,
                    max_total=50000):
    """Find connected components among solutions via BFS.

    BFS explores the FULL solution space (not restricted to known solutions),
    so it may discover additional valid circuits beyond the input set.
    """
    sol_keys = {sol.tobytes() for sol in solutions}
    assigned = set()
    components = []
    total_explored = 0

    for sol in solutions:
        key = sol.tobytes()
        if key in assigned:
            continue

        remaining = max_nodes_per_comp
        if max_total > 0:
            remaining = min(remaining, max_total - total_explored)
        if remaining <= 0:
            # Budget exhausted — remaining solutions are unassigned
            break

        visited, exhaustive = bfs_component(
            sol, n, s, target_tt, layout, max_nodes=remaining)

        # Which known solutions are in this component?
        found_known = sol_keys & set(visited.keys())
        assigned |= found_known

        distances = list(visited.values())
        components.append({
            'total_reachable': len(visited),
            'known_in_component': len(found_known),
            'max_distance': max(distances) if distances else 0,
            'mean_distance': float(np.mean(distances)) if distances else 0,
            'exhaustive': exhaustive,
        })
        total_explored += len(visited)

    return components, len(assigned), total_explored


# ====================================================================
# Main experiment
# ====================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    sol_dir = os.path.join(data_dir, 'solutions')
    reach_dir = os.path.join(data_dir, 'reachability')
    os.makedirs(reach_dir, exist_ok=True)

    n = 4

    # Load Phase 1 results
    with open(os.path.join(data_dir, 'npn4_circuit_sizes.json')) as f:
        p1 = json.load(f)
    classes = p1['classes']
    by_c = {}
    for cls in classes:
        by_c.setdefault(cls['circuit_size'], []).append(cls)

    # Select representative functions
    rng = np.random.default_rng(42)
    selected = []

    for c in [2, 5, 7]:
        group = by_c.get(c, [])
        if c == 5 and len(group) > 8:
            idx = rng.choice(len(group), 8, replace=False)
            group = [group[i] for i in idx]
        selected.extend([(cls, c) for cls in group])

    print(f"Selected {len(selected)} functions: "
          f"C=2:{sum(1 for _,c in selected if c==2)}, "
          f"C=5:{sum(1 for _,c in selected if c==5)}, "
          f"C=7:{sum(1 for _,c in selected if c==7)}\n", flush=True)

    all_results = []
    t_start = time.time()

    for i, (cls, c) in enumerate(selected):
        tt = cls['canonical_tt']
        tt_hex = cls['tt_hex']

        for delta in [0, 1]:
            s = max(c, 1) + delta
            layout = parse_layout(n, s)

            # Load solutions
            sol_path = os.path.join(sol_dir, f"{tt_hex}_s{s}.npy")
            if not os.path.exists(sol_path):
                continue
            solutions = np.load(sol_path)

            if len(solutions) < 2:
                continue

            t0 = time.time()

            # BFS cap: generous for delta=0 (small spaces), tighter for delta=1
            max_per = 10000 if delta == 0 else 5000
            max_tot = 50000 if delta == 0 else 20000

            components, assigned, total = find_components(
                solutions, n, s, tt, layout,
                max_nodes_per_comp=max_per, max_total=max_tot)

            elapsed = time.time() - t0

            n_sol = len(solutions)
            n_comp = len(components)
            largest = max(c_['total_reachable'] for c_ in components) if components else 0
            max_dist = max(c_['max_distance'] for c_ in components) if components else 0
            all_exhaustive = all(c_['exhaustive'] for c_ in components)
            frac_assigned = assigned / n_sol if n_sol > 0 else 0

            result = {
                'tt_hex': tt_hex,
                'circuit_size': c,
                'delta': delta,
                's': s,
                'n_phase2_solutions': n_sol,
                'n_components': n_comp,
                'largest_component': largest,
                'max_graph_distance': max_dist,
                'phase2_assigned_fraction': round(frac_assigned, 4),
                'total_bfs_explored': total,
                'all_exhaustive': all_exhaustive,
                'component_details': components,
                'time_s': round(elapsed, 2),
            }
            all_results.append(result)

            tag = "FULL" if all_exhaustive else "CAPPED"
            print(f"  [{i+1}/{len(selected)}] {tt_hex} C={c} delta={delta} s={s}: "
                  f"{n_comp} comp, largest={largest}, max_dist={max_dist}, "
                  f"assigned={assigned}/{n_sol} ({frac_assigned:.0%}), "
                  f"total={total} [{tag}] ({elapsed:.1f}s)", flush=True)

        if (i + 1) % 5 == 0:
            print(flush=True)

    total_time = time.time() - t_start

    # ================================================================
    # Summary analysis
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("CONNECTIVITY SUMMARY", flush=True)
    print("=" * 70, flush=True)

    for c in [2, 5, 7]:
        print(f"\n  C={c}:", flush=True)
        for delta in [0, 1]:
            subset = [r for r in all_results
                      if r['circuit_size'] == c and r['delta'] == delta]
            if not subset:
                continue
            n_comp = [r['n_components'] for r in subset]
            largest = [r['largest_component'] for r in subset]
            max_dist = [r['max_graph_distance'] for r in subset]
            frac = [r['phase2_assigned_fraction'] for r in subset]
            exh = sum(1 for r in subset if r['all_exhaustive'])

            print(f"    delta={delta} ({len(subset)} funcs): "
                  f"components={np.mean(n_comp):.1f}±{np.std(n_comp):.1f}, "
                  f"largest={np.mean(largest):.0f}±{np.std(largest):.0f}, "
                  f"max_dist={np.mean(max_dist):.1f}±{np.std(max_dist):.1f}, "
                  f"assigned={np.mean(frac):.0%}, "
                  f"exhaustive={exh}/{len(subset)}", flush=True)

    # Fragmentation ratio: delta=0 components / delta=1 components
    print(f"\n  Fragmentation ratio (delta=0 components / delta=1 components):",
          flush=True)
    for c in [2, 5, 7]:
        d0 = {r['tt_hex']: r for r in all_results
              if r['circuit_size'] == c and r['delta'] == 0}
        d1 = {r['tt_hex']: r for r in all_results
              if r['circuit_size'] == c and r['delta'] == 1}
        ratios = []
        for hex_key in d0:
            if hex_key in d1 and d1[hex_key]['n_components'] > 0:
                ratios.append(d0[hex_key]['n_components'] /
                              d1[hex_key]['n_components'])
        if ratios:
            print(f"    C={c}: mean={np.mean(ratios):.2f}x, "
                  f"median={np.median(ratios):.2f}x, "
                  f"range=[{min(ratios):.1f}, {max(ratios):.1f}]", flush=True)

    # Save
    summary = {
        'n_functions': len(selected),
        'results': all_results,
        'total_time_s': round(total_time, 2),
    }
    out_path = os.path.join(reach_dir, 'summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal time: {total_time:.0f}s", flush=True)
    print(f"Saved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
