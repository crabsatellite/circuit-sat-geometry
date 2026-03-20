"""
Phase 7b: Complete BFS connectivity analysis for ALL 222 NPN classes.

Extends Phase 7 which only covered C in {2, 5, 7} (25 functions).
Tests the connectivity threshold conjecture on the full universe.

Output: data/reachability/summary_complete.json
"""

import json
import time
import os
import numpy as np
from collections import deque


# ====================================================================
# Circuit evaluation (copied from phase7 to be self-contained)
# ====================================================================

def evaluate_circuit(struct_vec, n, s, target_tt, layout):
    """Check if struct_vec encodes a circuit computing target_tt."""
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


def get_neighbors(struct_vec, n, s, layout):
    """Generate all valid single-edit neighbors."""
    neighbors = []
    for g in range(s):
        nw = n + g
        l_s, l_e = layout[g]['left']
        r_s, r_e = layout[g]['right']
        f_s, f_e = layout[g]['func']

        cur_left = np.argmax(struct_vec[l_s:l_e])
        for w in range(nw):
            if w != cur_left:
                nb = struct_vec.copy()
                nb[l_s:l_e] = 0
                nb[l_s + w] = 1
                neighbors.append(nb)

        cur_right = np.argmax(struct_vec[r_s:r_e])
        for w in range(nw):
            if w != cur_right:
                nb = struct_vec.copy()
                nb[r_s:r_e] = 0
                nb[r_s + w] = 1
                neighbors.append(nb)

        for b in range(4):
            nb = struct_vec.copy()
            nb[f_s + b] = 1 - nb[f_s + b]
            neighbors.append(nb)

    return neighbors


def bfs_component(seed, n, s, target_tt, layout, max_nodes=10000):
    """BFS from seed. Returns (visited_dict: key->distance, exhaustive: bool)."""
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
    """Find connected components among solutions via BFS."""
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
            break

        visited, exhaustive = bfs_component(
            sol, n, s, target_tt, layout, max_nodes=remaining)

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

def save_checkpoint(out_path, all_results, skipped, new_count,
                    missing_files, single_solution, elapsed):
    """Save current progress to disk (atomic write via temp file)."""
    summary = {
        'n_total_classes': 222,
        'n_results': len(all_results),
        'n_existing_reused': skipped,
        'n_new': new_count,
        'missing_files': missing_files,
        'single_solution': single_solution,
        'total_time_new_s': round(elapsed, 2),
        'complete': False,
        'results': all_results,
    }
    tmp_path = out_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp_path, out_path)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    sol_dir = os.path.join(data_dir, 'solutions')
    reach_dir = os.path.join(data_dir, 'reachability')
    os.makedirs(reach_dir, exist_ok=True)

    n = 4
    out_path = os.path.join(reach_dir, 'summary_complete.json')

    # Load Phase 1 results
    with open(os.path.join(data_dir, 'npn4_circuit_sizes.json')) as f:
        p1 = json.load(f)
    classes = p1['classes']

    # Load existing results from BOTH files to avoid re-running
    all_results = []
    existing_keys = set()

    # 1) Original phase7 results
    existing_path = os.path.join(reach_dir, 'summary.json')
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
        all_results.extend(existing['results'])
        for r in existing['results']:
            existing_keys.add((r['tt_hex'], r['delta']))

    # 2) Previous checkpoint (resume support)
    if os.path.exists(out_path):
        with open(out_path) as f:
            checkpoint = json.load(f)
        for r in checkpoint['results']:
            key = (r['tt_hex'], r['delta'])
            if key not in existing_keys:
                all_results.append(r)
                existing_keys.add(key)

    print(f"Loaded {len(existing_keys)} existing BFS results (resume checkpoint)")
    print(f"Running BFS for all 222 NPN classes at delta=0 and delta=1")
    print(f"Skipping {len(existing_keys)} already-computed entries\n", flush=True)

    skipped = 0
    new_count = 0
    missing_files = 0
    single_solution = 0
    t_start = time.time()

    for i, cls in enumerate(classes):
        tt = cls['canonical_tt']
        tt_hex = cls['tt_hex']
        c = cls['circuit_size']

        for delta in [0, 1]:
            if (tt_hex, delta) in existing_keys:
                skipped += 1
                continue

            s = max(c, 1) + delta
            layout = parse_layout(n, s)

            # Load solutions
            sol_path = os.path.join(sol_dir, f"{tt_hex}_s{s}.npy")
            if not os.path.exists(sol_path):
                missing_files += 1
                continue
            solutions = np.load(sol_path)

            n_sol = len(solutions)

            if n_sol < 2:
                single_solution += 1
                result = {
                    'tt_hex': tt_hex,
                    'circuit_size': c,
                    'delta': delta,
                    's': s,
                    'n_phase2_solutions': n_sol,
                    'n_components': n_sol,
                    'largest_component': 1 if n_sol == 1 else 0,
                    'max_graph_distance': 0,
                    'phase2_assigned_fraction': 1.0,
                    'total_bfs_explored': n_sol,
                    'all_exhaustive': True,
                    'component_details': [],
                    'time_s': 0.0,
                }
                all_results.append(result)
                new_count += 1
                print(f"  [{i+1}/222] {tt_hex} C={c} delta={delta} s={s}: "
                      f"{n_sol} solution(s) — trivial", flush=True)
                continue

            t0 = time.time()

            max_per = 10000 if delta == 0 else 5000
            max_tot = 50000 if delta == 0 else 20000

            components, assigned, total = find_components(
                solutions, n, s, tt, layout,
                max_nodes_per_comp=max_per, max_total=max_tot)

            elapsed = time.time() - t0

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
            new_count += 1

            # Checkpoint after every BFS run
            save_checkpoint(out_path, all_results, skipped, new_count,
                            missing_files, single_solution,
                            time.time() - t_start)

            tag = "FULL" if all_exhaustive else "CAPPED"
            print(f"  [{i+1}/222] {tt_hex} C={c} delta={delta} s={s}: "
                  f"{n_comp} comp, largest={largest}, max_dist={max_dist}, "
                  f"assigned={assigned}/{n_sol} ({frac_assigned:.0%}), "
                  f"total={total} [{tag}] ({elapsed:.1f}s)", flush=True)

    total_time = time.time() - t_start

    print(f"\n{'=' * 70}")
    print(f"COMPLETE BFS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Existing results reused: {skipped}")
    print(f"New BFS runs: {new_count}")
    print(f"Missing solution files: {missing_files}")
    print(f"Single/zero solution (trivial): {single_solution}")
    print(f"Total results: {len(all_results)}")
    print(f"New computation time: {total_time:.0f}s")

    # ================================================================
    # Connectivity analysis by complexity level
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"CONNECTIVITY THRESHOLD ANALYSIS")
    print(f"{'=' * 70}\n")

    # Group by circuit_size
    c_values = sorted(set(r['circuit_size'] for r in all_results))

    conjecture_support = 0
    conjecture_violations = 0
    conjecture_tested = 0

    for c in c_values:
        d0 = [r for r in all_results if r['circuit_size'] == c and r['delta'] == 0]
        d1 = [r for r in all_results if r['circuit_size'] == c and r['delta'] == 1]

        if not d0:
            continue

        # Delta=0 analysis: disconnection
        disconnected_d0 = sum(1 for r in d0 if r['n_components'] > 1)
        connected_d0 = sum(1 for r in d0 if r['n_components'] == 1)
        single_sol_d0 = sum(1 for r in d0 if r['n_phase2_solutions'] <= 1)
        multi_sol_d0 = [r for r in d0 if r['n_phase2_solutions'] >= 2]

        # Delta=1 analysis: large component emergence
        large_comp_d1 = sum(1 for r in d1
                           if r['largest_component'] >= 100 or
                           (r['n_phase2_solutions'] >= 2 and
                            r['largest_component'] >= r['n_phase2_solutions'] * 0.5))

        print(f"  C={c} ({len(d0)} classes):")
        print(f"    delta=0: {disconnected_d0} disconnected, "
              f"{connected_d0} connected, {single_sol_d0} single-solution")

        if multi_sol_d0:
            comp_counts = [r['n_components'] for r in multi_sol_d0]
            largest_sizes = [r['largest_component'] for r in multi_sol_d0]
            print(f"    delta=0 (multi-sol): components mean={np.mean(comp_counts):.1f}, "
                  f"largest mean={np.mean(largest_sizes):.0f}")

        if d1:
            d1_multi = [r for r in d1 if r['n_phase2_solutions'] >= 2]
            if d1_multi:
                comp_d1 = [r['n_components'] for r in d1_multi]
                largest_d1 = [r['largest_component'] for r in d1_multi]
                print(f"    delta=1 (multi-sol): components mean={np.mean(comp_d1):.1f}, "
                      f"largest mean={np.mean(largest_d1):.0f}, "
                      f"large-component={large_comp_d1}/{len(d1_multi)}")

        # Conjecture check (for C >= 2 with multi-solution)
        if c >= 2:
            for r in multi_sol_d0:
                conjecture_tested += 1
                if r['n_components'] > 1:
                    conjecture_support += 1
                else:
                    conjecture_violations += 1
                    print(f"    *** VIOLATION: {r['tt_hex']} has {r['n_components']} "
                          f"component(s) at delta=0 with {r['n_phase2_solutions']} solutions ***")

        print()

    print(f"{'=' * 70}")
    print(f"CONJECTURE VERDICT (C >= 2, multi-solution at delta=0)")
    print(f"{'=' * 70}")
    print(f"  Tested: {conjecture_tested}")
    print(f"  Support (disconnected at delta=0): {conjecture_support}")
    print(f"  Violations (connected at delta=0): {conjecture_violations}")
    if conjecture_tested > 0:
        print(f"  Rate: {conjecture_support}/{conjecture_tested} "
              f"({100*conjecture_support/conjecture_tested:.1f}%)")

    # ================================================================
    # Final save (mark complete)
    # ================================================================
    summary = {
        'n_total_classes': 222,
        'n_results': len(all_results),
        'n_existing_reused': skipped,
        'n_new': new_count,
        'conjecture_tested': conjecture_tested,
        'conjecture_support': conjecture_support,
        'conjecture_violations': conjecture_violations,
        'total_time_new_s': round(total_time, 2),
        'complete': True,
        'results': all_results,
    }
    tmp_path = out_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp_path, out_path)

    print(f"\nSaved to {out_path} (complete={True})")


if __name__ == '__main__':
    main()
