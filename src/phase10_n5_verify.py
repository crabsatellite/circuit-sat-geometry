"""
Phase 10: n=5 partial verification of three-phase connectivity structure.

Uses external NPN5 classification + our SAT skeleton to identify
C(f) <= 3 classes at n=5, then runs solution enumeration + BFS connectivity.

Requires: pa-npn NPN classification data (https://github.com/alanctpnk/pa-npn).
Place the repository at data/external/pa-npn/ or set the NPN5_CSV environment
variable to the path of the npn-5-args.csv file.

Expected from Knuth TAOCP 4A:
  C=0: 2 classes, C=1: 2 classes, C=2: 5 classes, C=3: 20 classes
"""

import csv
import json
import time
import os
import sys
import numpy as np
from pysat.solvers import Glucose4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_circuit_sizes import CircuitSATSkeleton


def load_npn5_truth_tables(csv_path):
    """Load NPN5 canonical truth tables from pa-npn CSV."""
    classes = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tt = int(row['hex_func'], 16)
            classes.append({
                'canonical_tt': tt,
                'orbit_size': int(row['npn_class_len']),
                'tt_hex': row['hex_func'],
            })
    return classes


def is_projection_or_negation_n5(tt):
    """Check if tt is a projection or negation for n=5."""
    mask = 0xFFFFFFFF  # 2^32 - 1
    for i in range(5):
        var_tt = sum((1 << x) for x in range(32) if (x >> i) & 1)
        if tt == var_tt or tt == (mask ^ var_tt):
            return True
    return False


def find_low_complexity_classes(n, classes, max_size=3):
    """Find classes with C(f) <= max_size using SAT.

    Returns dict: tt -> circuit_size for resolved classes.
    """
    mask = (1 << (1 << n)) - 1
    resolved = {}

    # Special cases: constants -> C=1, projections -> C=0
    for cls in classes:
        tt = cls['canonical_tt']
        if tt == 0 or tt == mask:
            resolved[tt] = 1
        elif is_projection_or_negation_n5(tt):
            resolved[tt] = 0

    print(f"  Special cases: {len(resolved)} resolved", flush=True)

    for s in range(1, max_size + 1):
        unresolved = [c for c in classes if c['canonical_tt'] not in resolved]
        if not unresolved:
            break

        # Use conflict budget: SAT instances resolve fast,
        # UNSAT instances hit budget and get deferred
        budget = 100_000 if s <= 2 else 200_000

        print(f"  Building skeleton for n={n}, s={s} "
              f"({len(unresolved)} TTs to test, budget={budget//1000}K)...",
              flush=True)
        t0 = time.time()
        skeleton = CircuitSATSkeleton(n, s)
        build_time = time.time() - t0
        print(f"    Encoding: {len(skeleton.clauses)} clauses, "
              f"{skeleton.var_id} vars, built in {build_time:.2f}s", flush=True)

        newly_resolved = 0
        budget_exhausted = 0
        t0 = time.time()

        for i, cls in enumerate(unresolved):
            tt = cls['canonical_tt']
            sat, _ = skeleton.check_sat(tt, conf_budget=budget)
            if sat is True:
                resolved[tt] = s
                newly_resolved += 1
            elif sat is None:
                budget_exhausted += 1

            if (i + 1) % 50000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(unresolved) - i - 1) / rate
                print(f"    Progress: {i+1}/{len(unresolved)} "
                      f"({elapsed:.1f}s, {rate:.0f}/s, ETA {eta:.0f}s), "
                      f"found {newly_resolved} SAT", flush=True)

        solve_time = time.time() - t0
        print(f"    s={s}: {solve_time:.1f}s, "
              f"found {newly_resolved} with C={s}, "
              f"{budget_exhausted} budget-exhausted", flush=True)

    return resolved


def enumerate_solutions_n5(n, tt, s, max_solutions=1000, timeout=60):
    """Enumerate structurally distinct solutions for one function at size s."""
    skeleton = CircuitSATSkeleton(n, s)
    struct_ids = skeleton.get_structural_var_ids()

    solutions = []
    clauses = list(skeleton.clauses)
    assumptions = skeleton.make_output_assumptions(tt)

    t0 = time.time()
    while len(solutions) < max_solutions:
        if time.time() - t0 > timeout:
            break

        solver = Glucose4(bootstrap_with=clauses)
        result = solver.solve(assumptions=assumptions)

        if not result:
            solver.delete()
            break

        model = solver.get_model()
        solver.delete()

        # Extract structural bits
        struct_vec = []
        for vid in struct_ids:
            struct_vec.append(1 if model[vid - 1] > 0 else 0)

        solutions.append(struct_vec)

        # Add blocking clause
        blocking = [-model[vid - 1] for vid in struct_ids]
        clauses.append(blocking)

    return np.array(solutions) if solutions else np.array([]).reshape(0, 0)


def evaluate_circuit_n5(n, s, struct_vec, struct_ids, skeleton):
    """Evaluate a circuit defined by structural variables on all 2^n inputs."""
    num_assigns = 1 << n
    # Direct evaluation from structural vector
    nw_per_gate = [n + g for g in range(s)]

    # Decode structural vector
    idx = 0
    gate_left = []
    gate_right = []
    gate_func = []

    for g in range(s):
        nw = n + g
        # Left connection (one-hot)
        left_bits = struct_vec[idx:idx + nw]
        left_wire = int(np.argmax(left_bits)) if sum(left_bits) > 0 else 0
        idx += nw
        # Right connection (one-hot)
        right_bits = struct_vec[idx:idx + nw]
        right_wire = int(np.argmax(right_bits)) if sum(right_bits) > 0 else 0
        idx += nw
        # Function bits
        fb = struct_vec[idx:idx + 4]
        idx += 4

        gate_left.append(left_wire)
        gate_right.append(right_wire)
        gate_func.append(fb)

    # Evaluate on all inputs
    outputs = []
    for x in range(num_assigns):
        wire_vals = []
        for i in range(n):
            wire_vals.append((x >> i) & 1)

        for g in range(s):
            a = wire_vals[gate_left[g]]
            b = wire_vals[gate_right[g]]
            fb = gate_func[g]
            val = fb[2 * a + b]
            wire_vals.append(val)

        outputs.append(wire_vals[-1])

    return outputs


def bfs_connectivity_n5(n, s, target_tt, seed_solutions, max_per_component=10000,
                        max_total=50000):
    """BFS on local-move graph for n=5 circuits."""
    num_assigns = 1 << n
    struct_dim_per_gate = []
    for g in range(s):
        struct_dim_per_gate.append(2 * (n + g) + 4)
    total_dim = sum(struct_dim_per_gate)

    target_bits = [(target_tt >> x) & 1 for x in range(num_assigns)]

    def decode_circuit(vec):
        idx = 0
        gates = []
        for g in range(s):
            nw = n + g
            left = list(vec[idx:idx+nw])
            idx += nw
            right = list(vec[idx:idx+nw])
            idx += nw
            func = list(vec[idx:idx+4])
            idx += 4
            gates.append((left, right, func))
        return gates

    def eval_circuit(gates):
        outputs = []
        for x in range(num_assigns):
            wires = [(x >> i) & 1 for i in range(n)]
            for left, right, func in gates:
                lw = next((i for i, v in enumerate(left) if v), 0)
                rw = next((i for i, v in enumerate(right) if v), 0)
                a, b = wires[lw], wires[rw]
                wires.append(func[2*a + b])
            outputs.append(wires[-1])
        return outputs

    def is_valid(vec):
        gates = decode_circuit(vec)
        return eval_circuit(gates) == target_bits

    def get_neighbours(vec):
        neighbours = []
        idx = 0
        for g in range(s):
            nw = n + g
            # Reconnect left input
            for w in range(nw):
                if vec[idx + w] == 0:  # not currently selected
                    # Find current hot bit
                    cur = next((i for i in range(nw) if vec[idx+i]), 0)
                    new_vec = list(vec)
                    new_vec[idx + cur] = 0
                    new_vec[idx + w] = 1
                    if is_valid(new_vec):
                        neighbours.append(tuple(new_vec))
            idx += nw
            # Reconnect right input
            for w in range(nw):
                if vec[idx + w] == 0:
                    cur = next((i for i in range(nw) if vec[idx+i]), 0)
                    new_vec = list(vec)
                    new_vec[idx + cur] = 0
                    new_vec[idx + w] = 1
                    if is_valid(new_vec):
                        neighbours.append(tuple(new_vec))
            idx += nw
            # Flip function bits
            for b in range(4):
                new_vec = list(vec)
                new_vec[idx + b] = 1 - new_vec[idx + b]
                if is_valid(new_vec):
                    neighbours.append(tuple(new_vec))
            idx += 4
        return neighbours

    # BFS from all seed solutions
    all_visited = set()
    components = []
    total_explored = 0

    seeds = [tuple(s) for s in seed_solutions]
    # Deduplicate seeds
    seed_set = set(seeds)

    for seed in seeds:
        if seed in all_visited:
            continue
        if total_explored >= max_total:
            break

        # BFS from this seed
        queue = [seed]
        visited = {seed}
        head = 0

        while head < len(queue) and len(visited) < max_per_component:
            if total_explored >= max_total:
                break
            node = queue[head]
            head += 1
            total_explored += 1

            for nb in get_neighbours(node):
                if nb not in visited and nb not in all_visited:
                    visited.add(nb)
                    queue.append(nb)

        exhaustive = head >= len(queue)
        components.append({
            'size': len(visited),
            'exhaustive': exhaustive,
            'seeds_in_component': sum(1 for s in seed_set if s in visited),
        })
        all_visited.update(visited)

    return {
        'n_components': len(components),
        'largest_component': max(c['size'] for c in components) if components else 0,
        'total_explored': total_explored,
        'all_exhaustive': all(c['exhaustive'] for c in components),
        'components': components,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '..')
    data_dir = os.path.join(base_dir, 'data')

    n = 5
    csv_path = os.environ.get('NPN5_CSV',
                              os.path.join(data_dir, 'external', 'pa-npn',
                                           'generated', 'npn-5-args.csv'))

    print("=" * 60)
    print("Phase 10: n=5 Partial Verification")
    print("=" * 60)

    # Step 1: Load NPN5 truth tables
    print(f"\nStep 1: Loading NPN5 truth tables from pa-npn...", flush=True)
    t0 = time.time()
    classes = load_npn5_truth_tables(csv_path)
    print(f"  Loaded {len(classes)} classes in {time.time()-t0:.1f}s", flush=True)

    # Step 2: Find C(f) <= 3 classes via SAT
    print(f"\nStep 2: Finding C(f) <= 3 classes via SAT (n={n})...", flush=True)
    t0 = time.time()
    resolved = find_low_complexity_classes(n, classes, max_size=3)
    total_time = time.time() - t0

    # Summarize
    size_dist = {}
    for tt, c in resolved.items():
        size_dist[c] = size_dist.get(c, 0) + 1
    print(f"\n  Resolved distribution (C <= 3):", flush=True)
    for c in sorted(size_dist):
        print(f"    C={c}: {size_dist[c]} classes", flush=True)
    print(f"  Total time: {total_time:.1f}s", flush=True)

    # Cross-validate against Knuth
    expected = {0: 2, 1: 2, 2: 5, 3: 20}
    for c, exp in expected.items():
        got = size_dist.get(c, 0)
        status = "OK" if got == exp else f"MISMATCH (expected {exp})"
        print(f"    C={c}: {got} {status}", flush=True)

    # Step 3: Solution enumeration for C=2 and C=3 classes
    targets = []
    for cls in classes:
        tt = cls['canonical_tt']
        if tt in resolved and resolved[tt] in (2, 3):
            cls['circuit_size'] = resolved[tt]
            targets.append(cls)

    print(f"\nStep 3: Solution enumeration for {len(targets)} target classes "
          f"(C=2: {sum(1 for t in targets if t['circuit_size']==2)}, "
          f"C=3: {sum(1 for t in targets if t['circuit_size']==3)})...", flush=True)

    results = []
    for i, cls in enumerate(targets):
        tt = cls['canonical_tt']
        c = cls['circuit_size']
        print(f"  [{i+1}/{len(targets)}] tt=0x{cls['tt_hex']}, C={c}: ", end='',
              flush=True)

        # Enumerate at delta=0
        t0 = time.time()
        sols_d0 = enumerate_solutions_n5(n, tt, c, max_solutions=1000, timeout=60)
        t_d0 = time.time() - t0

        # Enumerate at delta=1
        t1 = time.time()
        sols_d1 = enumerate_solutions_n5(n, tt, c + 1, max_solutions=1000, timeout=60)
        t_d1 = time.time() - t1

        print(f"delta=0: {len(sols_d0)} sols ({t_d0:.1f}s), "
              f"delta=1: {len(sols_d1)} sols ({t_d1:.1f}s)", flush=True)

        result = {
            'tt_hex': cls['tt_hex'],
            'circuit_size': c,
            'n_solutions_d0': len(sols_d0),
            'n_solutions_d1': len(sols_d1),
        }

        # Step 4: BFS connectivity at delta=0
        if len(sols_d0) >= 2:
            print(f"    BFS delta=0: ", end='', flush=True)
            t0 = time.time()
            bfs_d0 = bfs_connectivity_n5(n, c, tt, sols_d0,
                                          max_per_component=10000,
                                          max_total=50000)
            t_bfs = time.time() - t0
            print(f"{bfs_d0['n_components']} components, "
                  f"largest={bfs_d0['largest_component']}, "
                  f"exhaustive={bfs_d0['all_exhaustive']} ({t_bfs:.1f}s)",
                  flush=True)
            result['bfs_d0'] = bfs_d0

        # BFS at delta=1
        if len(sols_d1) >= 2:
            print(f"    BFS delta=1: ", end='', flush=True)
            t0 = time.time()
            bfs_d1 = bfs_connectivity_n5(n, c + 1, tt, sols_d1,
                                          max_per_component=5000,
                                          max_total=20000)
            t_bfs = time.time() - t0
            print(f"{bfs_d1['n_components']} components, "
                  f"largest={bfs_d1['largest_component']}, "
                  f"exhaustive={bfs_d1['all_exhaustive']} ({t_bfs:.1f}s)",
                  flush=True)
            result['bfs_d1'] = bfs_d1

        results.append(result)

    # Save results
    out_path = os.path.join(data_dir, 'n5_verification.json')
    output = {
        'n': n,
        'n_total_classes': len(classes),
        'resolved_distribution': {str(k): v for k, v in size_dist.items()},
        'n_targets': len(targets),
        'results': results,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Three-Phase Verification at n=5")
    print(f"{'='*60}")

    # Phase 2 check: all C=2 should be totally isolated
    c2_results = [r for r in results if r['circuit_size'] == 2]
    c2_isolated = sum(1 for r in c2_results
                      if 'bfs_d0' in r and
                      r['bfs_d0']['largest_component'] == 1)
    print(f"C=2 (totally isolated?): {c2_isolated}/{len(c2_results)} "
          f"have all-isolated vertices at delta=0")

    # Phase 3 check: all C=3 should be disconnected at delta=0
    c3_results = [r for r in results if r['circuit_size'] == 3]
    c3_disconnected = sum(1 for r in c3_results
                         if 'bfs_d0' in r and
                         r['bfs_d0']['n_components'] > 1)
    print(f"C=3 (fragmented?): {c3_disconnected}/{len(c3_results)} "
          f"disconnected at delta=0")

    # Connectivity transition: large component at delta=1?
    c23_large_d1 = sum(1 for r in results
                       if 'bfs_d1' in r and
                       r['bfs_d1']['largest_component'] >= 100)
    print(f"C=2,3 (large component at delta=1?): {c23_large_d1}/{len(results)}")


if __name__ == '__main__':
    main()
