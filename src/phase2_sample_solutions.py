"""
Phase 2: Sample circuit-SAT solution spaces.

For each NPN4 class with known circuit size C(f), and for each s in [C(f), C(f)+range]:
  - Build skeleton per circuit size s (reuse across all TTs at that size)
  - Enumerate solutions via iterated blocking clauses on structural variables
  - Extract structural variable assignments (connections + gate functions)
  - Save solution matrices for Phase 3 (TDA)

Output: data/solutions/<tt_hex>_s<size>.npy  (each is an N x D binary matrix)
        data/solutions_meta.json             (metadata for all instances)
"""

import json
import time
import os
import numpy as np
from pysat.solvers import Glucose4
from phase1_circuit_sizes import CircuitSATSkeleton


def enumerate_solutions(skeleton, target_tt, max_solutions, structural_var_ids,
                        time_limit=60.0):
    """Enumerate distinct circuit structures via blocking clauses.

    Args:
        time_limit: Max seconds per instance. Stops early if exceeded.
    """
    solutions = []
    solver = Glucose4(bootstrap_with=skeleton.clauses)
    assumptions = skeleton.make_output_assumptions(target_tt)
    t0 = time.time()

    while len(solutions) < max_solutions:
        if time.time() - t0 > time_limit:
            break

        if not solver.solve(assumptions=assumptions):
            break

        model = solver.get_model()
        # Extract structural variable assignment (binary vector)
        struct = [1 if model[vid - 1] > 0 else 0 for vid in structural_var_ids]
        solutions.append(struct)

        # Block this structural assignment (force at least one difference)
        blocking = [-model[vid - 1] for vid in structural_var_ids]
        solver.add_clause(blocking)

    solver.delete()
    if solutions:
        return np.array(solutions, dtype=np.uint8)
    return np.empty((0, len(structural_var_ids)), dtype=np.uint8)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    sol_dir = os.path.join(data_dir, 'solutions')
    os.makedirs(sol_dir, exist_ok=True)

    config_path = os.path.join(script_dir, '..', 'configs', 'default.json')
    with open(config_path) as f:
        config = json.load(f)

    n = config['n']
    max_solutions = config['max_solutions_per_instance']
    above_range = config['circuit_size_range_above_min']

    # Load Phase 1 results
    p1_path = os.path.join(data_dir, 'npn4_circuit_sizes.json')
    with open(p1_path) as f:
        p1 = json.load(f)
    classes = p1['classes']

    print(f"Loaded {len(classes)} NPN4 classes", flush=True)
    print(f"Config: max_solutions={max_solutions}, above_range={above_range}", flush=True)

    # Group by circuit size s needed
    instances_by_s = {}
    for cls in classes:
        c = cls['circuit_size']
        if c < 0:
            continue  # skip unresolved
        for s in range(max(c, 1), c + above_range + 1):
            instances_by_s.setdefault(s, []).append({
                'tt': cls['canonical_tt'],
                'tt_hex': cls['tt_hex'],
                'circuit_size': c,
                'target_s': s,
                'delta': s - c,  # 0 = minimum, 1+ = above minimum
            })

    total_instances = sum(len(v) for v in instances_by_s.values())
    print(f"Total instances to sample: {total_instances} "
          f"(across s={min(instances_by_s)}..{max(instances_by_s)})\n", flush=True)

    all_meta = []
    done = 0

    for s in sorted(instances_by_s.keys()):
        insts = instances_by_s[s]
        print(f"--- s={s}: {len(insts)} instances ---", flush=True)

        t0 = time.time()
        skeleton = CircuitSATSkeleton(n, s)
        struct_vars = skeleton.get_structural_var_ids()
        build_time = time.time() - t0
        print(f"  Skeleton: {len(skeleton.clauses)} clauses, "
              f"{skeleton.var_id} vars, {len(struct_vars)} structural vars, "
              f"built in {build_time:.2f}s", flush=True)

        # Time limit per instance: 60s for small s, 30s for large s
        inst_time_limit = 30.0 if s >= 6 else 60.0

        for inst in insts:
            fname = f"{inst['tt_hex']}_s{s}.npy"
            fpath = os.path.join(sol_dir, fname)

            # Skip if already computed (resume support)
            if os.path.exists(fpath):
                solutions = np.load(fpath)
                n_sol = solutions.shape[0]
                solve_time = 0.0
            else:
                t1 = time.time()
                solutions = enumerate_solutions(
                    skeleton, inst['tt'], max_solutions, struct_vars,
                    time_limit=inst_time_limit)
                solve_time = time.time() - t1
                n_sol = solutions.shape[0]
                np.save(fpath, solutions)

            meta = {
                'tt_hex': inst['tt_hex'],
                'tt': inst['tt'],
                'circuit_size': inst['circuit_size'],
                'target_s': s,
                'delta': inst['delta'],
                'n_solutions': n_sol,
                'exhausted': n_sol < max_solutions,
                'structural_dim': len(struct_vars),
                'solve_time_s': round(solve_time, 4),
                'file': fname,
            }
            all_meta.append(meta)
            done += 1

            tag = "EXHAUSTED" if meta['exhausted'] else "CAPPED"
            if solve_time > 5 or done % 50 == 0 or n_sol == 0:
                print(f"  [{done}/{total_instances}] {inst['tt_hex']} s={s} "
                      f"delta={inst['delta']}: {n_sol} solutions "
                      f"({solve_time:.2f}s) [{tag}]", flush=True)

        layer_time = time.time() - t0
        n_sols = [m['n_solutions'] for m in all_meta if m['target_s'] == s]
        print(f"  s={s} done: {len(insts)} instances, "
              f"median={int(np.median(n_sols))} solutions, "
              f"total {layer_time:.1f}s\n", flush=True)

    # Save metadata
    meta_path = os.path.join(data_dir, 'solutions_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'n': n,
            'max_solutions': max_solutions,
            'above_range': above_range,
            'total_instances': total_instances,
            'instances': all_meta,
        }, f, indent=2)
    print(f"\nSaved metadata to {meta_path}", flush=True)

    # Summary
    print("\n=== Summary ===", flush=True)
    for delta in range(above_range + 1):
        subset = [m for m in all_meta if m['delta'] == delta]
        if not subset:
            continue
        sols = [m['n_solutions'] for m in subset]
        exh = sum(1 for m in subset if m['exhausted'])
        print(f"  delta={delta} (s=C(f)+{delta}): {len(subset)} instances, "
              f"median={int(np.median(sols))} solutions, "
              f"{exh}/{len(subset)} exhausted", flush=True)


if __name__ == '__main__':
    main()
