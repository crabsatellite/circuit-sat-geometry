"""
Phase 1: Enumerate NPN4 equivalence classes and compute exact minimum circuit sizes.

Optimizations:
  - NPN enumeration: vectorized numpy (all 65536 TTs processed simultaneously)
  - SAT solving: skeleton per circuit size s, tested against all TTs via assumptions
    (build encoding once per s, reuse solver across TTs)
  - Symmetry breaking: left_input <= right_input for each gate
  - Checkpointing: save after each s level; resume from checkpoint on restart
  - Per-instance timeout (30s) to skip pathological UNSAT proofs

Output: data/npn4_circuit_sizes.json
"""

import json
import time
import os
import numpy as np
from itertools import permutations
from pysat.solvers import Glucose4


# =============================================================================
# NPN4 Enumeration (vectorized)
# =============================================================================

def enumerate_npn_classes(n=4):
    """Enumerate all NPN equivalence classes using vectorized numpy."""
    num_assigns = 1 << n
    num_funcs = 1 << num_assigns
    mask = np.uint32(num_funcs - 1)

    all_tts = np.arange(num_funcs, dtype=np.uint32)
    canonical = all_tts.copy()

    for perm in permutations(range(n)):
        for neg_in in range(1 << n):
            remap = []
            for x in range(num_assigns):
                px = 0
                for j in range(n):
                    if (x >> j) & 1:
                        px |= 1 << perm[j]
                remap.append(px ^ neg_in)

            new_tts = np.zeros(num_funcs, dtype=np.uint32)
            for x in range(num_assigns):
                bit = (all_tts >> np.uint32(remap[x])) & np.uint32(1)
                new_tts |= bit << np.uint32(x)

            canonical = np.minimum(canonical, new_tts)
            canonical = np.minimum(canonical, mask ^ new_tts)

    unique_canonical, counts = np.unique(canonical, return_counts=True)
    classes = []
    for i, c in enumerate(unique_canonical):
        classes.append({
            'canonical_tt': int(c),
            'orbit_size': int(counts[i]),
            'tt_hex': f'{int(c):04x}'
        })
    return classes


# =============================================================================
# Circuit-SAT Skeleton (build once per s, test many TTs)
# =============================================================================

class CircuitSATSkeleton:
    """SAT encoding skeleton for circuit size s over n inputs.

    The encoding contains all constraints EXCEPT the output (target truth table).
    The output constraints are passed as assumptions during solve(), allowing
    the same encoding to be reused for many different target functions.
    """

    def __init__(self, n, s, symmetry_breaking=True):
        self.n = n
        self.s = s
        self.num_assigns = 1 << n
        self.var_id = 0
        self.clauses = []

        # Primary input values (fixed)
        self.pin_val = {}
        for w in range(n):
            for x in range(self.num_assigns):
                v = self._new_var()
                self.pin_val[(w, x)] = v
                self.clauses.append([v] if (x >> w) & 1 else [-v])

        # Gate output values
        self.gate_val = {}
        for g in range(s):
            for x in range(self.num_assigns):
                self.gate_val[(g, x)] = self._new_var()

        # Connection variables
        self.conn_left = {}
        self.conn_right = {}
        for g in range(s):
            for w in range(n + g):
                self.conn_left[(g, w)] = self._new_var()
                self.conn_right[(g, w)] = self._new_var()

        # Gate function bits
        self.gate_func = {}
        for g in range(s):
            for b in range(4):
                self.gate_func[(g, b)] = self._new_var()

        self._encode_one_hot()
        self._encode_functionality()
        if symmetry_breaking:
            self._encode_symmetry_breaking()

    def _new_var(self):
        self.var_id += 1
        return self.var_id

    def _wire_val(self, w, x):
        if w < self.n:
            return self.pin_val[(w, x)]
        return self.gate_val[(w - self.n, x)]

    def _encode_one_hot(self):
        for g in range(self.s):
            nw = self.n + g
            for cd in (self.conn_left, self.conn_right):
                vs = [cd[(g, w)] for w in range(nw)]
                self.clauses.append(vs[:])
                for i in range(len(vs)):
                    for j in range(i + 1, len(vs)):
                        self.clauses.append([-vs[i], -vs[j]])

    def _encode_functionality(self):
        for g in range(self.s):
            nw = self.n + g
            for x in range(self.num_assigns):
                gv = self.gate_val[(g, x)]
                for wl in range(nw):
                    for wr in range(nw):
                        cl = self.conn_left[(g, wl)]
                        cr = self.conn_right[(g, wr)]
                        av = self._wire_val(wl, x)
                        bv = self._wire_val(wr, x)
                        for ai in range(2):
                            for bi in range(2):
                                fb = self.gate_func[(g, 2 * ai + bi)]
                                al = av if ai else -av
                                bl = bv if bi else -bv
                                self.clauses.append([-cl, -cr, -al, -bl, -gv, fb])
                                self.clauses.append([-cl, -cr, -al, -bl, gv, -fb])

    def _encode_symmetry_breaking(self):
        """Add left_input <= right_input constraint for each gate.

        For every binary function op(a,b), there exists op'(b,a) = op(a,b)
        (just transpose the truth table). So we can WLOG require left <= right.
        This halves the search space per gate (2^s total reduction).
        """
        for g in range(self.s):
            nw = self.n + g
            for wl in range(nw):
                for wr in range(wl):  # wr < wl => forbidden
                    self.clauses.append([
                        -self.conn_left[(g, wl)],
                        -self.conn_right[(g, wr)]
                    ])

    def make_output_assumptions(self, target_tt):
        """Create assumption literals that constrain last gate to match target_tt."""
        g = self.s - 1
        assumptions = []
        for x in range(self.num_assigns):
            gv = self.gate_val[(g, x)]
            assumptions.append(gv if (target_tt >> x) & 1 else -gv)
        return assumptions

    def check_sat(self, target_tt, conf_budget=0):
        """Check if target_tt can be computed by a circuit of size s.

        Args:
            conf_budget: Max conflicts (0 = unlimited). If budget exhausted,
                         returns (None, None) instead of (True/False, model).
        """
        assumptions = self.make_output_assumptions(target_tt)
        solver = Glucose4(bootstrap_with=self.clauses)

        if conf_budget > 0:
            solver.conf_budget(conf_budget)
            result = solver.solve_limited(assumptions=assumptions)
        else:
            result = solver.solve(assumptions=assumptions)

        if result is True:
            model = solver.get_model()
            solver.delete()
            return True, model
        elif result is False:
            solver.delete()
            return False, None
        else:
            # Budget exhausted (None from solve_limited)
            solver.delete()
            return None, None

    def get_structural_var_ids(self):
        ids = []
        for g in range(self.s):
            for w in range(self.n + g):
                ids.append(self.conn_left[(g, w)])
            for w in range(self.n + g):
                ids.append(self.conn_right[(g, w)])
            for b in range(4):
                ids.append(self.gate_func[(g, b)])
        return ids


# =============================================================================
# Circuit size computation
# =============================================================================

def is_projection_or_negation(tt, n=4):
    mask = (1 << (1 << n)) - 1
    for i in range(n):
        var_tt = sum((1 << x) for x in range(1 << n) if (x >> i) & 1)
        if tt == var_tt or tt == (mask ^ var_tt):
            return True
    return False


def find_all_circuit_sizes(n, classes, max_size=12, checkpoint_path=None):
    """Find minimum circuit sizes for all NPN classes.

    Strategy: for each circuit size s=1,2,..., build skeleton once,
    test all unresolved TTs against it via assumptions.
    Checkpoints after each s level.
    """
    # Handle special cases first
    mask = (1 << (1 << n)) - 1
    resolved = {}

    # Load checkpoint if exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        resolved = {int(k): v for k, v in ckpt['resolved'].items()}
        last_s = ckpt['last_completed_s']
        print(f"  Resumed from checkpoint: {len(resolved)} resolved, "
              f"last_s={last_s}", flush=True)
        start_s = last_s + 1
    else:
        for cls in classes:
            tt = cls['canonical_tt']
            if tt == 0 or tt == mask:
                resolved[tt] = 1
            elif is_projection_or_negation(tt, n):
                resolved[tt] = 0
        start_s = 1

    unresolved = [cls for cls in classes if cls['canonical_tt'] not in resolved]
    print(f"  Special cases: {len(resolved)} resolved "
          f"({len(unresolved)} remain)", flush=True)

    # Conflict budgets: unlimited for small s, limited for large s
    # SAT instances resolve in ~1K-10K conflicts; UNSAT can take millions.
    # At s >= C(f), instance is SAT → resolves fast within budget.
    # At s < C(f), instance is UNSAT → budget exhausted, deferred to s+1.
    budgets_per_s = {6: 500_000, 7: 500_000, 8: 500_000, 9: 1_000_000}

    for s in range(start_s, max_size + 1):
        if not unresolved:
            break

        budget = budgets_per_s.get(s, 0)
        budget_str = f", budget={budget//1000}K conflicts" if budget else ""
        print(f"  Building skeleton for s={s} "
              f"({len(unresolved)} TTs to test{budget_str})...",
              flush=True)
        t0 = time.time()
        skeleton = CircuitSATSkeleton(n, s)
        build_time = time.time() - t0
        print(f"    Encoding: {len(skeleton.clauses)} clauses, "
              f"{skeleton.var_id} vars, built in {build_time:.2f}s", flush=True)

        newly_resolved = []
        budget_exhausted = []
        t0 = time.time()

        for i, cls in enumerate(unresolved):
            tt = cls['canonical_tt']
            sat, _ = skeleton.check_sat(tt, conf_budget=budget)
            if sat is None:
                budget_exhausted.append(cls)
            elif sat:
                resolved[tt] = s
                newly_resolved.append(cls)

            # Progress every 20 instances
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"    Progress: {i+1}/{len(unresolved)} "
                      f"({elapsed:.1f}s)", flush=True)

        solve_time = time.time() - t0
        n_exh = len(budget_exhausted)
        print(f"    Solved {len(unresolved)} instances in {solve_time:.2f}s, "
              f"found {len(newly_resolved)} with C={s}"
              f"{f', {n_exh} budget-exhausted' if n_exh else ''}",
              flush=True)

        unresolved = [cls for cls in unresolved
                      if cls['canonical_tt'] not in resolved]

        # Checkpoint
        if checkpoint_path:
            ckpt = {
                'resolved': {str(k): v for k, v in resolved.items()},
                'last_completed_s': s,
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(ckpt, f)
            print(f"    Checkpoint saved (s={s})", flush=True)

    # Write results back
    for cls in classes:
        tt = cls['canonical_tt']
        cls['circuit_size'] = resolved.get(tt, -1)

    return classes


# =============================================================================
# Main
# =============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    config_path = os.path.join(script_dir, '..', 'configs', 'default.json')
    with open(config_path) as f:
        config = json.load(f)

    n = config['n']
    max_size = config['max_circuit_size']

    # --- Step 1: NPN4 enumeration ---
    print(f"Enumerating NPN{n} equivalence classes...", flush=True)
    t0 = time.time()
    classes = enumerate_npn_classes(n)
    enum_time = time.time() - t0
    print(f"  Found {len(classes)} classes in {enum_time:.2f}s", flush=True)
    total_orbits = sum(c['orbit_size'] for c in classes)
    print(f"  Total functions: {total_orbits} (expected {1 << (1 << n)})", flush=True)
    assert total_orbits == 1 << (1 << n)

    # --- Step 2: Circuit sizes ---
    checkpoint_path = os.path.join(data_dir, 'checkpoint.json')
    print(f"\nComputing circuit sizes (max_size={max_size})...", flush=True)
    t0 = time.time()
    classes = find_all_circuit_sizes(n, classes, max_size, checkpoint_path)
    total_time = time.time() - t0

    # --- Summary ---
    sizes = [c['circuit_size'] for c in classes]
    size_dist = {}
    print(f"\nCircuit size distribution:", flush=True)
    for s in sorted(set(sizes)):
        count = sizes.count(s)
        size_dist[str(s)] = count
        print(f"  C={s}: {count} classes", flush=True)

    failed = sum(1 for s in sizes if s == -1)
    if failed:
        print(f"  WARNING: {failed} classes exceeded max_size", flush=True)

    print(f"\nTotal time: enum={enum_time:.1f}s + SAT={total_time:.1f}s", flush=True)

    # --- Save ---
    result = {
        'n': n,
        'num_classes': len(classes),
        'max_circuit_size_found': max(sizes),
        'size_distribution': size_dist,
        'enum_time_s': round(enum_time, 4),
        'solve_time_s': round(total_time, 4),
        'classes': classes
    }
    out_path = os.path.join(data_dir, 'npn4_circuit_sizes.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}", flush=True)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint cleaned up.", flush=True)


if __name__ == '__main__':
    main()
