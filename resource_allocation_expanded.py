"""
=====================================================================
 RESOURCE ALLOCATION OPTIMIZATION SYSTEM  (DAA Mini Project)
 Complete single-file code
=====================================================================
 PROBLEM
   A college has a LIMITED BUDGET (the resource) and many funding
   requests (projects). Every request has a COST and a BENEFIT score.
   Which requests should be funded so the TOTAL BENEFIT is maximum
   without crossing the budget?   -> this is the 0/1 KNAPSACK problem.
   (Same model works for CPU time, man-hours, machine hours, storage...)

 WHAT THIS PROGRAM DOES
   1. Stores the funding requests (items) and the total budget.
   2. Solves the problem with 4 algorithms:
        - Backtracking (Brute Force)  : tries every combination   O(2^n)
        - Greedy (benefit/cost ratio) : fast but NOT always best  O(n log n)
        - Dynamic Programming         : always optimal            O(n*W)
        - Branch and Bound            : optimal, prunes bad paths O(2^n) worst
      (+ Fractional Greedy, used only as a theoretical UPPER BOUND)
   3. Compares them (benefit, budget used, time, operations).
   4. Benchmarks them on bigger inputs (10, 15, 20, 50, 100, 500 items).
   5. Shows LIVE graphs in pop-up windows (nothing needs to be saved):
        - animated allocation (watch items being funded one by one)
        - allocation of all 4 algorithms side by side
        - performance graphs (time, operations vs theory, solution quality)
        - DP table heat-map
      You choose which graphs to see; the program ALWAYS ends with the
      final OPTIMAL allocation graph.
   6. Built-in test cases:   python resource_allocation_system.py --test

 HOW TO RUN
   Step 1 (only once):   pip install matplotlib numpy
   Step 2:               python resource_allocation_system.py
   Then type the total budget (or press Enter for the default 60).
 Close each graph window to move to the next one.
 (Tip: take your project screenshots directly from these windows.)
=====================================================================
"""

import heapq            # priority queue (max-heap via negatives) for Branch & Bound
import random           # random instances for benchmarking / testing
import sys              # command-line flag  --test
import time             # to measure execution time

import matplotlib
# (No matplotlib.use("Agg") here: we WANT windows to open for live graphs)
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# PART 1: DATA  (items = funding requests, capacity = total budget)
# ---------------------------------------------------------------------
# Each request: name -> (cost in Rs. lakh, benefit score out of 100)
REQUESTS = {
    "Computer Lab Upgrade":     (20, 80),
    "Library Digital Access":   (8, 30),
    "Wi-Fi Expansion":          (12, 55),
    "Sports Equipment":         (10, 25),
    "Hostel Renovation":        (25, 70),
    "Research Grants":          (18, 75),
    "Faculty Training":         (6, 28),
    "Solar Panels":             (22, 60),
    "Canteen Upgrade":          (9, 20),
    "Lab Safety Equipment":     (7, 35),
    "Smart Classrooms":         (15, 62),
    "Bus Fleet Maintenance":    (14, 38),
}
DEFAULT_BUDGET = 60             # Rs. lakh


def make_items(requests=REQUESTS):
    """Item representation: list of (name, cost, benefit) tuples.
    Space O(n)."""
    return [(name, c, b) for name, (c, b) in requests.items()]


def random_instance(n, seed=1):
    """Random instance with n items (used for benchmarking and tests).
    cost 5..50, benefit 10..100, budget = 40% of the total cost."""
    rnd = random.Random(seed)       # fixed seed = same data every run
    items = [(f"Item {i}", rnd.randint(5, 50), rnd.randint(10, 100))
             for i in range(n)]
    capacity = int(0.4 * sum(c for _, c, _ in items))
    return items, capacity


def total_cost(items, chosen):
    return sum(items[i][1] for i in chosen)


def total_benefit(items, chosen):
    return sum(items[i][2] for i in chosen)


# ---------------------------------------------------------------------
# PART 2: THE ALGORITHMS
# Each returns (chosen, ops)
#   chosen = sorted list of item indices that are funded
#   ops    = number of basic steps done (used to compare with theory)
# ---------------------------------------------------------------------
def backtracking(items, capacity):
    """BACKTRACKING / BRUTE FORCE - for every item decide INCLUDE or
    EXCLUDE (a binary decision tree). A branch is cut (backtracked) only
    when the budget is exceeded. Always optimal but exponential.
    Time O(2^n), Space O(n) recursion depth."""
    n = len(items)
    best = {"value": -1, "set": []}
    ops = [0]
    current = []

    def explore(i, cost, value):
        ops[0] += 1                             # one tree node visited
        if i == n:                              # decided about every item
            if value > best["value"]:
                best["value"], best["set"] = value, current[:]
            return
        _, c, b = items[i]
        if cost + c <= capacity:                # branch 1: include item i
            current.append(i)
            explore(i + 1, cost + c, value + b)
            current.pop()                       # BACKTRACK
        explore(i + 1, cost, value)             # branch 2: exclude item i

    explore(0, 0, 0)
    return sorted(best["set"]), ops[0]


def greedy(items, capacity):
    """GREEDY - sort by benefit/cost ratio (best 'value for money' first)
    and fund an item whenever it still fits. Very fast, but a locally
    good choice can block a better combination, so NOT always optimal.
    Time O(n log n), Space O(n)."""
    order = sorted(range(len(items)),
                   key=lambda i: items[i][2] / items[i][1], reverse=True)
    ops = len(items) * max(1, len(items).bit_length())   # cost of the sort
    chosen, used = [], 0
    for i in order:
        ops += 1
        if used + items[i][1] <= capacity:
            chosen.append(i)
            used += items[i][1]
    return sorted(chosen), ops


def fractional_greedy_value(items, capacity):
    """Fractional knapsack (a request may be funded PARTIALLY). Greedy is
    optimal here, so this value is an UPPER BOUND for the 0/1 problem."""
    order = sorted(range(len(items)),
                   key=lambda i: items[i][2] / items[i][1], reverse=True)
    left, value = capacity, 0.0
    for i in order:
        _, c, b = items[i]
        if c <= left:
            left -= c
            value += b
        else:
            value += b * left / c
            break
    return value


def dp_table(items, capacity):
    """Build the DP table.  T[i][w] = best benefit using the first i items
    with budget w.
        T[i][w] = T[i-1][w]                                  (skip item i)
        T[i][w] = max(T[i-1][w], T[i-1][w-c] + b)  if c <= w (take item i)
    Time O(n*W), Space O(n*W)."""
    n = len(items)
    T = [[0] * (capacity + 1) for _ in range(n + 1)]
    ops = 0
    for i in range(1, n + 1):
        _, c, b = items[i - 1]
        prev, row = T[i - 1], T[i]
        for w in range(capacity + 1):
            ops += 1
            row[w] = prev[w]
            if c <= w and prev[w - c] + b > row[w]:
                row[w] = prev[w - c] + b
    return T, ops


def dynamic_programming(items, capacity):
    """DYNAMIC PROGRAMMING - fills the table above and then walks back
    through it to find which items were taken. Always optimal.
    Time O(n*W), Space O(n*W)."""
    T, ops = dp_table(items, capacity)
    chosen, w = [], capacity
    for i in range(len(items), 0, -1):          # trace back
        if T[i][w] != T[i - 1][w]:              # value changed -> item i taken
            chosen.append(i - 1)
            w -= items[i - 1][1]
    return sorted(chosen), ops + len(items)


def branch_and_bound(items, capacity):
    """BRANCH AND BOUND - best-first search. For every partial solution we
    compute an optimistic BOUND (fractional greedy on the remaining items).
    If the bound cannot beat the best solution found so far, that whole
    branch is discarded (pruned). Always optimal, usually far fewer nodes
    than backtracking.   Worst-case Time O(2^n), Space O(2^n) heap."""
    n = len(items)
    order = sorted(range(n), key=lambda i: items[i][2] / items[i][1], reverse=True)
    ops = [0]

    def bound(level, cost, value):
        """Optimistic estimate of the best possible value from this node."""
        left, est = capacity - cost, value
        for k in range(level, n):
            _, c, b = items[order[k]]
            if c <= left:
                left -= c
                est += b
            else:
                return est + b * left / c       # take a fraction and stop
        return est

    best_value, best_mask = 0, 0
    tie = 0                                     # tie-breaker for the heap
    heap = [(-bound(0, 0, 0), tie, 0, 0, 0, 0)]  # (-bound, tie, level, cost, value, mask)
    while heap:
        neg_b, _, level, cost, value, mask = heapq.heappop(heap)
        ops[0] += 1                             # one node expanded
        if -neg_b <= best_value or level == n:  # PRUNE: cannot beat the best
            continue
        idx = order[level]
        _, c, b = items[idx]
        if cost + c <= capacity:                # child 1: include item
            nv, nm = value + b, mask | (1 << idx)
            if nv > best_value:
                best_value, best_mask = nv, nm
            ub = bound(level + 1, cost + c, nv)
            if ub > best_value:
                tie += 1
                heapq.heappush(heap, (-ub, tie, level + 1, cost + c, nv, nm))
        ub = bound(level + 1, cost, value)      # child 2: exclude item
        if ub > best_value:
            tie += 1
            heapq.heappush(heap, (-ub, tie, level + 1, cost, value, mask))
    chosen = [i for i in range(n) if best_mask >> i & 1]
    return chosen, ops[0]


ALGORITHMS = {
    "Backtracking": backtracking,
    "Greedy": greedy,
    "DP": dynamic_programming,
    "Branch&Bound": branch_and_bound,
}
NAMES = tuple(ALGORITHMS)
BRUTE_FORCE_LIMIT = 20          # backtracking is skipped above this many items


# ---------------------------------------------------------------------
# PART 3: RUN ALL ALGORITHMS ON ONE INPUT AND MEASURE THEM
# ---------------------------------------------------------------------
def run_all(items, capacity):
    results = {}
    for name, algo in ALGORITHMS.items():
        t0 = time.perf_counter()
        chosen, ops = algo(items, capacity)
        ms = (time.perf_counter() - t0) * 1000          # milliseconds
        results[name] = {"chosen": chosen, "ops": ops, "ms": ms,
                         "benefit": total_benefit(items, chosen),
                         "cost": total_cost(items, chosen)}
    return results


def benchmark(sizes=(10, 15, 20, 50, 100, 500)):
    """Input size vs execution time. Each algorithm is repeated several
    times (adaptive) and the average is taken, because small runs are too
    fast to measure. Backtracking is skipped for n > 20 (2^n too large)."""
    table = {}
    for n in sizes:
        items, cap = random_instance(n)
        table[n] = {}
        for name, algo in ALGORITHMS.items():
            if name == "Backtracking" and n > BRUTE_FORCE_LIMIT:
                table[n][name] = None
                continue
            t0 = time.perf_counter()
            chosen, ops = algo(items, cap)              # first run (also a timing probe)
            first = time.perf_counter() - t0
            repeats = max(1, min(100, int(0.3 / max(first, 1e-9))))
            t0 = time.perf_counter()
            for _ in range(repeats):
                algo(items, cap)
            avg_ms = (time.perf_counter() - t0) * 1000 / repeats
            table[n][name] = {"ms": avg_ms, "ops": ops,
                              "benefit": total_benefit(items, chosen)}
    return table


# ---------------------------------------------------------------------
# PART 4: LIVE GRAPHS (windows open while the program runs)
# ---------------------------------------------------------------------
TEAL, GREY, ORANGE, NAVY = "#0E9AA7", "#c5ccd6", "#f4a261", "#1B3A4B"


def draw_allocation(ax, items, chosen, title, capacity, bar="benefit"):
    """Horizontal bars, one per request.
    teal = funded | grey = rejected. Bar length = benefit score."""
    names = [n for n, _, _ in items]
    vals = [b for _, _, b in items]
    colors = [TEAL if i in chosen else GREY for i in range(len(items))]
    ax.clear()
    ax.barh(names, vals, color=colors)
    for i, (_, c, b) in enumerate(items):
        ax.text(b + 1, i, f"Rs.{c}L", va="center", fontsize=7)
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlim(0, max(vals) * 1.2)
    ax.set_xlabel("Benefit score (label = cost)", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")


def live_allocation(name, items, result, capacity, delay=0.6):
    """WINDOW 1: animate the allocation. Requests are funded one by one
    and the budget bar fills up."""
    fig, (ax, gauge) = plt.subplots(1, 2, figsize=(11, 5.5),
                                    gridspec_kw={"width_ratios": [4, 1]})
    fig.canvas.manager.set_window_title(f"Live allocation: {name}")
    shown, used, gain = [], 0, 0
    for i in result["chosen"]:
        shown.append(i)
        used += items[i][1]
        gain += items[i][2]
        draw_allocation(ax, items, shown,
                        f"{name}: funding '{items[i][0]}'  |  benefit so far = {gain}",
                        capacity)
        gauge.clear()
        gauge.bar([0], [used], color=TEAL)
        gauge.axhline(capacity, color="#d62828", ls="--")
        gauge.set_ylim(0, capacity * 1.1)
        gauge.set_xticks([])
        gauge.set_title(f"Budget used\n{used} / {capacity}", fontsize=9)
        plt.pause(delay)
    ax.set_title(f"{name} finished: benefit = {gain}, budget used = {used}/{capacity}",
                 fontsize=10, fontweight="bold")
    print(f"\n[Window open] {name} animation finished. Close the window to continue.")
    plt.show()


def compare_allocations(items, results, capacity):
    """WINDOW 2: the 4 algorithms side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.canvas.manager.set_window_title("Allocation comparison")
    for ax, name in zip(axes.flat, NAMES):
        r = results[name]
        draw_allocation(ax, items, r["chosen"],
                        f"{name}: benefit {r['benefit']}, cost {r['cost']}/{capacity}",
                        capacity)
    fig.suptitle("Funded requests (teal) vs rejected (grey)")
    fig.tight_layout()
    print("[Window open] Allocation comparison. Close the window to continue.")
    plt.show()


def performance_graphs(bench):
    """WINDOW 3: (a) input size vs time, (b) operations vs theory,
    (c) solution quality of Greedy compared to the optimum."""
    sizes = list(bench)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.canvas.manager.set_window_title("Performance")
    markers = {"Backtracking": "s", "Greedy": "^", "DP": "o", "Branch&Bound": "D"}

    # (a) time
    for k in NAMES:
        pts = [(n, bench[n][k]["ms"]) for n in sizes if bench[n][k]]
        axes[0].plot(*zip(*pts), marker=markers[k], label=k)
    axes[0].set(xlabel="Number of requests (n)", ylabel="Average time (ms)",
                title="Input size vs execution time", yscale="log")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # (b) operations (experimental) vs theoretical growth
    for k in NAMES:
        pts = [(n, bench[n][k]["ops"]) for n in sizes if bench[n][k]]
        axes[1].plot(*zip(*pts), marker=markers[k], label=f"{k} (measured)")
    small = [n for n in sizes if n <= BRUTE_FORCE_LIMIT]
    axes[1].plot(small, [2 ** n for n in small], "k:", label="2^n (theory)")
    axes[1].set(xlabel="Number of requests (n)", ylabel="Basic operations",
                title="Measured operations vs theory", yscale="log")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    # (c) quality
    for k in ("Greedy",):
        pct = [100 * bench[n][k]["benefit"] / bench[n]["DP"]["benefit"] for n in sizes]
        axes[2].bar([str(n) for n in sizes], pct, color=ORANGE, label="Greedy")
        for x, v in enumerate(pct):
            axes[2].text(x, v + 0.05, f"{v:.2f}%", ha="center", fontsize=7)
    axes[2].axhline(100, color=TEAL, ls="--", label="Optimal (DP / B&B / Backtracking)")
    axes[2].set(xlabel="Number of requests (n)", ylabel="% of optimal benefit",
                title="Solution quality of Greedy", ylim=(98, 100.6))
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    print("[Window open] Performance graphs. Close the window to continue.")
    plt.show()


def dp_heatmap(items, capacity):
    """WINDOW 4: the DP table as a heat-map (rows = items, columns = budget)."""
    T, _ = dp_table(items, capacity)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.canvas.manager.set_window_title("DP table")
    im = ax.imshow(np.array(T), aspect="auto", cmap="YlGnBu", origin="upper")
    ax.set_yticks(range(len(items) + 1))
    ax.set_yticklabels(["(none)"] + [n for n, _, _ in items], fontsize=7)
    ax.set(xlabel="Budget available (Rs. lakh)", ylabel="Requests considered so far",
           title="Dynamic Programming table: best benefit T[i][w]")
    fig.colorbar(im, ax=ax, label="Best benefit")
    fig.tight_layout()
    print("[Window open] DP table heat-map. Close the window to continue.")
    plt.show()


def optimal_allocation_graph(items, results, capacity):
    """FINAL WINDOW: the optimal allocation. DP, Backtracking and Branch &
    Bound all give the same optimum, so we use DP (and confirm they match)."""
    best = results["DP"]
    assert best["benefit"] == results["Branch&Bound"]["benefit"]
    assert best["benefit"] == results["Backtracking"]["benefit"]
    chosen = best["chosen"]
    fig, (ax, pie) = plt.subplots(1, 2, figsize=(12, 6),
                                  gridspec_kw={"width_ratios": [3, 2]})
    fig.canvas.manager.set_window_title("Optimal allocation")
    draw_allocation(ax, items, chosen,
                    f"OPTIMAL ALLOCATION | Benefit = {best['benefit']} | "
                    f"Cost = {best['cost']} / {capacity}", capacity)
    sizes = [items[i][1] for i in chosen]
    labels = [items[i][0] for i in chosen]
    left = capacity - best["cost"]
    if left > 0:
        sizes.append(left)
        labels.append("Unspent")
    cmap = plt.get_cmap("tab20")
    colors = [cmap(k % 20) for k in range(len(chosen))] + ([GREY] if left > 0 else [])
    pie.pie(sizes, labels=labels, colors=colors, autopct=lambda p: f"{p:.0f}%",
            textprops={"fontsize": 7})
    pie.set_title("How the budget is split", fontsize=10, fontweight="bold")
    fig.tight_layout()
    print("\n[Window open] Optimal allocation: " + ", ".join(items[i][0] for i in chosen))
    plt.show()


# ---------------------------------------------------------------------
# PART 5: TEST CASES   (run with:  python resource_allocation_system.py --test)
# ---------------------------------------------------------------------
def run_tests():
    """Checks correctness of every algorithm."""
    passed = 0

    def check(label, condition):
        nonlocal passed
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
        passed += bool(condition)
        assert condition, label

    print("TEST 1: classic textbook example (cost,benefit)=(10,60),(20,100),(30,120), W=50")
    items = [("A", 10, 60), ("B", 20, 100), ("C", 30, 120)]
    for name, algo in ALGORITHMS.items():
        chosen, _ = algo(items, 50)
        expected = 160 if name == "Greedy" else 220
        check(f"{name:13s} benefit = {total_benefit(items, chosen)} (expected {expected})",
              total_benefit(items, chosen) == expected)

    print("TEST 2: nothing fits (budget smaller than every cost)")
    for name, algo in ALGORITHMS.items():
        chosen, _ = algo(items, 5)
        check(f"{name:13s} chooses nothing", chosen == [])

    print("TEST 3: budget big enough for everything")
    for name, algo in ALGORITHMS.items():
        chosen, _ = algo(items, 1000)
        check(f"{name:13s} chooses all items", chosen == [0, 1, 2])

    print("TEST 4: 200 random instances - exact algorithms must agree, Greedy <= optimum")
    ok = True
    for seed in range(200):
        n = random.Random(seed).randint(1, 12)
        its, cap = random_instance(n, seed)
        r = run_all(its, cap)
        opt = r["DP"]["benefit"]
        ok &= r["Backtracking"]["benefit"] == opt == r["Branch&Bound"]["benefit"]
        ok &= r["Greedy"]["benefit"] <= opt
        ok &= all(r[a]["cost"] <= cap for a in NAMES)   # budget never exceeded
        ok &= opt <= fractional_greedy_value(its, cap) + 1e-9
    check("all exact algorithms agree, Greedy never beats optimum, budget respected", ok)

    print("TEST 5: default campus dataset")
    its = make_items()
    r = run_all(its, DEFAULT_BUDGET)
    check("DP == Backtracking == Branch&Bound",
          r["DP"]["benefit"] == r["Backtracking"]["benefit"] == r["Branch&Bound"]["benefit"])
    print(f"\nAll {passed} checks passed.")


# ---------------------------------------------------------------------
# PART 6: MAIN PROGRAM
# ---------------------------------------------------------------------
def main():
    items = make_items()
    print("Funding requests (cost in Rs. lakh, benefit score):")
    for name, c, b in items:
        print(f"  {name:25s} cost = {c:3d}   benefit = {b:3d}   ratio = {b / c:.2f}")
    txt = input(f"\nEnter total budget in Rs. lakh [{DEFAULT_BUDGET}]: ").strip()
    capacity = int(txt) if txt.isdigit() and int(txt) > 0 else DEFAULT_BUDGET

    # Run all algorithms and print the comparison table.
    results = run_all(items, capacity)
    print(f"\nAllocation with budget = {capacity}")
    print("-" * 78)
    for name, r in results.items():
        print(f"{name:13s} benefit = {r['benefit']:4d} | cost = {r['cost']:3d}/{capacity} "
              f"| operations = {r['ops']:6d} | time = {r['ms']:.4f} ms")
        print("              funded:", ", ".join(items[i][0] for i in r["chosen"]) or "nothing")
    ub = fractional_greedy_value(items, capacity)
    gap = 100 * (results["DP"]["benefit"] - results["Greedy"]["benefit"]) / results["DP"]["benefit"]
    print(f"\nFractional (upper bound) = {ub:.1f} | Greedy is {gap:.1f}% below the optimum")

    # Benchmark on bigger inputs.
    print("\nBenchmark: average time in ms (random data; '-' = skipped, 2^n too large)")
    print(f"{'n':>5} {'Backtrack':>10} {'Greedy':>9} {'DP':>9} {'B&B':>9}")
    bench = benchmark()
    for n, row in bench.items():
        cells = ["      -   " if row[k] is None else f"{row[k]['ms']:10.3f}" for k in NAMES]
        print(f"{n:>5} {cells[0]} {cells[1][1:]:>9} {cells[2][1:]:>9} {cells[3][1:]:>9}")

    # ---- Ask which graph(s) the user wants to see ----
    print("\nWhich graph do you want to see?")
    print("  1 = Live allocation animation (one algorithm)")
    print("  2 = Allocation comparison (all 4 algorithms)")
    print("  3 = Performance graphs (time, operations, quality)")
    print("  4 = DP table heat-map")
    print("  5 = ALL graphs")
    print("  0 = Skip (only show the optimal allocation)")
    choice = input("Enter choice(s), e.g. 1 or 1,3 or 5 [5]: ").strip() or "5"
    picks = {c.strip() for c in choice.replace(" ", ",").split(",") if c.strip()}
    if "5" in picks:
        picks = {"1", "2", "3", "4"}

    if "1" in picks:
        algo = input("Algorithm to animate (Backtracking/Greedy/DP/Branch&Bound) [DP]: ").strip() or "DP"
        if algo not in NAMES:
            algo = "DP"
        live_allocation(algo, items, results[algo], capacity)
    if "2" in picks:
        compare_allocations(items, results, capacity)
    if "3" in picks:
        performance_graphs(bench)
    if "4" in picks:
        dp_heatmap(items, capacity)

    # ---- Always finish with the optimal allocation graph ----
    optimal_allocation_graph(items, results, capacity)


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_tests()
    else:
        main()