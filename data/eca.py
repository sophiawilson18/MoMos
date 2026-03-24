import numpy as np
import argparse
import os

def cellular_automaton(rule_number, size, burnin=1000, steps=48):
    rule = np.array([int(x) for x in f"{rule_number:08b}"], dtype=np.uint8)[::-1]

    total = burnin + steps + 1
    grid = np.zeros((total, size), dtype=np.uint8)
    grid[0] = np.random.randint(0, 2, size)

    for t in range(1, total):
        for i in range(size):
            left   = grid[t-1, (i-1) % size]
            center = grid[t-1, i]
            right  = grid[t-1, (i+1) % size]
            idx = left * 4 + center * 2 + right
            grid[t, i] = rule[idx]

    X = grid[burnin]
    Y = grid[burnin + steps]
    return X, Y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules",  type=int, nargs="+", required=True)
    parser.add_argument("--size",   type=int, default=64)
    parser.add_argument("--steps",  type=int, default=48)
    parser.add_argument("--burnin", type=int, default=1000)
    parser.add_argument("--n",      type=int, default=10)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    for rule in args.rules:
        rule_dir = os.path.join(script_dir, "output", f"rule{rule}")
        os.makedirs(rule_dir, exist_ok=True)

        X_all = np.zeros((args.n, args.size), dtype=np.uint8)
        Y_all = np.zeros((args.n, args.size), dtype=np.uint8)

        for i in range(args.n):
            X_all[i], Y_all[i] = cellular_automaton(rule, args.size, args.burnin, args.steps)

        np.save(os.path.join(rule_dir, "X.npy"), X_all)
        np.save(os.path.join(rule_dir, "Y.npy"), Y_all)

        print(f"Rule {rule}: saved {args.n} datapoints to output/rule{rule}/")
