import argparse
import csv
import os
import random
import time

import numpy as np
import gymnasium as gym


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--outdir", type=str, default=os.path.join("experiments", "2026-02-27"))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    env = gym.make(args.env_id)
    # Gymnasium uses env.reset(seed=...) for reproducibility
    obs, info = env.reset(seed=args.seed)

    csv_path = os.path.join(args.outdir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "episode_return",
                "episode_length",
                "seed",
                "env_id",
                "wall_time_sec",
            ],
        )
        writer.writeheader()

        returns = []
        lengths = []
        t0 = time.time()

        for ep in range(1, args.episodes + 1):
            obs, info = env.reset(seed=args.seed + ep)  # vary per-episode seed
            done = False
            ep_ret = 0.0
            ep_len = 0

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_ret += float(reward)
                ep_len += 1

            returns.append(ep_ret)
            lengths.append(ep_len)

            writer.writerow(
                {
                    "episode": ep,
                    "episode_return": ep_ret,
                    "episode_length": ep_len,
                    "seed": args.seed,
                    "env_id": args.env_id,
                    "wall_time_sec": round(time.time() - t0, 3),
                }
            )

            if ep % 10 == 0 or ep == 1:
                print(f"[ep {ep:03d}] return={ep_ret:.1f} len={ep_len}")

    env.close()

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    mean_len = float(np.mean(lengths))

    print("\n=== Random Policy Baseline ===")
    print(f"env_id: {args.env_id}")
    print(f"episodes: {args.episodes}")
    print(f"seed: {args.seed}")
    print(f"mean_return: {mean_ret:.2f} ± {std_ret:.2f}")
    print(f"mean_length: {mean_len:.2f}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()