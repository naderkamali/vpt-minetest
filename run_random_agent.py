from argparse import ArgumentParser

from gym.wrappers import TimeLimit
from minetester import Minetest


def main(minetest_path, max_steps, show, seed):
    env = Minetest(minetest_executable=minetest_path, seed=seed)
    env = TimeLimit(env, max_episode_steps=max_steps)

    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if show:
            env.render()
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run random agent in Minetest environment")

    parser.add_argument(
        "--minetest-path",
        type=str,
        default="../minetest/bin/minetest",
        help="Path to minetest executable.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1e6, help="Maximum number of episode steps.",
    )
    parser.add_argument("--show", action="store_true", help="Render the environment.")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed of the environment.",
    )

    args = parser.parse_args()

    main(args.minetest_path, args.max_steps, args.show, args.seed)
