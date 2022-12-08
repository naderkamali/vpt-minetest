from argparse import ArgumentParser

from gym.wrappers import TimeLimit
from minetest_env import Minetest


def main(minetest_dir, max_steps, show):
    env = Minetest(autostart_minetest=True, minetest_dir=minetest_dir)
    env = TimeLimit(env, max_episode_steps=max_steps)

    with env:
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, rew, done, info = env.step(action)
            if show:
                env.render()


if __name__ == "__main__":
    parser = ArgumentParser("Run random agent in Minetest environment")

    parser.add_argument(
        "--minetest-dir",
        type=str,
        default="../minetest",
        help="Path to minetest directory.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1e6, help="Maximum number of episode steps.",
    )
    parser.add_argument("--show", action="store_true", help="Render the environment.")

    args = parser.parse_args()

    main(args.minetest_dir, args.max_steps, args.show)