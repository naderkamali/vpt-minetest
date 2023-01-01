import pickle
from argparse import ArgumentParser

from agent import MineRLAgent
from gym.wrappers import Monitor, TimeLimit
from minetest_env import Minetest

MINERL_TO_MINETEST_ACTIONS = {
    "ESC": "ESC",  # no VPT action
    "attack": "DIG",
    "back": "BACKWARD",
    "camera": "MOUSE",
    "drop": "DROP",
    "forward": "FORWARD",
    "hotbar.1": "SLOT_1",
    "hotbar.2": "SLOT_2",
    "hotbar.3": "SLOT_3",
    "hotbar.4": "SLOT_4",
    "hotbar.5": "SLOT_5",
    "hotbar.6": "SLOT_6",
    "hotbar.7": "SLOT_7",
    "hotbar.8": "SLOT_8",
    "hotbar.9": "SLOT_1",
    "inventory": "INVENTORY",
    "jump": "JUMP",
    "left": "LEFT",
    "pickItem": "MIDDLE",  # no VPT action
    "right": "RIGHT",
    "sneak": "SNEAK",
    "sprint": None,
    "swapHands": None,  # no VPT action
    "use": "PLACE",
}


def minerl_to_minetest_action(minerl_action, minetest_env):
    minetest_action = {}
    for minerl_key, minetest_key in MINERL_TO_MINETEST_ACTIONS.items():
        if minetest_key and minerl_key in minerl_action:
            if minetest_key != "MOUSE":
                minetest_action[minetest_key] = int(minerl_action[minerl_key][0])
            else:
                # TODO this translation of the camera action maybe wrong
                camera_action = minerl_action[minerl_key][0]
                mouse_action = [0, 0]
                mouse_action[0] = round(
                    0.5
                    * minetest_env.display_size[0]
                    * camera_action[0]
                    / minetest_env.fov_x,
                )
                mouse_action[1] = round(
                    0.5
                    * minetest_env.display_size[1]
                    * camera_action[1]
                    / minetest_env.fov_y,
                )
                print(f"Camera action {camera_action}, mouse action {mouse_action}")
                minetest_action[minetest_key] = mouse_action
    minetest_action["HOTBAR_NEXT"] = minetest_action["HOTBAR_PREV"] = 0
    minetest_action["ESC"] = minetest_action["MIDDLE"] = 0
    return minetest_action


def minetest_to_minerl_obs(minetest_obs):
    return {"pov": minetest_obs}


def main(
    model, weights, video_dir, minetest_path, max_steps, show, seed, show_agent_pov
):
    env = Minetest(minetest_executable=minetest_path, seed=seed)
    env = TimeLimit(env, max_episode_steps=max_steps)
    env.metadata["render.modes"] = ["rgb_array", "ansi"]
    env.metadata["video.frames_per_second"] = 20
    env = Monitor(
        env, video_dir, video_callable=lambda _: True, force=False, resume=True
    )
    print("---Loading model---")
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(
        env,
        policy_kwargs=policy_kwargs,
        pi_head_kwargs=pi_head_kwargs,
        show_agent_perspective=show_agent_pov,
    )
    agent.load_weights(weights)

    print("---Launching Minetest enviroment---")
    obs = minetest_to_minerl_obs(env.reset())
    done = False
    while not done:
        minerl_action = agent.get_action(obs)
        minetest_action = minerl_to_minetest_action(minerl_action, env)
        obs, reward, done, info = env.step(minetest_action)
        obs = minetest_to_minerl_obs(obs)
        if show:
            env.render()
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument(
        "--weights",
        type=str,
        default="../VPT-models/foundation-model-2x.weights",
        help="Path to the '.weights' file to be loaded.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="../VPT-models/foundation-model-2x.model",
        help="Path to the '.model' file to be loaded.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="../videos",
        help="Path to the video recordings.",
    )
    parser.add_argument(
        "--minetest-path",
        type=str,
        default="../minetest/bin/minetest",
        help="Path to minetest executable.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1e6, help="Maximum number of episode steps."
    )
    parser.add_argument("--show", action="store_true", help="Render the environment.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the environment.")
    parser.add_argument(
        "--show-agent-pov", action="store_true", help="Show agent's point of view."
    )

    args = parser.parse_args()

    main(
        args.model,
        args.weights,
        args.video_dir,
        args.minetest_path,
        args.max_steps,
        args.show,
        args.seed,
        args.show_agent_pov,
    )
