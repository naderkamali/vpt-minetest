import pickle
from argparse import ArgumentParser

from agent import MineRLAgent
from gym.wrappers import Monitor, TimeLimit
from minetest_env import DISPLAY_SIZE, FOV_X, FOV_Y, Minetest

MINERL_TO_MINETEST_ACTIONS = {
    "ESC": "ESC",
    "attack": "dig",
    "back": "backward",
    "camera": "mouse",
    "drop": "drop",
    "forward": "forward",
    "hotbar.1": "slot1",
    "hotbar.2": "slot2",
    "hotbar.3": "slot3",
    "hotbar.4": "slot4",
    "hotbar.5": "slot5",
    "hotbar.6": "slot6",
    "hotbar.7": "slot7",
    "hotbar.8": "slot8",
    "hotbar.9": "slot1",
    "inventory": "inventory",
    "jump": "jump",
    "left": "left",
    "pickItem": "middle",
    "right": "right",
    "sneak": "sneak",
    "sprint": None,
    "swapHands": None,
    "use": "place",
}


def minerl_to_minetest_action(minerl_action):
    minetest_action = {}
    for minerl_key, minetest_key in MINERL_TO_MINETEST_ACTIONS.items():
        if minetest_key and minerl_key in minerl_action:
            if minetest_key != "mouse":
                minetest_action[minetest_key] = int(minerl_action[minerl_key][0])
            else:
                # TODO this translation of the camera action maybe wrong
                camera_action = minerl_action[minerl_key][0]
                mouse_action = [0, 0]
                mouse_action[0] = 2 * round(DISPLAY_SIZE[0] * camera_action[0] / FOV_X)
                mouse_action[1] = 2 * round(DISPLAY_SIZE[1] * camera_action[1] / FOV_Y)
                print(f"Camera action {camera_action}, mouse action {mouse_action}")
                minetest_action[minetest_key] = mouse_action
    return minetest_action


def minetest_to_minerl_obs(minetest_obs):
    return {"pov": minetest_obs}


def main(model, weights, video_dir, max_steps, show):
    env = Minetest()
    env = TimeLimit(env, max_episode_steps=max_steps)
    env.metadata["render.modes"] = ["rgb_array", "ansi"]
    env.metadata["video.frames_per_second"] = 20
    env = Monitor(env, video_dir, video_callable=lambda _: True, force=False, resume=True)
    print("---Loading model---")
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    print("---Launching Minetest enviroment---")
    obs = minetest_to_minerl_obs(env.reset())
    done = False
    while not done:
        minerl_action = agent.get_action(obs)
        minetest_action = minerl_to_minetest_action(minerl_action)
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
        default="../VPT-models/foundation-model-1x.weights",
        help="Path to the '.weights' file to be loaded.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="../VPT-models/foundation-model-1x.model",
        help="Path to the '.model' file to be loaded.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="../videos",
        help="Path to the video recordings.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1e6, help="Maximum number of episode steps."
    )
    parser.add_argument("--show", action="store_true", help="Render the environment.")

    args = parser.parse_args()

    main(args.model, args.weights, args.video_dir, args.max_steps, args.show)
