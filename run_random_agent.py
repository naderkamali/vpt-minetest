from minetest_env import Minetest

with Minetest(autostart_minetest=True, minetest_dir="../minetest") as env:
    obs = env.reset()
    render = False
    done = False
    while not done:
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if render:
            env.render()
