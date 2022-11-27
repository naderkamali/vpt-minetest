# Minetest Gym Environment

The gym env sends actions to the Minetest client and receives back image observations.
Can only be used with [EleutherAI's minetest fork](https://github.com/EleutherAI/minetest).

You can check your setup by running
- the minetest server
- the minetest client
- `python run_random_agent.py`

# Run VPT agents

- Download VPT model and weights file from https://github.com/openai/Video-Pre-Training#agent-model-zoo
- start minetest server and client
- `python run_vpt_agent.py --model MODEL_FILE --weights WEIGHTS_FILE`




