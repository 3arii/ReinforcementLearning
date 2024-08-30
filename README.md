# Instructions:

1. `python3 train.py`
2. `python3 generate_trajectory.py`

# Explanation

- The `train.py` will train the environment and save it to a `ppo_humanoid.zip` which can be loaded and used later.
- The `generate_trajectory.py` will generate two tuples and save it as `npy` files. The first is actions and the scond being observations.
- The action is the prediction using the trained model and the observation is the action applied to the environment
