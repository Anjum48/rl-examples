# Proximal Policy Optimisation (PPO)
In my tests, I've found that certain modifications need to be made in order to fully solve some environments.
If you have any suggestions, feel free to submit a PR!

## Continuous Environments
- Generally require large batch sizes, e.g. 8192
- If `sigma` starts to increase instead of decrease on TensorBoard, try fewer epochs


### `Pendulum-v0`
- Works fine across a broad range of settings

###  `MountainCarContinuous-v0`
- Not fully tested yet

### `LunarLanderContinuous-v2`
- Not fully tested yet

### `BipedalWalker-v2`
- Reward scaling can sometimes cause issues due to the -100 penalty
- Works best with separate actor-critic networks when using LSTM e.g. `ppo_lstm.py`
- Per episode advantage normalisation when using LSTM (instead of per episode)
- Small amount of L2 kernel regularisation e.g. 0.0001 helps get above 300 points

### `BipedalWalkerHardcore-v2`
- Not solved - need a more robust exploration mechanism (curiosity?)

### `CarRacing-v0`
- Works best with a joined actor-critic e.g. `ppo_lstm_joined.py`
- Not sure why the reward curve goes through highs and lows. Maybe need to play with epochs/learning rate

## Discrete Environments
- Generally require small batch sizes, e.g. 128
- Work in progress...

### `CartPole-v1`
- Works fine across a broad range of settings

### `Pong-v0`
- Work in progress...