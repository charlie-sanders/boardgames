from gym.envs.registration import register

print('NOW REGISTERING singlestock-v0')
register(
  id='singlestock-v0',
  entry_point='gym_singlestock.envs:SingleStockEnv',
)
