from gym.envs.registration import register

print('NOW REGISTERING isolation-v0')
register(
  id='isolation-v0',
  entry_point='gym_isolation.envs:Isolation',
)
