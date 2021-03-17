from gym.envs.registration import register

print('NOW REGISTERING tictictoe-v0')
register(
  id='tictictoe-v0',
  entry_point='gym_tictictoe.envs:TicTicToe',
)
