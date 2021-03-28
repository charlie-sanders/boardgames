
from gym.envs.registration import register
from .checkers import CheckersEnv

print('NOW REGISTERING checkers-v0')
register(
  id='checkers-v0',
  entry_point='gym_checkers.envs:CheckersEnv',
)
