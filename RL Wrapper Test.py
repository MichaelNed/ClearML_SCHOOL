from stable_baselines3.common.env_checker import check_env
from RL_Wrapper import RoboEnv

# Instantiate the environment
env = RoboEnv(RenderMode = True)
# Check the environment
check_env(env)

# Reset the environment
obs = env.reset()
# Run for x steps with random actions
for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # If you are working in a remote Jupyter notebook, you need to use the alternate rendering code you have been using before
    #env.render() 
    if done:
        obs = env.reset()
print(obs)