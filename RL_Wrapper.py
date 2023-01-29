import gym
from gym import spaces
import numpy as np
import robosuite as suite
from scipy.spatial.transform import Rotation as R


def quat_to_rpy(q):
            #convert quaternion to roll, pitch, yaw
            rpy = R.from_quat(q).as_euler('xyz', degrees=True)
            #transform yaw to be between -90 and 90
            if rpy[2]>90:
                rpy[2] = rpy[2]-180
            elif rpy[2]<-90:
                rpy[2] = rpy[2]+180
            return rpy[2]


class RoboEnv(gym.Env):
    def __init__(self, RenderMode = False, Task = 'PickPlace'): # Add any arguments you need (Environment settings; Render mode  and task are used as examples)
        super(RoboEnv, self).__init__()
        # Initialize environment variables
        self.RenderMode = RenderMode
        self.Task = Task

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1, high=1, shape= (8,))
        # Example for using image as input:
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(36,), dtype=np.float64)

        # Instantiate the environment
        self.env = suite.make(env_name= self.Task, 
                                robots="Panda",
                                has_renderer=self.RenderMode,
                                has_offscreen_renderer=False,
                                horizon=500,    
                                use_camera_obs=False,)


    def step(self, action):
        # Execute one time step within the environment
        # action = # Process the action if needed
        #Call the environment step function
        obs, reward, done, _ = self.env.step(action)
        # You may find it useful to create helper functions for the following

        gripper_pos = obs["robot0_eef_pos"]
        yaw_robot = quat_to_rpy(obs["robot0_eef_quat"])


        obs = np.hstack((obs["robot0_proprio-state"],self.target_pos))
        obs = np.hstack((obs, self.target_yaw))

        reward1 = 1 / np.linalg.norm(self.target_pos - gripper_pos)
        reward2 = 1 / np.linalg.norm(self.target_yaw - yaw_robot)
        reward2 = np.clip(reward2,-2,2)


        print("Reward1: ", reward1)
        print("Reward2: ", reward2)


        reward = reward1 + reward2
        #self.env.render()
        # done = # Calculate if the episode is done if you want to terminate the episode early
        return obs, reward, done, _

    def reset(self):
        # Reset the state of the environment to an initial state
        # Call the environment reset function
        obs = self.env.reset()
        # Reset any variables that need to be reset
        # Example of generating random values
        x = np.random.uniform(-0.3, 0.3)
        y = np.random.uniform(-0.5, 0.5)
        z = np.random.uniform(0.8, 1.3)
        yaw = np.random.uniform(-90, 90)

        self.target_pos = np.array([x, y, z], dtype=np.float64)
        self.target_yaw = np.array([yaw])


        obs = np.hstack((obs["robot0_proprio-state"],self.target_pos))
        obs = np.hstack((obs, self.target_yaw))

        obs = np.array(obs,dtype=np.float64)
        return obs



    def render(self):
        # Render the environment to the screen
    
        self.env.render()

    def close (self):
        # Close the environment
        self.env.close()