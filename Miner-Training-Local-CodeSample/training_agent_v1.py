import numpy as np
import gym
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras import optimizers
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import pandas as pd
import datetime
import numpy as np
from MinerEnv_v1 import MinerEnv
import keras.backend as K

HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

# Create header for saving DQN learning file
now = datetime.datetime.now() #Getting the latest datetime
header = ["Ep", "Step", "Reward", "Total_reward", "Action", "Epsilon", "Done", "Termination_Code"] #Defining header for the save file
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv"
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(f, encoding='utf-8', index=False, header=True)

# Parameters for training a DQN model
N_EPISODE = 10000 #The number of episodes for training
MAX_STEP = 100   #The number of steps for each episode
BATCH_SIZE = 32   #The number of experiences for each replay
MEMORY_SIZE = 100000 #The size of the batch for storing experiences
SAVE_NETWORK = 100  # After this number of episodes, the DQN model is saved for testing later.
INITIAL_REPLAY_SIZE = 1000 #The number of experiences are stored in the memory batch before starting replaying

training_steps = 200000
anneal_steps = 195000
# Initialize environment
ENV_NAME = 'GoldMining'
env = MinerEnv(HOST, PORT) #Creating a communication environment between the DQN model and the game environment (GAME_SOCKET_DUMMY.py)
env.start()  # Connect to the game
np.random.seed(123)
# env.seed(123)
nb_actions = env.ACTIONNUM
WINDOW_LENGTH = 1

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH*2,) + env.INPUT_DIM
model = Sequential()
model.add(Permute((2, 3, 1), input_shape=input_shape))
model.add(Convolution2D(32, (7, 7), strides=(1, 1), activation='tanh'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='tanh'))
model.add(Flatten())
model.add(Dense(256, activation='tanh'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=MEMORY_SIZE, window_length=WINDOW_LENGTH)
policy = EpsGreedyQPolicy()
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, enable_double_dqn=True,
               enable_dueling_network=False, dueling_type='avg', target_model_update=1e-2, gamma=0.97,
               policy=policy, vary_eps=True, strategy='linear', anneal_steps=anneal_steps)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=training_steps, visualize=False, verbose=2, nb_max_episode_steps=MAX_STEP)

# After training is done, we save the final weights.
# dqn.save_weights('TrainedModels/duel_dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
dqn.save_model("TrainedModels/", "DQNmodel_" + now.strftime("%Y%m%d-%H%M"))

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)
