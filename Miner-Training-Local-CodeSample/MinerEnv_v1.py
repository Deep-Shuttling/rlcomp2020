import sys
import numpy as np
from GAME_SOCKET_DUMMY import GameSocket    # in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from MINER_STATE import State
from random import seed
from random import randint

TreeID = 1
TrapID = 2
SwampID = 3
MAP_MAX_X = 21  # Width of the Map
MAP_MAX_Y = 9   # Height of the Map

class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()

        # define action space
        # self.INPUT_DIM = (21, 9, 2)  # The number of input values for the DQN model
        self.INPUT_DIM = (21, 9)
        self.ACTIONNUM = 6  # The number of actions output from the DQN model
        # define state space

        self.gameState = None
        self.reward = 0
        self.terminate = False
        self.gold_map = None
        self.dist_gold = None
        
        self.score_pre = self.state.score   # Storing the last score for designing the reward function
        self.energy_pre = self.state.energy # Storing the last energy for designing the reward function

        self.viewer = None
        self.steps_beyond_done = None

    def start(self):    # connect to server
        self.socket.connect()

    def end(self):  # disconnect server
        self.socket.close()

    def send_map_info(self, request):   # tell server which map to run
        self.socket.send(request)

    def reset(self):    # start new game
        # Choosing a map in the list
        # mapID = np.random.randint(1, 6)  # Choosing a map ID from 5 maps in Maps folder randomly
        mapID = 1
        posID_x = np.random.randint(MAP_MAX_X)  # Choosing a initial position of the DQN agent on
        # X-axes randomly
        posID_y = np.random.randint(MAP_MAX_Y)  # Choosing a initial position of the DQN agent on Y-axes randomly
        # Creating a request for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
        request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100")
        # Send the request to the game environment (GAME_SOCKET_DUMMY.py)
        self.send_map_info(request)

        # Initialize the game environment
        try:
            message = self.socket.receive() #receive game info from server
            self.state.init_state(message) #init state
        except Exception as e:
            import traceback
            traceback.print_exc()

        self.gameState = self.get_state()  # Get the state after resetting.
        # This function (get_state()) is an example of creating a state for the DQN model
        distance = 500
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.gold_map[i][j] > 0:
                    distance_temp = abs(self.state.x - i) + abs(self.state.y - j) #- self.gold_map[i][j]
                    if distance > distance_temp:
                        distance = distance_temp
        self.dist_gold = distance

        self.score_pre = self.state.score  # Storing the last score for designing the reward function
        self.energy_pre = self.state.energy  # Storing the last energy for designing the reward function

        self.reward = 0  # The amount of rewards for the entire episode
        self.terminate = False  # The variable indicates that the episode ends
        self.steps_beyond_done = None
        return self.gameState

    def step(self, action):     # step process
        self.socket.send(str(action))   # send action to server
        try:
            message = self.socket.receive()     # receive new state from server
            self.state.update_state(message)    # update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

        self.gameState = self.get_state()
        self.reward = self.get_reward()
        done = self.check_terminate()
        return self.gameState, self.reward, done, {}

    # Functions are customized by client
    def get_state(self):
        # Building the map
        view = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1, 2], dtype="float32")
        self.gold_map = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
                    view[i, j, 0] = -20*1.0/20
                    # view[i, j, 0] = -TreeID
                elif self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
                    view[i, j, 0] = -10*1.0/20
                    # view[i, j, 0] = -TrapID
                elif self.state.mapInfo.get_obstacle(i, j) == SwampID:    # Swamp
                    view[i, j, 0] = self.state.mapInfo.get_obstacle_value(i, j)*1.0/20
                    # view[i, j, 0] = -SwampID
                elif self.state.mapInfo.gold_amount(i, j) > 0:
                    view[i, j, 0] = self.state.mapInfo.gold_amount(i, j)*1.0/100
                    self.gold_map[i, j] = self.state.mapInfo.gold_amount(i, j)/50

        if self.state.status == 0:
            view[self.state.x, self.state.y, 1] = self.state.energy *1.0/10

        # for player in self.state.players:
        #     if player["playerId"] != self.state.id:        #         view[player["posx"], player["posy"], 1] -= 1
                
        #Convert the DQNState from list to array for training
        DQNState = np.array(view)

        return DQNState

    def get_reward(self):
        # Calculate reward
        a = self.state.lastAction
        e = self.state.energy
        e_pre = self.energy_pre
        score_action = self.state.score - self.score_pre
        energy_consume = self.energy_pre - self.state.energy
        if energy_consume == 0:
            energy_consume = 20
        elif energy_consume < 0:
            energy_consume = energy_consume * 2
        self.score_pre = self.state.score
        self.energy_pre = self.state.energy

        # calculate distance to gold
        distance = 500
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.gold_map[i][j] > 0:
                    distance_temp = abs(self.state.x - i) + abs(self.state.y - j)   # - self.gold_map[i][j]
                    if distance > distance_temp:
                        distance = distance_temp
        move_distance = (self.dist_gold - distance)
        if move_distance > 0:
            score_distance = move_distance * 20
        elif move_distance < 0:
            score_distance = move_distance * 10
        else:
            score_distance = 0
        self.dist_gold = distance
        reward = score_action * 2 + score_distance - energy_consume

        # If out of the map, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward += -50
            
        # Run out of energy, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward += -50
        # print ("reward",reward)
        return reward * 0.01

    def check_terminate(self):
        # Checking the status of the game
        # it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING

    def updateObservation(self):
        return

    def render(self, mode='human', close=False):
       return

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        raise NotImplementedError()

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()
