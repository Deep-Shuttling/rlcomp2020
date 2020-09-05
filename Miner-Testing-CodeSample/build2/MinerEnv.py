from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import sys
import numpy as np
from GAME_SOCKET_DUMMY import GameSocket  # in testing version, please use GameSocket instead of GameSocketDummy
from MINER_STATE import State

TreeID = 1
TrapID = 2
SwampID = 3
MAP_MAX_X = 21  # Width of the Map
MAP_MAX_Y = 9   # Height of the Map

class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()
        
        self.score_pre = self.state.score#Storing the last score for designing the reward function
        
    def start(self): #connect to server
        self.socket.connect()

    def end(self): #disconnect server
        self.socket.close()

    def send_map_info(self, request):#tell server which map to run
        self.socket.send(request)

    def reset(self): #start new game
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

        try:
            message = self.socket.receive() #receive game info from server
            print(message)
            self.state.init_state(message) #init state
            print(self.state.score)
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action): #step process
        self.socket.send(action) #send action to server
        try:
            message = self.socket.receive() #receive new state from server
            #print("New state: ", message)
            self.state.update_state(message) #update to local state
            print(self.state.score)
        except Exception as e:
            import traceback
            traceback.print_exc()

    # Functions are customized by client
    def get_state(self):
        # Building the map
        view = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1, 2], dtype="float32")
        self.gold_map = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
                    view[i, j, 0] = -20 * 1.0 / 20
                    # view[i, j, 0] = -TreeID
                elif self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
                    view[i, j, 0] = -10 * 1.0 / 20
                    # view[i, j, 0] = -TrapID
                elif self.state.mapInfo.get_obstacle(i, j) == SwampID:  # Swamp
                    view[i, j, 0] = self.state.mapInfo.get_obstacle_value(i, j) * 1.0 / 20
                    # view[i, j, 0] = -SwampID
                elif self.state.mapInfo.gold_amount(i, j) > 0:
                    view[i, j, 0] = self.state.mapInfo.gold_amount(i, j) * 1.0 / 100
                    self.gold_map[i, j] = self.state.mapInfo.gold_amount(i, j) / 50

        if self.state.status == 0:
            view[self.state.x, self.state.y, 1] = self.state.energy

        # for player in self.state.players:
        #     if player["playerId"] != self.state.id:
        #         view[player["posx"], player["posy"], 1] -= 1

        # Convert the DQNState from list to array for training
        DQNState = np.array(view)

        return DQNState

    def check_terminate(self):
        return self.state.status != State.STATUS_PLAYING
