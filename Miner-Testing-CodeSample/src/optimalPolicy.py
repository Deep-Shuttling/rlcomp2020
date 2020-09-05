from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import sys
import numpy as np
from GAME_SOCKET_DUMMY import GameSocket  # in testing version, please use GameSocket instead of GameSocketDummy
from MINER_STATE import State
import copy

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
        self.decay = 27
        self.area_affect = 3
        self.affect_eff = 0.92
        self.view = None
        self.energy_view = None
        self.current_action = None
        self.gold_map = None
        self.gold_map_origin = None
        
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
        # posID_x = 12
        # X-axes randomly
        posID_y = np.random.randint(MAP_MAX_Y)  # Choosing a initial position of the DQN agent on Y-axes randomly
        # posID_y = 1
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
        self.view = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        self.energy_view = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        self.gold_map = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        gold_opt = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
                    self.view[i, j] = -20
                    self.energy_view[i, j] = -20
                elif self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
                    self.view[i, j] = -10
                    self.energy_view[i, j] = -10
                elif self.state.mapInfo.get_obstacle(i, j) == SwampID:  # Swamp
                    self.view[i, j] = self.state.mapInfo.get_obstacle_value(i, j)
                    self.energy_view[i, j] = self.state.mapInfo.get_obstacle_value(i, j)
                elif self.state.mapInfo.gold_amount(i, j) > 0:
                    self.view[i, j] = self.state.mapInfo.gold_amount(i, j)
                    self.energy_view[i, j] = -4
                    self.gold_map[i, j] = self.state.mapInfo.gold_amount(i, j)
                else:
                    self.view[i, j] = -1
                    self.energy_view[i, j] = -1
        self.gold_map_origin = copy.deepcopy(self.gold_map)
        # print(self.gold_map)
        # player update goldmap
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                x = player["posx"]
                y = player["posy"]
                if 0 <= x <= self.state.mapInfo.max_x and 0 <= y <= self.state.mapInfo.max_y:
                    if self.gold_map[x][y] > 0:
                        if x != self.state.x or y != self.state.y:
                            self.gold_map[x][y] = self.gold_map[x][y] * 0.63
                            self.view[x][y] = self.gold_map[x][y]
                    else:
                        for t in range(1, self.area_affect + 1):
                            for k in range(-t, t):
                                if 0 <= x + k <= self.state.mapInfo.max_x and 0 <= y + t - abs(
                                        k) <= self.state.mapInfo.max_y:
                                    if self.gold_map[x + k][y + t - abs(k)] > 0:
                                        self.gold_map[x + k][y + t - abs(k)] = self.gold_map[x + k][
                                                                                   y + t - abs(k)] * pow(
                                            self.affect_eff, self.area_affect + 1 - t)
                                        self.view[x + k][y + t - abs(k)] = self.gold_map[x + k][y + t - abs(k)]
                                if 0 <= x - k <= self.state.mapInfo.max_x and 0 <= y - t + abs(
                                        k) <= self.state.mapInfo.max_y:
                                    if self.gold_map[x - k][y - t + abs(k)] > 0:
                                        self.gold_map[x - k][y - t + abs(k)] = self.gold_map[x - k][
                                                                                   y - t + abs(k)] * pow(
                                            self.affect_eff, self.area_affect + 1 - t)
                                        self.view[x - k][y - t + abs(k)] = self.gold_map[x - k][y - t + abs(k)]
        print(self.gold_map)
        arr = []
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    gold_est = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
                    gold_est[i][j] = self.gold_map[i][j]
                    for a in range(0, i):
                        gold_est[i - a - 1][j] = max(gold_est[i - a][j] - self.decay + self.view[i - a - 1][j], 0)
                    for b in range(i + 1, self.state.mapInfo.max_x + 1):
                        gold_est[b][j] = max(gold_est[b - 1][j] - self.decay + self.view[b][j], 0)
                    for c in range(0, j):
                        gold_est[i][j - c - 1] = max(gold_est[i][j - c] - self.decay + self.view[i][j - c - 1], 0)
                    for d in range(j + 1, self.state.mapInfo.max_y + 1):
                        gold_est[i][d] = max(gold_est[i][d - 1] - self.decay + self.view[i][d], 0)

                    for x in range(0, i):
                        for y in range(0, j):
                            gold_est[i - x - 1][j - y - 1] = max(gold_est[i - x][j - y - 1],
                                                                 gold_est[i - x - 1][j - y]) - self.decay + \
                                                             self.view[i - x - 1][j - y - 1]
                    for x in range(0, i):
                        for y in range(j + 1, self.state.mapInfo.max_y + 1):
                            gold_est[i - x - 1][y] = max(gold_est[i - x][y], gold_est[i - x - 1][y - 1]) - self.decay + \
                                                     self.view[i - x - 1][y]
                    for x in range(i + 1, self.state.mapInfo.max_x + 1):
                        for y in range(0, j):
                            gold_est[x][j - y - 1] = max(gold_est[x][j - y], gold_est[x - 1][j - y - 1]) - self.decay + \
                                                     self.view[x][j - y - 1]
                    for x in range(i + 1, self.state.mapInfo.max_x + 1):
                        for y in range(j + 1, self.state.mapInfo.max_y + 1):
                            gold_est[x][y] = max(gold_est[x - 1][y], gold_est[x][y - 1]) - self.decay + self.view[x][y]

                    # print(i, j, self.state.mapInfo.gold_amount(i, j))
                    # print(gold_est)
                    arr.append(gold_est)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                for t in range(len(arr)):
                    if gold_opt[i][j] < arr[t][i][j]:
                        gold_opt[i][j] = arr[t][i][j]
        # print(gold_opt)
        return np.array(gold_opt)

    def check_terminate(self):
        return self.state.status != State.STATUS_PLAYING

ACTION_GO_LEFT = 0
ACTION_GO_RIGHT = 1
ACTION_GO_UP = 2
ACTION_GO_DOWN = 3
ACTION_FREE = 4
ACTION_CRAFT = 5

HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

status_map = {0: "STATUS_PLAYING", 1: "STATUS_ELIMINATED_WENT_OUT_MAP", 2: "STATUS_ELIMINATED_OUT_OF_ENERGY",
                  3: "STATUS_ELIMINATED_INVALID_ACTION", 4: "STATUS_STOP_EMPTY_GOLD", 5: "STATUS_STOP_END_STEP"}
try:
    # Initialize environment
    minerEnv = MinerEnv(HOST, PORT)
    minerEnv.start()  # Connect to the game
    minerEnv.reset()
    map_opt = minerEnv.get_state()  ##Getting an initial state
    print(map_opt)
    while not minerEnv.check_terminate():
        try:
            action = None
            if minerEnv.gold_map[minerEnv.state.x, minerEnv.state.y] > 0:
                if minerEnv.state.energy > 5:
                    action = 5
                    if minerEnv.current_action == 4:
                        # print(minerEnv.state.energy)
                        if minerEnv.state.energy <= min(33, minerEnv.gold_map[
                                                                minerEnv.state.x, minerEnv.state.y] / 10) and minerEnv.state.stepCount < 95:
                            action = 4
                else:
                    action = 4

                minerEnv.current_action = action
            else:
                around = -1000
                if minerEnv.state.x > 0 and around < map_opt[minerEnv.state.x - 1, minerEnv.state.y]:
                    around = map_opt[minerEnv.state.x - 1, minerEnv.state.y]
                    if minerEnv.state.energy + minerEnv.energy_view[minerEnv.state.x - 1, minerEnv.state.y] > 0:
                        action = 0
                    else:
                        action = 4
                if minerEnv.state.x < minerEnv.state.mapInfo.max_x and around < map_opt[
                    minerEnv.state.x + 1, minerEnv.state.y]:
                    around = map_opt[minerEnv.state.x + 1, minerEnv.state.y]
                    if minerEnv.state.energy + minerEnv.energy_view[minerEnv.state.x + 1, minerEnv.state.y] > 0:
                        action = 1
                    else:
                        action = 4
                if minerEnv.state.y > 0 and around < map_opt[minerEnv.state.x, minerEnv.state.y - 1]:
                    around = map_opt[minerEnv.state.x, minerEnv.state.y - 1]
                    if minerEnv.state.energy + minerEnv.energy_view[minerEnv.state.x, minerEnv.state.y - 1] > 0:
                        action = 2
                    else:
                        action = 4
                if minerEnv.state.y < minerEnv.state.mapInfo.max_y and around < map_opt[
                    minerEnv.state.x, minerEnv.state.y + 1]:
                    around = map_opt[minerEnv.state.x, minerEnv.state.y + 1]
                    if minerEnv.state.energy + minerEnv.energy_view[minerEnv.state.x, minerEnv.state.y + 1] > 0:
                        action = 3
                    else:
                        action = 4
                if minerEnv.current_action == 4 and minerEnv.state.energy <= 35:
                    action = 4
                # if minerEnv.state.energy < 5:
                #     action = 4
                if minerEnv.state.stepCount > 98:
                    action = 4
                minerEnv.current_action = action

            print("Step: ", minerEnv.state.stepCount, ", next action = ", action)
            minerEnv.step(str(action))  # Performing the action in order to obtain the new state
            map_opt_next = minerEnv.get_state()  # Getting a new state
            # print(minerEnv.view[minerEnv.state.x][minerEnv.state.y])
            map_opt = map_opt_next
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("Finished.")
            break
    print(status_map[minerEnv.state.status])
except Exception as e:
    import traceback
    traceback.print_exc()
print("End game.")