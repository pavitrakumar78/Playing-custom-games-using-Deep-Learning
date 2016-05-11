from emulateShooter import Shooter
import pygame
from memory_store import *
from agent import Agent
import deep_q_learn
import datetime
import cPickle
from statistics import Statistics
import sys
import argparse

#had to increase depth - was getting max. rer depth reached by cpickle
sys.setrecursionlimit(10000) 


screen_width = 84
screen_height = 84
default_random_steps = 2500

parser = argparse.ArgumentParser(description = "EXPLANATION HERE...")

#value "store_true". This means that, if the option is specified, assign the value True to args.verbose. Not specifying it implies False.
parser.add_argument("--display_screen", default=False, action = "store_true", help="Display game screen?")
parser.add_argument("--play_games", type = int, default=0, help="How many games to play from loaded network(requires a load_weights param also)")
parser.add_argument("--train_model", default = False,  action = "store_true")
parser.add_argument("--load_weights", type = str, help = "Load pre-trained network.")
parser.add_argument("--save_models", type = bool, default = False, help = "To save the trained networks for later use?")
parser.add_argument("--save_model_dir",type = str, default = "RecentlyTrainedModels", help = "Name of folder to save the trained networks.")

args = parser.parse_args()

#print args
#sys.exit(1)

env = Shooter(play = args.play_games)

mem = ReplayMemory()

"""
TODO - Try increasing the controls to all directions and increase the difficulty! 
- a more complicated example!
"""
# ----------------------
# Experiment Parameters
# ----------------------
STEPS_PER_EPOCH = 50000
EPOCHS = 100  #can try to increase but we get good performace for 70 itself! 
STEPS_PER_TEST = 10000
# ----------------------
# ALE Parameters
# ----------------------
#BASE_ROM_PATH = "../roms/"
#ROM = 'breakout.bin'
FRAME_SKIP = 4
REPEAT_ACTION_PROBABILITY = 0
# ----------------------
# Agent/Network parameters:
# ----------------------
UPDATE_RULE = 'sgd' #also try "rmsprop"
BATCH_ACCUMULATOR = 'mean'
LEARNING_RATE = .00002 #also try adjusting this! 0.00002 best!
DISCOUNT = .95
RMS_DECAY = .9 # (Rho)
RMS_EPSILON = 1e-6
MOMENTUM = 0.95
CLIP_DELTA = 0
EPSILON_START = 1.0
EPSILON_MIN = .1
EPSILON_DECAY = 1000000
PHI_LENGTH = 4
UPDATE_FREQUENCY = 1
REPLAY_MEMORY_SIZE = 1000000
BATCH_SIZE = 32
NETWORK_TYPE = "nips_cuda" #also try nips_cuda,nips_dnn #NIPS CUDA BEST!
FREEZE_INTERVAL = -1
REPLAY_START_SIZE = 100
RESIZE_METHOD = 'crop'
RESIZED_WIDTH = 84
RESIZED_HEIGHT = 84
DEATH_ENDS_EPISODE = 'false'
MAX_START_NULLOPS = 0
DETERMINISTIC = True
CUDNN_DETERMINISTIC = False

if args.train_model:
        network = deep_q_learn.DeepQLearner(RESIZED_WIDTH,
                                            RESIZED_HEIGHT,
                                            env.numActions(),
                                            PHI_LENGTH,
                                            DISCOUNT,
                                            LEARNING_RATE,
                                            RMS_DECAY,
                                            RMS_EPSILON,
                                            MOMENTUM,
                                            CLIP_DELTA,
                                            FREEZE_INTERVAL,
                                            BATCH_SIZE,
                                            NETWORK_TYPE,
                                            UPDATE_RULE,
                                            BATCH_ACCUMULATOR,
                                            np.random.RandomState(123456))
else:
    if args.load_weights == None:
        print "Pre-Trained network not found!"
        sys.exit(1)
    else:
        network = cPickle.load(open(args.load_weights, 'r'))
        if network == None: #change to try-catch later
            print "Loading netowork failed!"
        print "Network loaded successfully!"


agent = Agent(env, mem, network)


if args.train_model:
    #stats = Statistics(agent, network, mem, env)

    agent.play_random(random_steps=default_random_steps)

    print "Traning Started....."

    for i in range(EPOCHS):
        #stats.reset()
        a = datetime.datetime.now().replace(microsecond=0)
        agent.train(train_steps = STEPS_PER_EPOCH,epoch = 1)
        agent.test(test_steps = STEPS_PER_TEST,epoch = 1)
        save_path = args.save_model_dir
        if args.save_models:
            path_file = args.save_model_dir+'/dep-q-shooter-nipscuda-8movectrl-'+str(i)+'-epoch.pkl'
            #print path_file
            net_file = open(path_file, 'w')
            cPickle.dump(network, net_file, -1)
            net_file.close()
        b = datetime.datetime.now().replace(microsecond=0)
        #stats.write(i + 1, "train")
        print "Completed "+str(i+1)+"/"+str(EPOCHS)+" epochs in ",(b-a)

    print "Training Ended....."

if args.play_games > 0:
    agent.play(args.play_games)