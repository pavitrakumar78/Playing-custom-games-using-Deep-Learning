from qvflappy import FlappyBird
import random
import pickle
import cPickle
import time
import sys
import logging
import numpy as np

def main():
    #modify below to False if want to train
    is_play = True
    #exploration_rate determines the amount of randomness in the system
    flappy = FlappyBird(is_play = is_play,alpha = 0.5,resolution_scale = 4,exploration_rate=0)   
    if is_play:
        """
        NOTE: use dox2unix if model is trained in windows and you want to use it on linux
        usage: dos2unix *.pkl (in cmd)
        """ 
        #uncomment to load prev-trained model's data
        flappy.load_Q_params(cPickle.load(open('a5Res4Exp9q-flappy54000.pkl','r')))
    
    action = random.randint(0,2)
    flappy.act(1)
    i = 0
    while True:
        flappy.calculate_Q()
        if not is_play:
            #if not traning, save the model
            i+=1
            if (flappy.times_dead % 1000) == 0:
                print "1000 times dead"
                net_file = open('E:/py-atarti-dqn/All Models Final/FlappyQ/Saved_models_new/flappy'+str(flappy.times_dead)+'.pkl', 'w')
                pickle.dump(flappy.ALL_Q_STUFF, net_file)
                net_file.close()
    

if __name__ == "__main__":
	main()