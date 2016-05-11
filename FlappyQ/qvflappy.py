from itertools import cycle
import random
import sys
import math
from PIL import Image
import scipy
import cv2
from sklearn.preprocessing import normalize
import copy
import pygame
from pygame.locals import *
import time
import logging
import datetime

import matplotlib.pyplot as plt

import numpy

class FlappyBird():


    def __init__(self,alpha,resolution_scale,exploration_rate,is_play=False):
        self.avg_score = 0
        self.Q_reward = 0
        self.times_dead = 0
        self.is_play = is_play
        if not self.is_play:
            score_record = open("scores.txt","w")
            score_record.write("Scores:\n")
            score_record.close()
        self.m_state={"vertical_distance": 0, "horizontal_distance": 0}
        self.m_state_dash= {"vertical_distance": 0, "horizontal_distance": 0}
        self.explore= exploration_rate
        self.action_to_perform= "do_nothing"
        self.resolution= resolution_scale
        self.alpha_QL= alpha
        self.vertical_dist_range= [0, 512]
        self.horizontal_dist_range= [0, 288]
        self.Q = [0]*1000
        for i in range(0,(512+0)/self.resolution):
            self.Q[i] = [0]*1000
            for j in range(0,288/self.resolution):
                self.Q[i][j] = {"click": 0, "do_nothing": 0}
        self.ALL_Q_STUFF = [self.m_state,self.m_state_dash,self.Q]
        self.reward = 0
        self.FRAME_SKIP = -4
        self.counter = self.FRAME_SKIP
        if self.is_play:
            self.FPS = 600
        else:
            self.FPS = 1500  #set to >1000 for traning
        self.SCREENWIDTH  = 288
        self.SCREENHEIGHT = 512
        # amount by which base can maximum shift to left
        self.PIPEGAPSIZE  = 200 # gap between upper and lower part of pipe
        self.BASEY        = self.SCREENHEIGHT * 0.79
        # image, sound and hitmask  dicts
        self.IMAGES, self.SOUNDS, self.HITMASKS = {}, {}, {}
        self.sensors = [0,0,0]
        # list of all possible players (tuple of 3 positions of flap)
        self.PLAYERS_LIST = (
            # red bird
            (
                'assets/sprites/redbird-upflap.png',
                'assets/sprites/redbird-midflap.png',
                'assets/sprites/redbird-downflap.png',
            ),
            # blue bird
            (
                'assets/sprites/redbird-upflap.png',
                'assets/sprites/redbird-midflap.png',
                'assets/sprites/redbird-downflap.png',
            ),
            # yellow bird
            (
                'assets/sprites/redbird-upflap.png',
                'assets/sprites/redbird-midflap.png',
                'assets/sprites/redbird-downflap.png',
            ),
        )
        # list of backgrounds
        self.BACKGROUNDS_LIST = (
            'assets/sprites/background-day.png',
            'assets/sprites/background-night.png',
        )

        # list of pipes
        self.PIPES_LIST = (
            'assets/sprites/pipe-green.png',
            'assets/sprites/pipe-red.png',
        )
        self.isDead = False

        pygame.init()
        self.FPSCLOCK = pygame.time.Clock()
        self.SCREEN = pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT))
        self.GAME_IMAGE = pygame.Surface((self.SCREENWIDTH, self.SCREENHEIGHT))
        pygame.display.set_caption('Flappy Bird')
        # numbers sprites for score display
        self.IMAGES['numbers'] = (
            pygame.image.load('assets/sprites/0.png').convert_alpha(),
            pygame.image.load('assets/sprites/1.png').convert_alpha(),
            pygame.image.load('assets/sprites/2.png').convert_alpha(),
            pygame.image.load('assets/sprites/3.png').convert_alpha(),
            pygame.image.load('assets/sprites/4.png').convert_alpha(),
            pygame.image.load('assets/sprites/5.png').convert_alpha(),
            pygame.image.load('assets/sprites/6.png').convert_alpha(),
            pygame.image.load('assets/sprites/7.png').convert_alpha(),
            pygame.image.load('assets/sprites/8.png').convert_alpha(),
            pygame.image.load('assets/sprites/9.png').convert_alpha()
        )
        # game over sprite
        self.IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
        # message sprite for welcome screen
        self.IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
        # base (ground) sprite
        self.IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.MI_playerIndexGen = 0
        self.MI_playery = 0
        self.MI_basex = 0

        self.IMAGES['background'] = pygame.image.load(self.BACKGROUNDS_LIST[0]).convert()
        # select random player sprites
        #randPlayer = random.randint(0, len(self.PLAYERS_LIST) - 1)
        self.IMAGES['player'] = (
            pygame.image.load(self.PLAYERS_LIST[0][0]).convert_alpha(),
            pygame.image.load(self.PLAYERS_LIST[0][1]).convert_alpha(),
            pygame.image.load(self.PLAYERS_LIST[0][2]).convert_alpha(),
        )
        # select random pipe sprites
        #pipeindex = random.randint(0, len(self.PIPES_LIST) - 1)
        self.IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(self.PIPES_LIST[0]).convert_alpha(), 180),
            pygame.image.load(self.PIPES_LIST[0]).convert_alpha(),
        )
        # hismask for pipes
        self.HITMASKS['pipe'] = (
            self.getHitmask(self.IMAGES['pipe'][0]),
            self.getHitmask(self.IMAGES['pipe'][1]),
        )
        # hitmask for player
        self.HITMASKS['player'] = (
            self.getHitmask(self.IMAGES['player'][0]),
            self.getHitmask(self.IMAGES['player'][1]),
            self.getHitmask(self.IMAGES['player'][2]),
        )
        movementInfo = self.showWelcomeAnimation()

        self.MI_playerIndexGen = movementInfo['playerIndexGen']
        self.MI_playery = movementInfo['playery']
        self.MI_basex = movementInfo['basex']
        """game globals"""
        self.score = self.playerIndex = self.loopIter = 0
        self.playerIndexGen = self.MI_playerIndexGen
        self.playerx, self.playery = int(self.SCREENWIDTH * 0.2), self.MI_playery
        self.basex = self.MI_basex
        self.baseShift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()
        # get 2 new pipes to add to upperPipes lowerPipes list
        self.newPipe1 = self.getRandomPipe()
        self.newPipe2 = self.getRandomPipe()
        # list of upper pipes
        self.upperPipes = [
            {'x': self.SCREENWIDTH - 30, 'y': self.newPipe1[0]['y']},
            {'x': self.SCREENWIDTH - 30 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[0]['y']},
        ]
        # list of lowerpipe
        self.lowerPipes = [
            {'x': self.SCREENWIDTH - 30, 'y': self.newPipe1[1]['y']},
            {'x': self.SCREENWIDTH - 30 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[1]['y']},
        ]
        self.pipeVelX = -4
        # player velocity, max velocity, downward accleration, accleration on flap
        self.playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1   # players downward accleration
        self.playerFlapAcc =  -9   # players speed on flapping
        self.playerFlapped = False # True when player flaps
        self.counter = self.FRAME_SKIP   


    def numActions(self):
        return 2

    def getActions(self):
        return [1,0]


    def rgb_int2tuple_FULL(self,rgbint):
        return (rgbint // 256 // 256 % 256, rgbint // 256 % 256, rgbint % 256)

    def rgb_int2grey(self,rgbint):
        return (rgbint // 256 // 256 % 256)*0.298 + (rgbint // 256 % 256)*0.587 + (rgbint % 256)*0.114

    def getScreen(self):
        #scale and crop vs crop and scale? - do whichever has less steps?.   
        #1
        cropped = pygame.Surface((288,405))
        cropped.blit(self.GAME_IMAGE,(0,0))
        cropped = pygame.transform.scale(cropped , (84,84))

        a = numpy.array(pygame.surfarray.array2d(cropped),dtype='float32')  #original->crop->scale->to array

        a *= 1/a.max()

        return a

    def isTerminal(self):
        return self.isDead

    def restart(self):

        #a = datetime.datetime.now().replace(microsecond=0)
        #print_stats = "Time: "+str(a)+"       Total Score: "+str(int(round(self.score)))
        #print print_stats
        #log = open("log_stuff.txt","a")
        #log.write(print_stats+"\n")
        #log.close()


        self.reward = 0  #try setting it as -100 for q learning in this file itself - NO, not tHIS@
        self.isDead = False
        self.score = self.playerIndex = self.loopIter = 0
        self.playerIndexGen = self.MI_playerIndexGen
        self.playerx, self.playery = int(self.SCREENWIDTH * 0.2), self.MI_playery

        self.basex = self.MI_basex
        self.baseShift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # get 2 new pipes to add to upperPipes lowerPipes list
        self.newPipe1 = self.getRandomPipe()
        self.newPipe2 = self.getRandomPipe()

        # list of upper pipes
        self.upperPipes = [
            {'x': self.SCREENWIDTH - 30, 'y': self.newPipe1[0]['y']},
            {'x': self.SCREENWIDTH - 30 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[0]['y']},
        ]

        # list of lowerpipe
        self.lowerPipes = [
            {'x': self.SCREENWIDTH - 30, 'y': self.newPipe1[1]['y']},
            {'x': self.SCREENWIDTH - 30 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[1]['y']},
        ]

        self.pipeVelX = -4

        # player velocity, max velocity, downward accleration, accleration on flap
        self.playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1   # players downward accleration
        self.playerFlapAcc =  -9   # players speed on flapping
        self.playerFlapped = False # True when player flaps

        self.counter = self.FRAME_SKIP

        movementInfo = self.showWelcomeAnimation()

        self.MI_playerIndexGen = movementInfo['playerIndexGen']
        self.MI_playery = movementInfo['playery']
        self.MI_basex = movementInfo['basex']

    def showWelcomeAnimation(self):
        #print self.BACKGROUNDS_LIST
        #randBg = random.randint(0, len(self.BACKGROUNDS_LIST) - 1)

        """Shows welcome screen animation of flappy bird"""
        # index of player to blit on screen
        playerIndex = 0
        playerIndexGen = cycle([0, 1, 2, 1])
        # iterator used to change playerIndex after every 5th iteration
        loopIter = 0

        playerx = int(self.SCREENWIDTH * 0.2)
        playery = int((self.SCREENHEIGHT - self.IMAGES['player'][0].get_height()) / 2)

        messagex = int((self.SCREENWIDTH - self.IMAGES['message'].get_width()) / 2)
        messagey = int(self.SCREENHEIGHT * 0.12)

        basex = 0
        # amount by which base can maximum shift to left
        baseShift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # player shm for up-down motion on welcome screen
        playerShmVals = {'val': 0, 'dir': 1}

        return {
                        'playery': playery + playerShmVals['val'],
                        'basex': basex,
                        'playerIndexGen': playerIndexGen,
                    }

    """
    def init_game_globals(self):

        score = playerIndex = loopIter = 0
        playerIndexGen = self.MI_playerIndexGen
        playerx, playery = int(self.SCREENWIDTH * 0.2), self.MI_playery

        basex = self.MI_basex
        baseShift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # get 2 new pipes to add to upperPipes lowerPipes list
        self.newPipe1 = self.getRandomPipe()
        self.newPipe2 = self.getRandomPipe()

        # list of upper pipes
        upperPipes = [
            {'x': self.SCREENWIDTH + 25, 'y': self.newPipe1[0]['y']},
            {'x': self.SCREENWIDTH + 25 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[0]['y']},
        ]

        # list of lowerpipe
        lowerPipes = [
            {'x': self.SCREENWIDTH + 25, 'y': self.newPipe1[1]['y']},
            {'x': self.SCREENWIDTH + 25 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[1]['y']},
        ]

        pipeVelX = -4

        # player velocity, max velocity, downward accleration, accleration on flap
        playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
        playerMaxVelY =  10   # max vel along Y, max descend speed
        playerMinVelY =  -8   # min vel along Y, max ascend speed
        playerAccY    =   1   # players downward accleration
        playerFlapAcc =  -9   # players speed on flapping
        playerFlapped = False # True when player flaps

        self.counter = self.FRAME_SKIP       
    """
    """
    Source:
    https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/
    http://mnemstudio.org/path-finding-q-learning-tutorial.htm
    http://cognitrn.psych.indiana.edu/CogsciSoftware/Robotics/images/tutorial.doc
    http://artint.info/html/ArtInt_265.html
    """
    """
    Basic Q learning representation:

    <s0,a0,r1,s1,a1,r2,s2,a2,r3,s3,a3,r4,s4...>,
    which means that the agent was in state s0 and did action a0, which resulted in it receiving reward r1 and being in state s1; 
    then it did action a1, received reward r2, and ended up in state s2; 
    then it did action a2, received reward r3, and ended up in state s3; and so on.
    """
    def calculate_Q(self):
        horizontal_distance = 9999
        vertical_distance = 9999

        horizontal_distance = self.sensors[0]
        vertical_distance = self.sensors[1]

        #print "H->",horizontal_distance
        #print "V->",vertical_distance

        #m' - current coordinates  (S')
        #m - prev. coordinates (S)
        self.m_state_dash["vertical_distance"] = vertical_distance
        self.m_state_dash["horizontal_distance"] = horizontal_distance



        #Step 3: Update Q(S, A)
        #our Q table is basically made up of possible states and actions
        #here, the 3rd dimension is the actions and the 2d arrays are the possible states - i.e the vertical and horizontal distances from the bird to the pipe
        
        #below  - why use max? - (x,0) - we dont care about negative values (negative means bird is below pipe)
        #increasing the resolution decreases the possible states - it is basically like grouping 4 pixels into 1 
        #i.e simplifying data by sacrificing accuracy of data
        
        #S
        state_bin_v = max( 
                    min ( 
                        math.floor((self.vertical_dist_range[1]-self.vertical_dist_range[0]-1)/self.resolution), 
                        math.floor( (self.m_state["vertical_distance"] - self.vertical_dist_range[0])/self.resolution )
                    ), 
                    0
                )
                
        state_bin_h = max( 
                    min ( 
                        math.floor((self.horizontal_dist_range[1]-self.horizontal_dist_range[0]-1)/self.resolution), 
                        math.floor( (self.m_state["horizontal_distance"] - self.horizontal_dist_range[0])/self.resolution )
                    ), 
                    0
                )

        #S'
        state_dash_bin_v = max( 
                    min ( 
                        math.floor((self.vertical_dist_range[1]-self.vertical_dist_range[0]-1)/self.resolution), 
                        math.floor( (self.m_state_dash["vertical_distance"] - self.vertical_dist_range[0])/self.resolution )
                    ), 
                    0
                )
        #print state_dash_bin_v  
        state_dash_bin_h = max( 
                    min ( 
                        math.floor((self.horizontal_dist_range[1]-self.horizontal_dist_range[0]-1)/self.resolution), 
                        math.floor( (self.m_state_dash["horizontal_distance"] - self.horizontal_dist_range[0])/self.resolution )
                    ), 
                    0
                )
        state_dash_bin_h = int(state_dash_bin_h)
        state_dash_bin_v = int(state_dash_bin_v)
        state_bin_h = int(state_bin_h)
        state_bin_v = int(state_bin_v)
        
        #print "PRE :::"
        #print state_dash_bin_h
        #print state_dash_bin_v
        #print state_bin_h
        #print state_bin_v


        #getting the Q values of the current state (current h and v distance)
        click_v = self.Q[state_dash_bin_v][state_dash_bin_h]["click"];
        do_nothing_v = self.Q[state_dash_bin_v][state_dash_bin_h]["do_nothing"]
        # V(s',a') - value/reward of doing action a' in state s' (poststate - future)
        V_s_dash_a_dash = max(click_v, do_nothing_v) #this is the max(Q(s',a'))
        #getting the current state using the current action performed
        Q_s_a = self.Q[state_bin_v][state_bin_h][self.action_to_perform]
        #updating the current(S) using the future(S') values.
        # Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        #self.Q_reward + (0.99~1(aprox))V_s_dash_a_dash = this is the actual current reward plus the discounted estimated future value
        #here discount is treated as 1
        self.Q[state_bin_v][state_bin_h][self.action_to_perform] = Q_s_a + self.alpha_QL * (self.Q_reward + V_s_dash_a_dash - Q_s_a)

        #print Q_s_a

        #Step 4: S <- S'
        #S is prestate and S' is the poststate
        self.m_state = copy.deepcopy(self.m_state_dash)

        #Step 1: Select and perform Action A
        if (random.random() < self.explore):
                if (random.randint(1,10)%2) == 0:
                    self.action_to_perform = "click"
                else:
                    self.action_to_perform = "do_nothing"
                #print "RANDOM ACTION  ",self.action_to_perform
                    #self.action_to_perform = ((random.randint(1,2)-1) == 0)? "click" : "do_nothing"
        else:
            """below is almost same as above, just computing  Q(s,a) - try to put it inside a function """
            state_bin_v = max( 
                        min ( 
                            math.floor((self.vertical_dist_range[1]-self.vertical_dist_range[0]-1)/self.resolution), 
                            math.floor( (self.m_state["vertical_distance"] - self.vertical_dist_range[0])/self.resolution )
                        ), 
                        0
                    )
                    
            state_bin_h = max( 
                        min ( 
                            math.floor((self.horizontal_dist_range[1]-self.horizontal_dist_range[0]-1)/self.resolution), 
                            math.floor( (self.m_state["horizontal_distance"] - self.horizontal_dist_range[0])/self.resolution )
                        ), 
                        0
                    )
            state_dash_bin_h = int(state_dash_bin_h)
            state_dash_bin_v = int(state_dash_bin_v)
            state_bin_h = int(state_bin_h)
            state_bin_v = int(state_bin_v)

            #print "POST :::"
            #print state_dash_bin_h
            #print state_dash_bin_v
            #print state_bin_h
            #print state_bin_v
        
            click_v = self.Q[state_bin_v][state_bin_h]["click"]
            do_nothing_v = self.Q[state_bin_v][state_bin_h]["do_nothing"]
            if click_v > do_nothing_v:
                #print "Q - DO CLICK"
                self.action_to_perform = "click"
            else:
                #print "Q - DO NOTHING"
                self.action_to_perform = "do_nothing"  

            #print "PREDICTED ACTION  ",self.action_to_perform
            #self.action_to_perform = click_v > do_nothing_v ? "click" : "do_nothing"
            

            #console.log("action performed: " + self.action_to_perform);
        self.ALL_Q_STUFF = [self.m_state,self.m_state_dash,self.Q]
        if (self.action_to_perform == "click"):
            self.act(1) #jump!
        else:
            self.act(0)

    def showScore(self,score):
        """displays score in center of screen"""
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0 # total width of all numbers to be printed
        for digit in scoreDigits:
            totalWidth += self.IMAGES['numbers'][digit].get_width()
        Xoffset = (self.SCREENWIDTH - totalWidth) / 2
        for digit in scoreDigits:
            #SCREEN.blit(self.IMAGES['numbers'][digit], (Xoffset, self.SCREENHEIGHT * 0.1))
            self.GAME_IMAGE.blit(self.IMAGES['numbers'][digit], (Xoffset, self.SCREENHEIGHT * 0.1))
            Xoffset += self.IMAGES['numbers'][digit].get_width()

    def act(self,action):
        #print "ENTERED"
        for i in range(4): #frame_skip
            #print "loop 4 frames"
            if action == 1 and i == 0:  #do only for first frame!
                #print "flying..."
                if self.playery > -2 * self.IMAGES['player'][0].get_height():
                    self.playerVelY = self.playerFlapAcc
                    self.playerFlapped = True
            #if action is 0, nothhing happens the bird just follows gravity


            # check for crash here
            crashTest = self.checkCrash({'x': self.playerx, 'y': self.playery, 'index': self.playerIndex},
                                   self.upperPipes, self.lowerPipes)
            if crashTest[0]:
                self.isDead = True
                self.times_dead += 1
                self.avg_score += self.score
                #print self.times_dead,self.score
                score_record = open("new_scores.txt","a")
                a = datetime.datetime.now().replace(microsecond=0)
                score_record.write("Time: "+str(a)+"    Score: "+str(self.score)+"\n")
                if not self.is_play:
                    if (self.times_dead%1000) == 0:
                        res = "---------------------AVERAGE SCORE: "+str(int(self.avg_score/self.times_dead))+"-----after "+str(self.times_dead)+" deaths--------------------------\n"
                        score_record.write(res)
                        score_record.close()
                self.Q_reward = -1000
                self.restart()
                #print "DEAD------------------"
                return {
                    'y': self.playery,
                    'groundCrash': crashTest[1],
                    'basex': self.basex,
                    'upperPipes': self.upperPipes,
                    'lowerPipes': self.lowerPipes,
                    'score': self.score,
                    'playerVelY': self.playerVelY,
                }
            self.reward += 0.1
            self.Q_reward = 1
            # check for score
            playerMidPos = self.playerx + self.IMAGES['player'][0].get_width() / 2
            for pipe in self.upperPipes:
                pipeMidPos = pipe['x'] + self.IMAGES['pipe'][0].get_width() / 2
                if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                    self.reward += 1
                    self.score += 1
                    #SOUNDS['point'].play()

            # playerIndex basex change
            if (self.loopIter + 1) % 3 == 0:
                self.playerIndex = self.playerIndexGen.next()
            self.loopIter = (self.loopIter + 1) % 30
            self.basex = -((-self.basex + 100) % self.baseShift)

            # player's movement
            if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
                self.playerVelY += self.playerAccY
            if self.playerFlapped:
                #print "stopped flapping"
                self.playerFlapped = False
            self.playerHeight = self.IMAGES['player'][self.playerIndex].get_height()
            self.playery += min(self.playerVelY, self.BASEY - self.playery - self.playerHeight)

            # move pipes to left
            for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
                uPipe['x'] += self.pipeVelX
                lPipe['x'] += self.pipeVelX

            # add new pipe when first pipe is about to touch left of screen
            if 0 < self.upperPipes[0]['x'] < 5:
                newPipe = self.getRandomPipe()
                self.upperPipes.append(newPipe[0])
                self.lowerPipes.append(newPipe[1])

            # remove first pipe if its out of the screen
            if self.upperPipes[0]['x'] < -self.IMAGES['pipe'][0].get_width():
                self.upperPipes.pop(0)
                self.lowerPipes.pop(0)

            # draw sprites
            #SCREEN.blit(IMAGES['background'], (0,0))

            #calculate distance between the bird and the next pipe's top and bottom edge

            """

            DONT USE UPPER PIPE!!!!  (execute blittest.py to see why)

            print "*"*50
            print "X-difference between bird and bottom left edge of next 1st TOPpipe: ",(self.upperPipes[0]['x']-self.playerx)
            print "X-difference between bird and bottom left edge of next 2nd TOPpipe: ",(self.upperPipes[1]['x']-self.playerx)
            
            print "X-difference between bird and top left edge of next 1st BOTTOMpipe: ",(self.lowerPipes[0]['x']-self.playerx)
            print "X-difference between bird and top left edge of next 2nd BOTTOMpipe: ",(self.lowerPipes[1]['x']-self.playerx)

            print "Y-difference between bird and bottom of the 1st TOPpipe: ",(self.playery-self.upperPipes[0]['y'])
            print "Y-difference between bird and bottom of the 2nd TOPpipe: ",(self.playery-self.upperPipes[1]['y'])
            
            print "Y-difference between bird and top of the 1st BOTTOMpipe: ",(self.lowerPipes[0]['y']-self.playery)
            print "Y-difference between bird and top of the 2nd BOTTOMpipe: ",(self.lowerPipes[1]['y']-self.playery)
            print "*"*50
            """
            alive = 1
            if self.isTerminal():
                #print "DEAD"
                alive = 0

            if self.lowerPipes[0]['x'] > self.playerx:
                self.sensors = [self.lowerPipes[0]['x']-(self.playerx+34),self.lowerPipes[0]['y']-(self.playery+24),alive]
                #print "first pipe data"
            else:
                self.sensors = [self.lowerPipes[1]['x']-(self.playerx+34),self.lowerPipes[1]['y']-(self.playery+24),alive]
                #print "2nd pipe data"

            dist = math.sqrt(math.pow(self.sensors[0],2)+math.pow(self.sensors[1],2))
            #print dist
            #print self.sensors
           # myfont = pygame.font.SysFont("ariel", 15)

            # render text
            #label = myfont.render("hello", 10, (255,0,0))

            self.GAME_IMAGE.blit(self.IMAGES['background'], (0,0))

            for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
                #SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                #SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

                self.GAME_IMAGE.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                self.GAME_IMAGE.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
                #self.GAME_IMAGE.blit(label, (uPipe['x'], uPipe['y']))


            #SCREEN.blit(IMAGES['base'], (basex, BASEY))
            self.GAME_IMAGE.blit(self.IMAGES['base'], (self.basex, self.BASEY))
            # print score so player overlaps the score
            #showScore(score)

            #SCREEN.blit(IMAGES['player'][playerIndex], (playerx, playery))

            self.GAME_IMAGE.blit(self.IMAGES['player'][self.playerIndex], (self.playerx, self.playery))
            #print "--"*50
            #self.GAME_IMAGE.blit(label, (uPipe['x'], uPipe['y']))
            #print (uPipe['x'], uPipe['y'])

            if self.is_play:
                self.showScore(self.score)
                self.SCREEN.blit(self.GAME_IMAGE,(0,0))   #uncomment to display game!
                pygame.display.update()
            
            self.FPSCLOCK.tick(self.FPS)
            
            #MI_playery = playery
            #MI_playerIndexGen = playerIndexGen
            #MI_basex = basex
        return self.reward


    def load_Q_params(self,q_arr):
        self.ALL_Q_STUFF = q_arr
        #[self.m_state,self.m_state_dash,self.Q]
        self.m_state = self.ALL_Q_STUFF[0]
        self.m_state_dash = self.ALL_Q_STUFF[1]
        self.Q = self.ALL_Q_STUFF[2]

    #def showGameOverScreen(self,crashInfo):
    #    self.isTerminal = True
    #    return
    def getSensors(self):
        return self.sensors[0],self.sensors[1],self.sensors[2]

    def playerShm(self,playerShm):
        """oscillates the value of playerShm['val'] between 8 and -8"""
        if abs(playerShm['val']) == 8:
            playerShm['dir'] *= -1

        if playerShm['dir'] == 1:
             playerShm['val'] += 1
        else:
            playerShm['val'] -= 1


    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapY = random.randrange(0, int(self.BASEY * 0.6 - self.PIPEGAPSIZE))
        gapY += int(self.BASEY * 0.2)
        pipeHeight = self.IMAGES['pipe'][0].get_height()
        pipeX = self.SCREENWIDTH + 10

        return [
            {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
            {'x': pipeX, 'y': gapY + self.PIPEGAPSIZE}, # lower pipe
        ]


    def checkCrash(self,player, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes."""
        pi = player['index']
        player['w'] = self.IMAGES['player'][0].get_width()
        player['h'] = self.IMAGES['player'][0].get_height()

        # if player crashes into ground
        if (player['y'] + player['h'] >= self.BASEY - 1) or (player['y'] - 10 < 0):
            return [True, True]
        else:

            playerRect = pygame.Rect(player['x'], player['y'],
                          player['w'], player['h'])
            pipeW = self.IMAGES['pipe'][0].get_width()
            pipeH = self.IMAGES['pipe'][0].get_height()

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

                # player and upper/lower pipe hitmasks
                pHitMask = self.HITMASKS['player'][pi]
                uHitmask = self.HITMASKS['pipe'][0]
                lHitmask = self.HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self.pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return [True, False]

        return [False, False]

    def pixelCollision(self,rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in xrange(rect.width):
            for y in xrange(rect.height):
                if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                    return True
        return False

    def getHitmask(self,image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x,y))[3]))
        return mask
    BACKGROUNDS_LIST = (
            'assets/sprites/background-day.png',
            'assets/sprites/background-night.png',
    )
"""
def main():
    #game()
    flappy = FlappyBird(is_play = False,alpha = 0.5,resolution_scale = 2,exploration_rate=0.9)
    action = random.randint(0,2)
    flappy.act(1)
    i = 0
    while True:
        i+=1
        #action = random.randint(0,2)
        #flappy.act(0)
        flappy.calculate_Q()
        if flappy.isTerminal():
            score_record = open("scores.txt","a")
            score_record.write("Step: "+str(i)+" Score: "+str(flappy.reward)+"\n")
            score_record.close()
            #flappy.restart()
            

if __name__ == "__main__":
   main()
"""