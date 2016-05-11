from itertools import cycle
import random
import sys

from PIL import Image
import scipy
import cv2
from sklearn.preprocessing import normalize

import pygame
from pygame.locals import *

import matplotlib.pyplot as plt

import numpy

class FlappyBird():
    def __init__(self):
        self.reward = 0
        self.FRAME_SKIP = -4
        self.counter = self.FRAME_SKIP
        self.FPS = 30  #30 for playing #1200 for training
        self.SCREENWIDTH  = 288
        self.SCREENHEIGHT = 512
        # amount by which base can maximum shift to left
        self.PIPEGAPSIZE  = 170 # gap between upper and lower part of pipe #was 200 when i trained
        self.BASEY        = self.SCREENHEIGHT * 0.79
        # image, sound and hitmask  dicts
        self.IMAGES, self.SOUNDS, self.HITMASKS = {}, {}, {}
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
            {'x': self.SCREENWIDTH + 25, 'y': self.newPipe1[0]['y']},
            {'x': self.SCREENWIDTH + 25 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[0]['y']},
        ]
        # list of lowerpipe
        self.lowerPipes = [
            {'x': self.SCREENWIDTH + 25, 'y': self.newPipe1[1]['y']},
            {'x': self.SCREENWIDTH + 25 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[1]['y']},
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
        
        a = numpy.array(pygame.surfarray.array2d(cropped))  #original->crop->scale->to array
        #for row in a:
        #    print [self.rgb_int2tuple_FULL(pic) for pic in row]
        #a *= 1/a.max()
        
        #2
        img_g = Image.fromarray(a).convert('L')
        aa = numpy.array(img_g.getdata(),numpy.uint8)
        aa = aa.reshape(84,84)
        #for row in aa:
        #    print row
        #print aa.shape
        #plt.imshow(aa)
        #plt.show()
        return aa

    def isTerminal(self):
        return self.isDead

    def restart(self):
        self.reward = 0
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
            {'x': self.SCREENWIDTH + 25, 'y': self.newPipe1[0]['y']},
            {'x': self.SCREENWIDTH + 25 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[0]['y']},
        ]

        # list of lowerpipe
        self.lowerPipes = [
            {'x': self.SCREENWIDTH + 25, 'y': self.newPipe1[1]['y']},
            {'x': self.SCREENWIDTH + 25 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[1]['y']},
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

    def act(self,action):
        #print "ENTERED"
        for i in range(4): #frame_skip
            #print "loop 4 frames"
            if action == 1:
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
                return 0
                #print "DEAD------------------"
                #return {
                #    'y': self.playery,
                #    'groundCrash': crashTest[1],
                #    'basex': self.basex,
                #    'upperPipes': self.upperPipes,
                #    'lowerPipes': self.lowerPipes,
                #    'score': self.score,
                #    'playerVelY': self.playerVelY,
                #}
            self.reward += 0.1
            # check for score
            playerMidPos = self.playerx + self.IMAGES['player'][0].get_width() / 2
            for pipe in self.upperPipes:
                pipeMidPos = pipe['x'] + self.IMAGES['pipe'][0].get_width() / 2
                if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                    self.reward += 1
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

            self.GAME_IMAGE.blit(self.IMAGES['background'], (0,0))

            for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
                #SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                #SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

                self.GAME_IMAGE.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                self.GAME_IMAGE.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

            #SCREEN.blit(IMAGES['base'], (basex, BASEY))
            self.GAME_IMAGE.blit(self.IMAGES['base'], (self.basex, self.BASEY))
            # print score so player overlaps the score
            #showScore(score)

            #SCREEN.blit(IMAGES['player'][playerIndex], (playerx, playery))

            self.GAME_IMAGE.blit(self.IMAGES['player'][self.playerIndex], (self.playerx, self.playery))

            self.SCREEN.blit(self.GAME_IMAGE,(0,0))
            pygame.display.update()
            
            self.FPSCLOCK.tick(self.FPS)
            
            #MI_playery = playery
            #MI_playerIndexGen = playerIndexGen
            #MI_basex = basex
        return self.reward




    #def showGameOverScreen(self,crashInfo):
    #    self.isTerminal = True
    #    return


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