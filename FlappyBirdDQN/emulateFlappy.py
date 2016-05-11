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
	def __init__(self,play=False):
		self.reward = 0
		self.play = play
		if self.play:
			self.FPS = 2000  #30-60 for playing
		else:
			self.FPS = 1500
		self.SCREENWIDTH  = 288
		self.SCREENHEIGHT = 512
		# amount by which base can maximum shift to left
		# gap between upper and lower part of pipe #was 200 when i trained
		# this gap determines the difficulty of the game
		self.PIPEGAPSIZE  = 170 
		self.BASEY	= self.SCREENHEIGHT * 0.79
		# image, sound and hitmask  dicts
		self.IMAGES, self.SOUNDS, self.HITMASKS = {}, {}, {}
		# list of all possible players (tuple of 3 positions of flap)
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
		# base (ground) sprite
		self.IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()
		
		self.MI_playery = 0
		self.MI_basex = 0

		self.IMAGES['background'] = pygame.image.load('assets/sprites/background-day.png').convert()
		# select random player sprites
		#randPlayer = random.randint(0, len(self.PLAYERS_LIST) - 1)
		self.IMAGES['player'] = pygame.image.load('assets/sprites/redbird-midflap.png').convert_alpha()
		
		# select random pipe sprites
		#pipeindex = random.randint(0, len(self.PIPES_LIST) - 1)
		self.IMAGES['pipe'] = (
			pygame.transform.rotate(pygame.image.load('assets/sprites/pipe-green.png').convert_alpha(), 180),
			pygame.image.load('assets/sprites/pipe-green.png').convert_alpha(),
		)
		# hismask for pipes
		self.HITMASKS['pipe'] = (
			self.getHitmask(self.IMAGES['pipe'][0]),
			self.getHitmask(self.IMAGES['pipe'][1])
		)
		# hitmask for player
		self.HITMASKS['player'] = self.getHitmask(self.IMAGES['player'])

		
		#middle of the screen
		self.MI_playery = int((self.SCREENHEIGHT - self.IMAGES['player'].get_height()) / 2)
		self.MI_basex = 0

		"""game globals"""

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
		self.playerVelY	=  -9   # player's velocity along Y, default same as playerFlapped
		self.playerMaxVelY =  10   # max vel along Y, max descend speed
		self.playerMinVelY =  -8   # min vel along Y, max ascend speed
		self.playerAccY	=   1   # players downward accleration
		self.playerFlapAcc =  -9   # players speed on flapping
		self.playerFlapped = False # True when player flaps


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
		#original->crop->scale->to array
		#1
		
		cropped = pygame.Surface((288,405))
		cropped.blit(self.GAME_IMAGE,(0,0))
		cropped = pygame.transform.scale(cropped , (84,84))
		
		a = numpy.array(pygame.surfarray.array2d(cropped))  
		
		#2
		img_g = Image.fromarray(a).convert('L')
		aa = numpy.array(img_g.getdata(),numpy.uint8)
		aa = aa.reshape(84,84)

		return aa

	def isTerminal(self):
		return self.isDead

	def restart(self):
		self.reward = 0
		self.isDead = False
		self.score = 0

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
		self.playerVelY	=  -9   # player's velocity along Y, default same as playerFlapped
		self.playerMaxVelY =  10   # max vel along Y, max descend speed
		self.playerMinVelY =  -8   # min vel along Y, max ascend speed
		self.playerAccY	=   1   # players downward accleration
		self.playerFlapAcc =  -9   # players speed on flapping
		self.playerFlapped = False # True when player flaps


		self.MI_playery = int((self.SCREENHEIGHT - self.IMAGES['player'].get_height()) / 2)
		self.MI_basex = 0


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
		for i in range(4): 
			if action == 1:
				if self.playery > -2 * self.IMAGES['player'].get_height():
					self.playerVelY = self.playerFlapAcc
					self.playerFlapped = True
			#if action is 0, nothhing happens the bird just follows gravity

			# check for crash here
			crashTest = self.checkCrash({'x': self.playerx, 'y': self.playery},
								   self.upperPipes, self.lowerPipes)
			if crashTest[0]:
				self.isDead = True
				log = open("score_log_stuff.txt","a")
				res = "Game Over! Score: "+str(self.score) #," Agent reward: ",self.reward
				log.write(res+"\n"	)
				log.close()
				#print res
				#if self.score!=0:
				#	print " Conversion: ",self.reward/self.score 
				return 0

			self.reward += 0.1
			# check for score
			playerMidPos = self.playerx + self.IMAGES['player'].get_width() / 2
			for pipe in self.upperPipes:
				pipeMidPos = pipe['x'] + self.IMAGES['pipe'][0].get_width() / 2
				if pipeMidPos <= playerMidPos < pipeMidPos + 4:
					self.reward += 1
					self.score += 1

			self.basex = -((-self.basex + 100) % self.baseShift)

			# player's movement
			if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
				self.playerVelY += self.playerAccY
			if self.playerFlapped:
				#print "stopped flapping"
				self.playerFlapped = False
			self.playerHeight = self.IMAGES['player'].get_height()
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


			self.GAME_IMAGE.blit(self.IMAGES['background'], (0,0))

			for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
				self.GAME_IMAGE.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
				self.GAME_IMAGE.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))


			self.GAME_IMAGE.blit(self.IMAGES['base'], (self.basex, self.BASEY))
			
			# print score so player overlaps the score
			

			self.GAME_IMAGE.blit(self.IMAGES['player'], (self.playerx, self.playery))

			if self.play:
				self.showScore(self.score)
				self.SCREEN.blit(self.GAME_IMAGE,(0,0))
				pygame.display.update()
			
			self.FPSCLOCK.tick(self.FPS)
			
		return self.reward


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
		player['w'] = self.IMAGES['player'].get_width()
		player['h'] = self.IMAGES['player'].get_height()

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
				pHitMask = self.HITMASKS['player']
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
for testing:
"""
"""
def main():
	flappy = FlappyBird(play=True)
	for i in range(100):
		action = random.randint(0,2)
		flappy.act(action)
		if flappy.isTerminal():
			flappy.restart()

if __name__ == "__main__":
	main()
"""