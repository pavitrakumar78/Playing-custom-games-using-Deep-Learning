#! /usr/bin/env python

#Import
import os, sys, pygame, random
from pygame.locals import *
import matplotlib.pyplot as plt

import numpy as numpy
from PIL import Image

play_to_user = True

pygame.init()
#pygame.display.set_caption("Space Shooter")

screen = pygame.display.set_mode((400, 600))
#pygame.mouse.set_visible(1)

#Background
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill((0,0,0))

GAME_IMAGE = pygame.Surface((400, 600))

total_score = 0

class Shooter:
	def __init__(self,play):
		global total_score
		self.prev_action = 0
		self.play = play
		if self.play:
			self.FPS = 50
		else:
			self.FPS = 1500
		#self.screen = pygame.display.set_mode((800, 600))
		pygame.mouse.set_visible(1)
		self.clock = pygame.time.Clock()

		total_score = 0
		self.rate = 0.05
		self.isDead = False

		self.player = Player()
		self.score = Score()
			
		#Player/Enemy
		self.playerSprite = pygame.sprite.RenderPlain((self.player))
			
		self.enemySprites = pygame.sprite.RenderPlain(())
		self.enemySprites.add(Enemy(100))
		self.enemySprites.add(Enemy(200))
		self.enemySprites.add(Enemy(300))
			
		#Score/and game over
		self.scoreSprite = pygame.sprite.Group(self.score)
		self.gameOverSprite = pygame.sprite.RenderPlain(())
			
		self.keepGoing = True
		self.counter = 0

#		self.game()
	def act(self,action):
		#Main Loop
		global total_score
		global play_to_user
		i =0
		while i<4:
			i+=1
			self.clock.tick(self.FPS) #30-60 for playing
			
			total_score += self.rate
			#below is for 8-input game
			"""
			if action == 1: #up
				self.player.dy = -10
			elif action == 2: #down
				self.player.dy = 10
			elif action == 3: #left
				self.player.dx = -10
			elif action == 4: #right
				self.player.dx = 10 
			elif action == 5: #up + left
				self.player.dy = -10
				self.player.dx = -10
			elif action == 6: #up + right
				self.player.dy = -10
				self.player.dx = 10 
			elif action == 7: #down + right
				self.player.dy = 10
				self.player.dx = 10
			elif action == 8: #down + left
				self.player.dy = 10
				self.player.dx = -10
			else:
				if self.prev_action == 1: #up
					self.player.dy = 0
				elif self.prev_action == 2: #down
					self.player.dy = 0
				elif self.prev_action == 3: #left
					self.player.dx = 0
				elif self.prev_action == 4: #right
					self.player.dx = 0
				elif self.prev_action == 5: #up + left
					self.player.dy = 0
					self.player.dx = 0
				elif self.prev_action == 6: #up + right
					self.player.dy = 0
					self.player.dx = 0 
				elif self.prev_action == 7: #down + right
					self.player.dy = 0
					self.player.dx = 0
				elif self.prev_action == 8: #down + left
					self.player.dy = 0
					self.player.dx = 0
			"""
			
			if i==1: #do action for only 1 of the 4 frames - try this also!
				#below is for 2 input game
				if action == 0: #left
					self.player.dx = -10
				elif action == 1: #right
					self.player.dx = 10  

				#below is for 4 input game
				#if action == 0: #left
				#		self.player.dx = -10
				#elif action == 1: #right
				#		self.player.dx = 10 
				#elif action == 2: #up
				#		self.player.dy = -10
				#elif action == 3: #down
				#		self.player.dy = 10

				#below is for 8 input game
				#if action == 0: #left
				#		self.player.dx = -10
				#elif action == 1: #right
				#		self.player.dx = 10 
				#elif action == 2: #up
				#		self.player.dy = -10
				#elif action == 3: #down
				#		self.player.dy = 10
				#elif action == 4: #up + left
				#	self.player.dy = -10
				#	self.player.dx = -10
				#elif action == 5: #up + right
				#	self.player.dy = -10
				#	self.player.dx = 10 
				#elif action == 6: #down + right
				#	self.player.dy = 10
				#	self.player.dx = 10
				#elif action == 7: #down + left
				#	self.player.dy = 10
				#	self.player.dx = -10
					

			#Update and draw on the screen
				
			#Update		
			screen.blit(background, (0,0))
			self.playerSprite.update()
			self.enemySprites.update()
			self.scoreSprite.update()
			

			#Draw
			GAME_IMAGE.blit(background,(0,0))
			self.playerSprite.draw(GAME_IMAGE)
			self.enemySprites.draw(GAME_IMAGE)
			self.scoreSprite.draw(GAME_IMAGE)

			if self.play:
				screen.blit(GAME_IMAGE,(0,0))	#uncomment this to display the game playing.


			pygame.display.flip()
			self.prev_action = action

			#Spawn new enemies
			self.counter += 1
			if self.counter >= 600: #spawn 1 every 'x' iterations..
					self.enemySprites.add(Enemy(random.randrange(15,385))) #300 is the where it spawns
					self.counter = 0 
									
			#Check if enemy collides with player 
			#											group1,			group2
			for hit in pygame.sprite.groupcollide(self.enemySprites, self.playerSprite, 1, 0): 
					#1 indicates that remove the sprite from group1 (enempysprite) if it collides
					self.enemySprites.add(Enemy(random.randrange(15,385)))
					self.score.shield -= 100 #initial value was 100, we instantly kill it if it hits enemy- no lives
					if total_score > 4:
							total_score -= 4 #PUNISHMENT!
					if self.score.shield <= 0:
							self.isDead = True
		return total_score

	def getScreen(self):
		#scale and crop vs crop and scale? - do whichever has less steps.	
		#1	
		cropped = pygame.Surface((400,600))
		cropped.blit(GAME_IMAGE,(0,-40))
		cropped = pygame.transform.scale(cropped , (84,84))
				
		a = numpy.array(pygame.surfarray.array2d(cropped))	#original->crop->scale->to array

		img_g = Image.fromarray(a).convert('L')
		aa = numpy.array(img_g.getdata(),numpy.uint8)
		aa = aa.reshape(84,84)

		return aa

	def isTerminal(self):
		return self.isDead

	def restart(self):
		global total_score
		total_score = 0
		self.rate = 0.05
		self.isDead = False

		self.player = Player()
		self.score = Score()
			
		#Player/Enemy
		self.playerSprite = pygame.sprite.RenderPlain((self.player))
			
		self.enemySprites = pygame.sprite.RenderPlain(())
		self.enemySprites.add(Enemy(100))
		self.enemySprites.add(Enemy(200))
		self.enemySprites.add(Enemy(300))
			
		#Score/and game over
		self.scoreSprite = pygame.sprite.Group(self.score)
		self.gameOverSprite = pygame.sprite.RenderPlain(())
			
		self.keepGoing = True
		self.counter = 0


	def numActions(self):
		return 2 
		#return 4
		#return 8

	def getActions(self):
		return [0,1] #0 is go left and 1 is go right
		#return [0,1,2,3] #0 is go left and 1 is go right 2 is go up 3 is go down
		#return [0,1,2,3,4,5,6,7]

#Load Images
def load_image(name, colorkey=None):
		fullname = os.path.join('sprites', name)
		try:
				image = pygame.image.load(fullname)
		except pygame.error, message:
				print 'Cannot load image:', fullname
				raise SystemExit, message
		image = image.convert()
		if colorkey is not None:
				if colorkey is -1:
						colorkey = image.get_at((0,0))
				image.set_colorkey(colorkey, RLEACCEL)
		return image, image.get_rect()

				

#Player
class Player(pygame.sprite.Sprite):
		def __init__(self):
				pygame.sprite.Sprite.__init__(self)
				self.image, self.rect = load_image("player.png", -1)
				self.rect.center = (200,500)
				self.dx = 0
				self.dy = 0
				self.reset()
				self.lasertimer = 0
				self.lasermax = 5
				self.bombamount = 1
				self.bombtimer = 0
				self.bombmax = 10
				
		def update(self):
				self.rect.move_ip((self.dx, self.dy))
																
				#Player Boundaries		
				if self.rect.left < 0:
					self.rect.left = 0
				elif self.rect.right > 400:
					self.rect.right = 400
				
				if self.rect.top <= 260:
					self.rect.top = 260
				elif self.rect.bottom >= 600:
					self.rect.bottom = 600
				
		def reset(self):
				self.rect.bottom = 600	


#Enemy class
class Enemy(pygame.sprite.Sprite):
		def __init__(self, centerx):
				pygame.sprite.Sprite.__init__(self)
				self.image, self.rect = load_image("enemy.png", -1)
				self.rect = self.image.get_rect()
				self.dy = 8
				self.reset()
				
		def update(self):
				self.rect.centerx += self.dx
				self.rect.centery += self.dy
				if self.rect.top > screen.get_height():
						self.reset()
				#if self.rect.left < 0:
				#		self.reset()
				#if self.rect.right > 400:
				#		self.reset()
				if self.rect.centerx <= 15 or self.rect.centerx >= 385:
						#self.reset()
						self.dx = -1*self.dx	
					
		
		def reset(self):
				self.rect.bottom = 0
				self.rect.centerx = random.randrange(10, screen.get_width())
				self.dy = random.randrange(5, 10)
				random_x_speed = random.randrange(0,5)
				arr = [-2,-1,0,1,2]
				if self.rect.centerx > 380 or self.rect.centerx < 20:
					self.dx = 0
				else:
					self.dx = arr[random_x_speed]

class Score(pygame.sprite.Sprite):
		def __init__(self):
				pygame.sprite.Sprite.__init__(self)
				self.shield = 100
				self.score = 0
				self.bomb = 1
				self.font = pygame.font.SysFont("Arial", 28)
				
		def update(self):
				global total_score
				self.text = "Score: %d" % (total_score)
				self.image = self.font.render(self.text, 1, (0, 255, 0))
				self.rect = self.image.get_rect()
				self.rect.center = (200,20)
				
"""below is for manual hand-playing-testing"""

"""
	#Game Module		
	def game(self):
			#Game Objects
			player = Player()
			score = Score()
			
			#Player/Enemy
			playerSprite = pygame.sprite.RenderPlain((player))
			
			enemySprites = pygame.sprite.RenderPlain(())
			enemySprites.add(Enemy(200))
			enemySprites.add(Enemy(300))
			enemySprites.add(Enemy(400))
			
			#Score/and game over
			scoreSprite = pygame.sprite.Group(score)
			gameOverSprite = pygame.sprite.RenderPlain(())
			
				
			#Set Clock
			keepGoing = True
			counter = 0
		
			#Main Loop
			while keepGoing:
				self.clock.tick(60)
				self.total_score += self.rate
				#input
				for event in pygame.event.get():
							if event.type == pygame.QUIT:
									keepGoing = False
									sys.exit(0)
							elif event.type == pygame.KEYDOWN:
									if event.key == pygame.K_ESCAPE:
											sys.exit(0)
											keepGoing = False
									elif event.key == pygame.K_LEFT:
											player.dx = -10
									elif event.key == K_RIGHT:
											player.dx = 10
									elif event.key == K_UP:
											player.dy = -10
									elif event.key == K_DOWN:
											player.dy = 10
							elif event.type == KEYUP:
									if event.key == K_LEFT:
											player.dx = 0
									elif event.key == K_RIGHT:
											player.dx = 0
									elif event.key == K_UP:
											player.dy = 0
									elif event.key == K_DOWN:
										player.dy = 0
									
							
							
				#Update and draw on the screen
				
				#Update		
				screen.blit(background, (0,0))		
				playerSprite.update()
				enemySprites.update()
				scoreSprite.update()
				
				#Draw
				playerSprite.draw(screen)
				enemySprites.draw(screen)
				scoreSprite.draw(screen)

				pygame.display.flip()
			
				#Spawn new enemies
				counter += 1
				if counter >= 1000: #spawn 1 every 'x' iterations..
						enemySprites.add(Enemy(300)) #300 is the where it spawns
						counter = 0 
						print "count end"		
				print len(enemySprites)
									
				#Check if enemy collides with player 
				#																			group1,			group2
				for hit in pygame.sprite.groupcollide(enemySprites, playerSprite, 1, 0): 
						#1 indicates that remove the sprite from group1 (enempysprite) if it collides
						enemySprites.add(Enemy(300))
						score.shield -= 10
						if score.shield <= 0:
								#game()
								self.restart()
			return self.total_score
"""
							
"""
for testing:
"""
"""
#Main
def main():
		#game()
		global play_to_user
		play_to_user = True
		shoot = Shooter(play = True)
		for i in range(100):
			action = random.randrange(shoot.numActions())
			#action = int(raw_input()) #does not work properly in sublime!
			if action==9:
				shoot.getScreen()
			else:
				shoot.act(action)
			if shoot.isTerminal():
				shoot.restart()

if __name__ == "__main__":
	main()
"""


"""
controls:
1 - up
2 - down
3 - left
4 - right

5 - up + left
6 - up + right
7 - down + left
8 - down + right

0 - no action
"""