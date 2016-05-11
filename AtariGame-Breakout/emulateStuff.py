import sys
from ale_python_interface import ALEInterface
import cv2
import numpy as np


class Emulate:
  def __init__(self, rom_file, display_screen=False,frame_skip=4,screen_height=84,screen_width=84,repeat_action_probability=0,color_averaging=True,random_seed=0,record_screen_path='screen_pics',record_sound_filename=None,minimal_action_set=True):
    self.ale = ALEInterface()
    if display_screen:
      if sys.platform == 'darwin':
        import pygame
        pygame.init()
        self.ale.setBool('sound', False) # Sound doesn't work on OSX
      elif sys.platform.startswith('linux'):
        self.ale.setBool('sound', True)
      self.ale.setBool('display_screen', True)

    self.ale.setInt('frame_skip', frame_skip)
    self.ale.setFloat('repeat_action_probability', repeat_action_probability)
    self.ale.setBool('color_averaging', color_averaging)

    if random_seed:
      self.ale.setInt('random_seed', random_seed)

    self.ale.loadROM(rom_file)

    if minimal_action_set:
      self.actions = self.ale.getMinimalActionSet()
    else:
      self.actions = self.ale.getLegalActionSet()

    self.dims = (screen_width,screen_height)

  def numActions(self):
    return len(self.actions)

  def getActions(self):
  	return self.actions

  def restart(self):
    self.ale.reset_game()

  def act(self, action):
    reward = self.ale.act(self.actions[action])
    return reward

  def getScreen(self):
    screen = self.ale.getScreenGrayscale()
    resized = cv2.resize(screen, self.dims)
    return resized

  def getScreenGray(self):
    screen = self.ale.getScreenGrayscale()
    resized = cv2.resize(screen, self.dims)
    rotated = np.rot90(resized,k=1)
    return rotated

  def getScreenColor(self):
    screen = self.ale.getScreenRGB()
    resized = cv2.resize(screen, self.dims)
    rotated = np.rot90(resized,k=1)
    return rotated

  def isTerminal(self):
    return self.ale.game_over()