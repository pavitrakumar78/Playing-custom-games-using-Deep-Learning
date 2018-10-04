# Playing custom games using Deep Learning

Implementation of Google's paper on playing atari games using deep learning in python.

Paper Authors: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller  
Paper Link: http://arxiv.org/abs/1312.5602

This project presents an implementation of a model (based on the above linked paper) that successfully learns control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards. This model is tested on variety of Atari and custom made games and its performance is compared with human players.

Dependencies:

- Python 2.7
- numpy
- Lasagne
- Theano
- matplotlib
- scipy
- Arcade Learning environment (for Atari games)
- pygame (for flappy bird and shooter)
- GPU with CC score of greater than or equal to 3 (refer http://deeplearning.net/software/theano/library/tensor/nnet/conv.html and https://developer.nvidia.com/cuda-gpus)  


## Atari Games  
The model is trained for 2 Atari games - Space Invaders and Breakout. The model was trained for about 12-13 hrs and has achieved  good performace that is consistent with the paper.  

To run the agent:  

### Breakout:

```
python AtariGame-Breakout/tester.py --rom_file breakout.bin --play_games 10 --display_screen --load_weights breakout_models/dep-q-rmsprop-breakout99-epoch.pkl 
```


### Space Invaders:  

```
python AtariGame-SpaceInvaders/tester.py --rom_file space_invaders.bin --play_games 10 --display_screen --load_weights spaceinvaders_models/dep-q-rmsprop-space_invaders99-epoch.pkl
``` 
## Custom Games  

### Flappy Bird - Q Learning 

I have trained a plain vanilla Q learning (based on http://sarvagyavaish.github.io/FlappyBirdRL/) based agent where the agent gets information such as the x and y distance from the pipes to compare the performance of this game-specific model to a generalized model as described in the Google's paper. Training time is about 2-3 hrs.

To run the agent:

```
python FlappyQ/run_qvflappy.py
```  

### Flappy Bird - DQN

Similar to the Atari games, I have trained the same model with minor only minor modificaions to the parameters to play Flappy Bird - although the performance is not as good as the Q learning mode which had explicit game data - it still gets a decent average score of about 20-30.

To run the agent:

``` 
python FlappyBirdDQN/ftester.py --play_games 10 --display_screen --load_weights flappy_models/dep-q-flappy-60-epoch.pkl
```  

### Shooter game

This is a very simple game I made using pygame where the player controls a  "spaceship" is tasked to dodge the incomming "meteoroids" and stay alive as long as possible. I also tried an (silly?) experiment where I trained different models wherein each model had agents with different degrees of control over the space ship and compared the performance of the same.

To run the agent with just 2 control setting (left and right):  

``` 
python ShooterDQN/stester2.py --play_games 10 --display_screen --load_weights shooter_models/dep-q-shooter-nipscuda-8movectrl-99-epoch.pkl 
```  

To run the agent with just 4 control setting (left, right, top and bottom):  

``` 
python ShooterDQN/stester4.py --play_games 10 --display_screen --load_weights shooter_models/dep-q-shooter-nipscuda-4movectrl-99-epoch.pkl 
```  

To run the agent with just 8 control setting (all directions):  

``` 
python ShooterDQN/stester8.py --play_games 10 --display_screen --load_weights shooter_models/dep-q-shooter-nipscuda-2movectrl-80-epoch.pkl 
```  

## Statistics
For all the below graphs, the X axis is the traning timeline and the Y axis the score funtion for each game.  
(Note: scores in Shooter anf flappy bird have been modified (reward amplified) because the original +1 or -1 is not applicable since the player does not have "lives" here and rewards are also very sparse in the these 2 games.)  

Atari Breakout:  
<img src="https://github.com/pavitrakumar78/Playing-custom-games-using-Deep-Learning/blob/master/Stats/AtariStatBreakout.png" width="300" height="300" />  

Atari Space Invaders:  
<img src="https://github.com/pavitrakumar78/Playing-custom-games-using-Deep-Learning/blob/master/Stats/AtariStatSpaceInvaders.png" width="300" height="300" />     

Flappy Q Learning:  
<img src="https://github.com/pavitrakumar78/Playing-custom-games-using-Deep-Learning/blob/master/Stats/FlappyQStat.png" width="300" height="300" />    

Flappy DQN:  
<img src="https://github.com/pavitrakumar78/Playing-custom-games-using-Deep-Learning/blob/master/Stats/FlappyDQNStat.png" width="300" height="300" />    

Shooter (4-control):  
<img src="https://github.com/pavitrakumar78/Playing-custom-games-using-Deep-Learning/blob/master/Stats/Shooter-4ControlStat.png" width="300" height="300" />    


## Pics:

Atari Breakout:  
<img src="https://github.com/pavitrakumar78/Playing-custom-games-using-Deep-Learning/blob/master/Pics/ataribreakout.png" width="400" height="400" />  

Atari Space Invaders:  
<img src="https://github.com/pavitrakumar78/Playing-custom-games-using-Deep-Learning/blob/master/Pics/atarispaceinvaders.png" width="400" height="400" />    

Flappy Bird - DQN:  
<img src="https://github.com/pavitrakumar78/Playing-custom-games-using-Deep-Learning/blob/master/Pics/flappydqn.png" width="300" height="600" />    

Flappy Bird - Q Learning:  
<img src="https://github.com/pavitrakumar78/Playing-custom-games-using-Deep-Learning/blob/master/Pics/flappyqn.png" width="300" height="600" />   

Shooter (custom game):  
<img src="https://github.com/pavitrakumar78/Playing-custom-games-using-Deep-Learning/blob/master/Pics/shooter.png" width="300" height="600" />    


## Note:
Number of epochs and train cycles has been adjusted such that all the above code when used for traning takes only about 12-15 hrs max. depending on your CPU and GPU (My CPU: i5 3.4 GHz and GPU: nVidia GeForce 660). Also, do not expect super human level performance (as said in Google's paper) from the models as I have trained it only for 12-15 hrs - more traning with further parameter tuning can improve the scores of all the above games.


## Resources used:  

The deep Q network used in this project is a modified version of spragunr's dqn code (https://github.com/spragunr/deep_q_rl).  
[1] Deep Learning in Neural Networks: An Overview http://arxiv.org/abs/1404.7828  
[2] The Arcade Learning Environment: https://www.jair.org/media/3912/live-3912-7087-jair.pdf  
[3] ImageNet Classification with Deep Convolutional Neural Networks:  https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf  
[4]	Lasagne:		https://github.com/Lasagne/Lasagne  
[5]	Theano:		http://deeplearning.net/software/theano/  
[6]	CUDA:		https://developer.nvidia.com/cudnn  
[7]	Pygame:		http://www.pygame.org/docs/  
[8]	General:		http://deeplearning.net/  


