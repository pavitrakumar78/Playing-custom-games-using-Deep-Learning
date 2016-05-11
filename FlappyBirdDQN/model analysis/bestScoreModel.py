


import os

#epoch = 40

num_games = 50

for i in range(58,65):
	os.system("python ftester.py --play_games "+str(num_games)+" --display_screen --load_weights flappy_models/dep-q-flappy-"+str(i)+"-epoch.pkl")


