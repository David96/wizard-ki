from environment import WizardEnv
import sys
from threading import Thread
import time

if len(sys.argv) != 3:
    print("wrong number of args")

num_players = int(sys.argv[1])
num_games = int(sys.argv[2])

players = []
for i in range(num_players):
    players.append(WizardEnv("KI %d" % i, i == 0))

for game in range(num_games):
    threads = []
    for p in players:
        t = Thread(target=p.play_game)
        threads.append(t)
        t.start()
        time.sleep(0.5)
    print("Started game %d" % game)
    players[0].send('{"action": "start_game"}')
    for t in threads:
        t.join()
