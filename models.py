import json
from scipy.optimize import LinearConstraint, minimize
import numpy as np

#preprocessing step
#the variable processed is a list of (payoffs, freqs) tuples
#payoffs is an 8 tuple representing the outcomes of a game in the form: (r0c0o0, r0c0o1, r0c1o0, r0c1o1, r1c0o0, r1c0o1, r1c1o0, r1c1o1)
#where r = row, c = column, o = outcome (for which player)
#freqs is a 4 tuple of the number of times each action was played in the data, in the form: (row 0, row 1, column 0, column 1)
#everything is normalized between 0 and 1

#NOTE: this is only currently handling 2x2 games. The one 3x3 rock paper scissors game is currently being removed.

games = json.load(open('games.json', 'r'))
data = open("dataset.csv", 'r').read().split("\n")
data = [x.split(",") for x in data]
data = data[1:-1]

for i in range(len(data)):
    if data[i][2] == 'row':
        data[i] = [int(data[i][0]), 0, data[i][3], int(data[i][4])]
    else:
        data[i] = [int(data[i][0]), 1, data[i][3], int(data[i][4])]

i = 0
while i < len(data):
    if data[i][0] == 15:
        del data[i]
    else:
        if data[i][0] > 15:
            data[i][0] -= 1
        i += 1

del games[15]

# for x in data:
#     print(x)

# def lst_to_game(l):
#     return (((l[0],l[1]), (l[2],l[3])), ((l[4],l[5]), (l[6],l[7])))

processed = []

for id, game in enumerate(games):
    # print(id)
    if id == 19:
        continue
    p = game["payoffs"]
    k = list(p.keys())
    p0a0 = k[0]
    p0a1 = k[1]
    k1 = list(p[p0a0].keys())
    p1a0 = k1[0]
    p1a1 = k1[1]
    l = [p[p0a0][p1a0][0], p[p0a0][p1a0][1], p[p0a0][p1a1][0], p[p0a0][p1a1][1], p[p0a1][p1a0][0], p[p0a1][p1a0][1], p[p0a1][p1a1][0],p[p0a1][p1a1][1]]
    mn = min(l)
    l = [x-mn for x in l]
    mx = max(l)
    payoffs = [x / mx for x in l]
    # print(p)
    # print(payoffs)
    # print(data[id*4])
    freqs = []
    if data[id*4+2][2] == p0a0:
        freqs = [data[id*4+2][3], data[id*4+3][3]]
    else:
        freqs = [data[id*4+3][3], data[id*4+2][3]]

    if data[id*4][2] == p1a0:
        freqs += [data[id*4][3], data[id*4+1][3]]
    else:
        freqs += [data[id*4+1][3], data[id*4][3]]

    mx = max(freqs)
    freqs = (freqs[0] / (freqs[0] + freqs[1]), freqs[1] / (freqs[0] + freqs[1]), freqs[2] / (freqs[2] + freqs[3]), freqs[3] / (freqs[2] + freqs[3]))
    # print(data[id*4:id*4+4])
    # print(p0a0, p0a1, p1a0, p1a1)
    # print(freqs)
    # print()
    processed.append((payoffs, freqs))

#end of preprocessing

def best_reponse(game, s):
    # print(game, s)
    #   0        1       2       3       4       5       6       7
    #(r0c0o0, r0c0o1, r0c1o0, r0c1o1, r1c0o0, r1c0o1, r1c1o0, r1c1o1)
    action_0_utility = s[2] * game[0] + s[3] * game[2]
    action_1_utility = s[2] * game[4] + s[3] * game[6]
    if action_0_utility >= action_1_utility:
        p0 = (1, 0)
    else:
        p0 = (0, 1)
    action_0_utility = s[0] * game[1] + s[1] * game[5]
    action_1_utility = s[0] * game[3] + s[1] * game[7]
    if action_0_utility >= action_1_utility:
        p1 = (1, 0)
    else:
        p1 = (0, 1)
    return (p0[0], p0[1], p1[0], p1[1])

#find all level k strategies up to the specified k (inclusive)
#returns a tuple corresponding to the probability of each action for each level k strategy in the form (row 0, row 1, column 0, column 1)
#level 0 is currently uniform random
#does not support mixed strategies past level 0, is this necessary??
def find_level_k_strategies(game, k):
    level_k = [(0.5, 0.5, 0.5, 0.5)]
    for i in range(k):
        level_k.append(best_reponse(game, level_k[i]))

    return level_k

max_k = 5

for pair in processed:
    print("payoffs:")
    print(pair[0])
    print("freqs:")
    print(pair[1])
    print("level k strategies:")
    level_k = find_level_k_strategies(pair[0], max_k)
    for i, x in enumerate(level_k):
        print("k =", i, ":", x)

    print()

train = processed[:15]
test = processed[15:]
print(len(processed))

constraint0 = LinearConstraint(np.ones(max_k+1), lb=1, ub=1)

def level_k_diff(x, processed, max_k):
    total = 0
    for game, freqs in processed:
        level_k = find_level_k_strategies(game, max_k)
        est = [0, 0, 0, 0]
        for i in range(max_k+1):
            est[0] += level_k[i][0] * x[i]
            est[1] += level_k[i][1] * x[i]
            est[2] += level_k[i][2] * x[i]
            est[3] += level_k[i][3] * x[i]
        for i in range(4):
            total += (est[i] - game[i]) ** 2
    return total

x0 = np.zeros(max_k+1)
x0[0] = 1
bnds = [(0, 1) for x in range(max_k+1)]
fitted = minimize(level_k_diff, x0, (train, max_k), bounds = bnds, constraints = constraint0)
print(fitted)

print("Level-k model")
print("Train loss:", level_k_diff(fitted.x, train, max_k) / len(train))
print("Test loss:", level_k_diff(fitted.x, test, max_k) / len(test))

def CH_pi(x, game, k):
    if k == 0:
        return np.array([0.5,0.5,0.5,0.5])
    mixed = np.zeros(4)
    for i in range(k):
        mixed += x[i] * CH_pi(x, game, i)
    mixed /= np.sum(x[:k])
    return np.array(best_reponse(game, mixed)).astype("float64")

# for k in range(1, max_k+1):
# print(processed[0][0])
def CH_diff(x, processed, max_k):
    total = 0
    for game, freqs in processed:
        est = np.array([0,0,0,0]).astype("float64")
        for i in range(max_k):
            est += CH_pi(x, game, i) * x[i]
        for i in range(4):
            total += (est[i] - game[i]) ** 2
    return total

fitted_CH = minimize(CH_diff, x0, (train, max_k), bounds = bnds, constraints = constraint0)
print("CH")
print(fitted_CH)
print("Train loss:", CH_diff(fitted_CH.x, train, max_k) / len(train))
print("Test loss:", CH_diff(fitted_CH.x, test, max_k) / len(test))
