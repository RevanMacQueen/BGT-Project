import numpy as np
from itertools import product
from copy import deepcopy
import pandas as pd

def random_argmax(a):
    '''
    like np.argmax, but returns a random index in the case of ties
    '''
    return np.random.choice(np.flatnonzero(a == a.max()))


def get_payoffs(payoffs, actions):
    actions = tuple(([a] for a in actions))
    idx = np.ix_(np.arange(len(actions)), *actions)
    return np.squeeze(payoffs[idx])  


def p_outcomes(s):
    '''
    returns a probability distribution over outcomes
    '''
    p = s[0]
    for s_i in s[1:]:
        p =  np.multiply.outer(p, s_i)
    return p


def expected_utility(s, payoffs):
    '''
    Gets the expected utilitiy for a player under strategy profile s
    
    Parameters
        s : (list) list where S[i] gives the probability of player i playing each action
        payoffs: (np.array) payoffs for i
    Returns
        eu : (float) expected utility for i
    '''
    p = p_outcomes(s)
    return np.sum(np.multiply(p, payoffs))


def p_traj(a, s):
    '''
    Returns the probability of a sequence of actions
    a being sampled from strategy s.

    Paramters:
        a : (np.Array) 2 x n array of actions chosen
        s : (np.Array) 2 x n array of strategies 
    '''

    assert a.shape == s.shape
    L = a.shape[1]
    prod = 1
    for l in range(L):
        idx = np.where(a[:, l])
        prod *= s[:, l][idx][0]

    return prod


def load_data(file):
    data = np.genfromtxt(file, delimiter=',', dtype=str)
    data = np.delete(data, 0, 0)

    new_data = []

    # turn each game into entry in this array

    def remove_blank(l):
        return list(filter(lambda a: a != '', l))

    def replace_letters(l):
        return [[1,0] if x=='c' else [0, 1] for x in l]

    for i in range(0, data.shape[0], 2):
        p1 = data[i]
        p2 = data[i+1]

        game = int(p1[0])
        player1 = p1[2]
        player2 = p2[2]
        p1_actions = np.array(replace_letters(remove_blank(p1[3:]))).T
        p2_actions =  np.array(replace_letters(remove_blank(p2[3:]))).T
    
        actions =  [p1_actions, p2_actions]

        new_data.append([game, player1, player2, actions])

    df = pd.DataFrame(new_data)
    df.columns = ['Game', 'Player_1', 'Player_2', 'Actions']

    return list(df.Actions)