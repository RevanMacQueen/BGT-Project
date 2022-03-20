import numpy as np
from itertools import product
from copy import deepcopy


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
