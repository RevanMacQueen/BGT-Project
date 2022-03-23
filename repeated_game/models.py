import numpy as np

from scipy.optimize import minimize, LinearConstraint
from scipy.special import softmax
from utils import expected_utility

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


class Level_K_Model:
    def __init__(self, K):
        self.K = K
        self.payoffs = np.zeros([2, 2, 2])
        self.payoffs[0, :, :] = np.array([[0, -1],
                                          [1, -1]])
        
        
        
        self.payoffs[1, :, :] = np.array([[0, 1],
                                         [-1, -1]])
        self.alphas = None
      


    def log_likelihood_independent(self, params, dataset):
        '''
        Given the dataset and the fitted model, returns the log likelihood. This assumes a players level may change as the game goes on

        NOTE: assumes each player has a fixed level
        
        Parameters:
            params : np.array of alpha_ks, the freq of that level in population and lambda_ for QBR
            dataset : list of plays, each play is an np array
        Returns:
            ll : loglikelihood of dataset

        '''
        
        alphas = params[0:-2]
        lambda_ = params[-2] 
        kappa  = params[-1] 

        ll = 0
        for play in dataset: 
            for player in range(2):
                traj = play[player]
                other_traj  = play[1-player] # trajectory of other player
                
                pred_s = self.predict_traj(traj, other_traj, self.K, lambda_, kappa, overall=True, alphas=alphas)  
               
                L = traj.shape[1] #length of trajectory
                for l in range(L):
                    idx = np.where(traj[:, l])
                    ll += np.log(pred_s[:, l][idx][0])

        return -ll # since we are minimizing the negative log likelihood


    def log_likelihood(self, params, dataset):
        '''
        Given the dataset and the fitted model, returns the log likelihood.

        NOTE: assumes each player has a fixed level
        
        Parameters:
            params : np.array of alpha_ks, the freq of that level in population and lambda_ for QBR
            dataset : list of plays, each play is an np array
        Returns:
            ll : loglikelihood of dataset

        '''
        alphas = params[0:-2]
        lambda_ = params[-2] 
        kappa  = params[-1] 
       

        ll = 0
        for play in dataset: 
            for player in range(2):
                sum = 0
                for i, k in enumerate(range(self.K+1)): # condition on a specific value of k for a player
                    alpha_k = alphas[i]
    
                    traj = play[player]
                    other_traj  = play[1-player] # trajectory of other player
                    pred_s = self.predict_traj(traj, other_traj, k, lambda_, kappa)
                    
                    prob = p_traj(traj, pred_s) # probability of that trajectory
                    
                    sum += alpha_k * prob # this could be 0, so small epsilon added

                ll += np.log(sum)
                #print(np.log(sum))

        return -ll # since we are minimizing the negative log likelihood


    def fit(self, dataset):
        params = np.zeros(self.K+3) # +2 since level 0 and the lambda parameter, kappa parameter
        params[0] = 1 # inital guess is that all players are level 0
        params[-2] = 5 # intial guess for lambda
        params[-1] = 0.5 # intial guess for kappa

        const_arr = np.ones(self.K+3)
        const_arr[-1] = 0
        const_arr[-2] = 0
        #print(const_arr)
        constraint = LinearConstraint(const_arr, lb=1, ub=1)
        bnds = [(0, 1) for x in range(self.K+1)]
        bnds.append((0, 1000)) 
        bnds.append((0, 1)) 
        #print(bnds)

        result = minimize(
            self.log_likelihood_independent ,
            params, 
            args=(dataset),
            bounds=bnds, 
            constraints=constraint) 
        #print(result)

        assert result.status == 0 # make sure the optimization was successful
        return result


    def predict_traj(self, traj, other_traj, K, lambda_, kappa, overall=False, alphas=None):
        '''
        Returns straetgy predictions against other player
        
        '''
        def get_weights(h, kappa):
            if h == 1:
                return np.ones(1)
            
            l = [1-kappa]
            for i in range(1, h):
                l.append(l[i-1]*kappa)

            weight_left = 1- np.sum(l)
            l[0] = l[0] + weight_left
            
            l = np.flip(np.array(l))

            return l 

        L = other_traj.shape[1]
        pred_i = np.zeros((2, self.K+1, L)) # level-k prediction for each stage game for i
        pred_other = np.zeros((2, self.K+1, L)) # level-k prediction for each stage game for other player

        for l in range(L):
            # first determine level 0 strategy 
            start_hist_idx = 0 #max(l-self.h, 0)
            end_hist_idx = l

            if l == 0: # there is no history, so level-0 strategies are uniform
                lvl_0_s_i = np.ones((2)) / 2
                lvl_0_s_other = np.ones((2)) / 2
            else:
                hist_i = traj[:, start_hist_idx:end_hist_idx] # limited history of i's actions
                hist_other = other_traj[:, start_hist_idx:end_hist_idx] # limited history of -i's actions

                w = get_weights(hist_i.shape[1], kappa)
                lvl_0_s_i =  0.999*np.dot(hist_i, w)+ 0.001 * np.ones((2))/2
                lvl_0_s_other=  0.999*np.dot(hist_other, w)+ 0.001 * np.ones((2))/2


            # These become the level 0 strategies
            pred_i[:, 0, l] = lvl_0_s_i
            pred_other[:, 0, l] = lvl_0_s_other

            # Now, for higher levels:
            for k in range(1, K+1):
                pred_i[:, k, l] = 0.999*self.compute_BR(pred_other[:, k-1, l], lambda_) + 0.001 * np.ones((2))/2
                pred_other[:, k, l] =  0.999*self.compute_BR(pred_i[:, k-1, l], lambda_)+ 0.001 * np.ones((2))/2
 

        if overall: # if return the overall prediction
            assert alphas is not None
            pred_i_ = np.zeros((2, L))

            for l in range(L):
                pred_i[:, 0, l] = np.ones((2)) / 2
                pred_i_[:, l] = np.dot(pred_i[:, :, l], alphas)
            pred_i = pred_i_
     
        else:
            pred_i = np.squeeze(pred_i[:, K, :]) # only return the level K predictions


        # if K == 0:
        #     return np.ones((2, L)) / (np.ones(L)*2)
        # else:
        return pred_i   


    def compute_BR(self,  s_other, lambda_):
        '''
        Computes a best response

        Parameters:
            s_other : (np.Array) strategy of other player

        NOTE: this ONLY works with symetric payoffs and 2 actions
        '''
    
        #get EU of action 0
        s = [np.array([1, 0]), s_other]
        eu_0 = expected_utility(s, self.payoffs[0])

        # get EU of action 1
        s = [np.array([0, 1]), s_other]
        eu_1 = expected_utility(s, self.payoffs[0])

        # return action with greater EU
        return softmax(np.array([eu_0, eu_1])*lambda_)


class Temporal_Level_Model:
    def __init__(self):
        self.payoffs = np.zeros([2, 2, 2])
        self.payoffs[0, :, :] = np.array([[0, -1],
                                          [1, -2]])
        self.payoffs[1, :, :] = np.array([[0, 1],
                                         [-1, -2]])
        self.alphas = None
       


    def log_likelihood(self, params, dataset):
        '''
        Given the dataset and the fitted model, returns the log likelihood.

        NOTE: assumes each player has a fixed level
        
        Parameters:
            params : np.array of alpha_ks, the freq of that level in population and lambda_ for QBR
            dataset : list of plays, each play is an np array
        Returns:
            ll : loglikelihood of dataset

        '''
        alphas = params[0:3] # frequency of different reasoning types in the population
        
        gamma = params[-3]  # parameter for forward looking type
        lambda_ = params[-2] 
        kappa = params[-1] 
      

        ll = 0
        for play in dataset: 
            for player in range(2):
                sum = 0
                for k, alpha_k in enumerate(alphas): # condition on a specific type of  player
                    traj = play[player]
                    other_traj  = play[1-player] # trajectory of other player
                  
                    pred_s = self.predict_traj(traj, other_traj, lambda_, gamma, alphas, kappa, k)
                    prob = p_traj(traj, pred_s) # probability of that trajectory
                    sum += alpha_k * (prob) # this could be 0, so small epsilon added
                ll += np.log(sum)

        return -ll # since we are minimizing the negative log likelihood


        # ll = 0
        # for play in dataset: 
        #     for player in range(2):
        #         traj = play[player]
        #         other_traj  = play[1-player] # trajectory of other player    
        #         pred_s = self.predict_traj(traj, other_traj, lambda_, gamma, alphas, kappa)
                   
        #         L = traj.shape[1] #length of trajectory
        #         for l in range(L):
        #             idx = np.where(traj[:, l])
        #             ll += np.log(pred_s[:, l][idx][0])

        return -ll # since we are minimizing the negative log likelihood


    def fit(self, dataset):
        params = np.zeros(6) # +2 since level 0 and the lambda parameter
    
        for i in range(3):
            params[i] = 1/3

        params[3] = 0.5 #gamma 
        params[4] = 1 # lambda
        params[5] = 0.5 # kappa

        const_arr = np.zeros(6)
        for i in range(3):
            const_arr[i] = 1
       
        constraint = LinearConstraint(const_arr, lb=1, ub=1)
        bnds = [(0, 1) for x in range(3)]
       
        bnds.append((0, 1)) 
        bnds.append((0, 1000)) # lambda
        bnds.append((0, 1)) 
       
        result = minimize(
            self.log_likelihood ,
            params, 
            args=(dataset),
            bounds=bnds, 
            constraints=constraint) 
        #print(result)

        assert result.status == 0 # make sure the optimization was successful
        return result
    
        
    def predict_traj(self, traj, other_traj, lambda_, gamma, alphas, kappa, k): 
        '''
        I think about what action I can take

        what do I think you will play this round? 
        -> probably a best response to my last action

        what do I think you will play next round?
        -> probably a brest resonse to this aciton

        what will I play next round? 
            -> probably a best resonse to what I think you will play next round
        '''


        L = other_traj.shape[1]
        pred_i = np.zeros((2, len(alphas), L)) # level-k prediction for each stage game for 
        
        def get_weights(h, kappa):
            if h == 1:
                return np.ones(1)
            
            l = [1-kappa]
            for i in range(1, h):
                l.append(l[i-1]*kappa)

            weight_left = 1- np.sum(l)
            l[0] = l[0] + weight_left
            
            l = np.flip(np.array(l))

            return l 

        for l in range(L):
            for k in range(len(alphas)):
                if k == 0:
                    pred_i[:, k, l] = np.ones(2)/2

                if k == 1:
                    
                    if l == 0:
                        # if action 0                   
                        eu_0 = expected_utility([np.array([1,0]), np.array([0.5, 0.5])], self.payoffs[0]) 
                        # if action 1                 
                        eu_1 =  expected_utility([np.array([0,1]), np.array([0.5, 0.5])], self.payoffs[0])
                
                    else:
                        hist = traj[:, 0:l]
                        w = get_weights(hist.shape[1], kappa)
                        s_other_curr = self.BR(np.dot(hist, w), lambda_)
                        # if action 0
                        eu_0 = expected_utility([np.array([1,0]), s_other_curr], self.payoffs[0])
                        # if action 1
                        eu_1 = expected_utility([np.array([0,1]), s_other_curr], self.payoffs[0]) 

                    
                    pred_i[:, k, l]  = softmax(lambda_* np.array([eu_0, eu_1]))

                if k == 2:
                    if l == 0:
                        # if action 0
                        s_other_next = self.BR(np.array([1,0]), lambda_)
                        s_i_next = self.BR(s_other_next, lambda_)
                        eu_0 = expected_utility([np.array([1,0]), np.array([0.5, 0.5])], self.payoffs[0]) + gamma*expected_utility([s_i_next, s_other_next], self.payoffs[0])

                        # if action 1
                        s_other_next = self.BR(np.array([0,1]), lambda_)
                        s_i_next = self.BR(s_other_next, lambda_)
                        eu_1 =  expected_utility([np.array([0,1]), np.array([0.5, 0.5])], self.payoffs[0]) + gamma*expected_utility([s_i_next, s_other_next], self.payoffs[0])
                
                    else:
                        # if action 0

                        hist = traj[:, 0:l]
                        w = get_weights(hist.shape[1], kappa)
        
                        s_other_curr = self.BR(np.dot(hist, w), lambda_)

                        hist_new = np.c_[hist, np.array([1,0]) ]
                        w = get_weights(hist_new.shape[1], kappa)
                        s_other_next = self.BR(np.dot(hist_new, w), lambda_)
                        s_i_next = self.BR(s_other_next, lambda_)

                        eu_0 = expected_utility([np.array([1,0]), s_other_curr], self.payoffs[0]) + gamma*expected_utility([s_i_next, s_other_next], self.payoffs[0])

                        # if action 1

                        hist_new = np.c_[hist, np.array([0,1]) ]
                        w = get_weights(hist_new.shape[1], kappa)
                        s_other_next = self.BR(np.dot(hist_new, w), lambda_)
                        s_i_next = self.BR(s_other_next, lambda_)

                        eu_1 = expected_utility([np.array([0,1]), s_other_curr], self.payoffs[0]) + gamma*expected_utility([s_i_next, s_other_next], self.payoffs[0])

                    pred_i[:,k, l]  = softmax(lambda_* np.array([eu_0, eu_1]))


        pred_i_ = np.zeros((2, L))

        # for l in range(L):
        #     pred_i_[:, l] = np.dot(pred_i[:, :, l], alphas)
        # pred_i = pred_i_

        pred_i = np.squeeze(pred_i[:, k, :])
        
        return pred_i   


    def BR(self,  s_other, lambda_):
        '''
        Computes a best response

        Parameters:
            s_other : (np.Array) strategy profile of other player

        NOTE: this ONLY works with symetric payoffs and 2 actions
        '''
    
        #get EU of action 0
    
        s = [np.array([1, 0]), s_other]
        eu_0 = expected_utility(s, self.payoffs[0])

        # get EU of action 1
        s = [np.array([0, 1]), s_other]
        eu_1 = expected_utility(s, self.payoffs[0])

        return softmax(np.array([eu_0, eu_1])*lambda_)


class Temporal_Model:
    def __init__(self, h=1):
        self.payoffs = np.zeros([2, 2, 2])
        self.payoffs[0, :, :] = np.array([[0, -1],
                                          [1, -2]])
        self.payoffs[1, :, :] = np.array([[0, 1],
                                         [-1, -2]])
        self.alphas = None
        self.h = h # the history length


    def log_likelihood(self, params, dataset):
        '''
        Given the dataset and the fitted model, returns the log likelihood.

        NOTE: assumes each player has a fixed level
        
        Parameters:
            params : np.array of alpha_ks, the freq of that level in population and lambda_ for QBR
            dataset : list of plays, each play is an np array
        Returns:
            ll : loglikelihood of dataset

        '''
        alphas = params[0:3] # frequency of different risk types in population
        gammas = params[3:6] # parameter of risk types in population
        lambda_ = params[-1] 
      

        ll = 0
        for play in dataset: 
            for player in range(2):
                sum = 0
                for i, alpha_gamma in enumerate(alphas): # condition on a specific type of  player
                    traj = play[player]
                    other_traj  = play[1-player] # trajectory of other player
                    gamma = gammas[i]
                    pred_s = self.predict_traj(traj, other_traj, lambda_, gamma)
                    prob = p_traj(traj, pred_s) # probability of that trajectory
                    sum += alpha_gamma * (prob) # this could be 0, so small epsilon added
                ll += np.log(sum)

        return -ll # since we are minimizing the negative log likelihood


    def fit(self, dataset):
        params = np.zeros(7) # +2 since level 0 and the lambda parameter
    
        for i in range(3):
            params[i] = 1/3

        params[3] = 0.5
        params[4] = 1
        params[5] = 2
        params[6] = 1 # intial guess for lambda

        const_arr = np.zeros(7)
        for i in range(3):
            const_arr[i] = 1

       
        constraint = LinearConstraint(const_arr, lb=1, ub=1)
        bnds = [(0, 1) for x in range(3)]
       
        bnds.append((-1000, 1))
        bnds.append((1, 1))
        bnds.append((1, 1000))
        bnds.append((0, 1000)) # no bounds for lambda
       
        result = minimize(
            self.log_likelihood ,
            params, 
            args=(dataset),
            bounds=bnds, 
            constraints=constraint) 
      

        assert result.status == 0 # make sure the optimization was successful
    
        return result
        
    def predict_traj(self, traj, other_traj, lambda_, gamma): 
        '''
        I think about what action I can take

        what do I think you will play this round? 
        -> probably a best response to my last action

        what do I think you will play next round?
        -> probably a brest resonse to this aciton

        what will I play next round? 
            -> probably a best resonse to what I think you will play next round
        '''
        L = other_traj.shape[1]
        pred_i = np.zeros((2, L)) # level-k prediction for each stage game for i

        for l in range(L):

            # I could extend this to include history
            if l == 0:
                # if action 0
                s_other_next = self.BR(np.array([1,0]), lambda_)
                s_i_next = self.BR(s_other_next, lambda_)
                eu_0 = expected_utility([np.array([1,0]), np.array([0.5, 0.5])], self.payoffs[0]) + gamma*expected_utility([s_i_next, s_other_next], self.payoffs[0])

                # if action 1
                s_other_next = self.BR(np.array([0,1]), lambda_)
                s_i_next = self.BR(s_other_next, lambda_)
                eu_1 =  expected_utility([np.array([0,1]), np.array([0.5, 0.5])], self.payoffs[0]) + gamma*expected_utility([s_i_next, s_other_next], self.payoffs[0])
           
            else:
                # if action 0
                s_other_curr = self.BR(traj[:, l-1], lambda_)
                s_other_next = self.BR(np.array([1,0]), lambda_)
                s_i_next = self.BR(s_other_next, lambda_)
                eu_0 = expected_utility([np.array([1,0]), s_other_curr], self.payoffs[0]) + gamma*expected_utility([s_i_next, s_other_next], self.payoffs[0])

                # if action 1
                s_other_curr = self.BR(traj[:, l-1], lambda_)
                s_other_next = self.BR(np.array([0,1]), lambda_)
                s_i_next = self.BR(s_other_next, lambda_)
                eu_1 = expected_utility([np.array([0,1]), s_other_curr], self.payoffs[0]) + gamma*expected_utility([s_i_next, s_other_next], self.payoffs[0])

            pred_i[:, l]  = softmax(lambda_* np.array([eu_0, eu_1]))

        return pred_i   


    def BR(self,  s_other, lambda_):
        '''
        Computes a best response

        Parameters:
            s_other : (np.Array) strategy profile of other player

        NOTE: this ONLY works with symetric payoffs and 2 actions
        '''
    
        #get EU of action 0
    
        s = [np.array([1, 0]), s_other]
        eu_0 = expected_utility(s, self.payoffs[0])

        # get EU of action 1
        s = [np.array([0, 1]), s_other]
        eu_1 = expected_utility(s, self.payoffs[0])

        # return action with greater EU
        return softmax(np.array([eu_0, eu_1])*lambda_)
