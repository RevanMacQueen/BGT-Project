import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.special import softmax
from utils import expected_utility


class History_Mapping_Model:
    def __init__(self, h):
        self.h = h
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
        weights = params
        ll = 0
        for play in dataset: 
            for player in range(2):
                traj = play[player]
                other_traj  = play[1-player] # trajectory of other player
                
                pred_s = self.predict_traj(traj, other_traj, weights)  
               
                L = traj.shape[1] #length of trajectory
                for l in range(L):
                    idx = np.where(traj[:, l])
                    ll += np.log(pred_s[:, l][idx][0]+0.00001)


        return -ll # since we are minimizing the negative log likelihood


    def fit(self, dataset):
        num_features = len(self.get_features())
        
        
        num_params = 0
        for subhist_len in range(self.h, 0, -1): # for each length of history less than 
            num_params += num_features * (2 ** subhist_len)

        constraints = []
        start_sub_array = 0 
        for subhist_len in range(self.h, 0, -1): # for each length of history less than 
            for w in range(2**subhist_len): # for each batch of feature weights
                start_i = w * num_features + start_sub_array
                const_arr = np.zeros(num_params)

                for i in range(start_i, start_i+ num_features):
                    const_arr[i] = 1
                
                constraints.append(LinearConstraint(const_arr, lb=1, ub=1))

            start_sub_array += 2**subhist_len*num_features


       

        # additional weights for subhistories

        params = np.ones(num_params)/num_features
        bnds = [(0, 1) for i in range(num_params)]
    

        result = minimize(
            self.log_likelihood ,
            params, 
            args=(dataset),
            bounds=bnds,
            constraints=constraints) 
      
        assert result.status == 0 # make sure the optimization was successful
        return result



    def get_features(self):
        '''
        returns a set of linear_4 like features
        '''

        def strat(a):
            s = np.zeros(2)
            s[a] = 1
            return s

        max_min = strat(np.argmax(np.min(self.payoffs[0], axis=1)))
        max_max = strat(np.argmax(np.max(self.payoffs[0], axis=1)))
        min_max = strat(np.argmin(np.max(self.payoffs[0], axis=1)))
        eff = strat(np.argmax(np.max(self.payoffs[0] + self.payoffs[0].T, axis=1)))
        fair = strat(np.argmin(np.min( np.abs(self.payoffs[0]-self.payoffs[0].T), axis=1)))

        unif = np.ones(2)/2

        return [max_min, max_max, min_max, eff, fair, unif]


    def hist_to_str(hist):
        hist = hist.astype(str).tolist()
        return [''.join(elmt) for elmt in hist]


    def predict_traj(self, traj, other_traj, weights):
        '''
        Returns straetgy predictions against other player
        
        '''

        def get_weights(hist, weights, num_weights):

            hist_len = len(hist)
            start_sub_array = 0
            for i in range(self.h, hist_len, -1):
                start_sub_array += num_weights * 2**i 

            c = [2**(i) for i in range(len(hist))]
            start = start_sub_array +  num_weights*np.dot(c, hist)

            return weights[start: start+num_weights]

        play_len = traj.shape[1]
        pred_i = np.zeros((2, play_len))

        for l in range(play_len):
            if l == 0:
                pred_i[:, l] = np.ones(2)/2
               
            else:
                features = self.get_features()  
                w = get_weights(other_traj[0, max(l-self.h, 0):l], weights, len(features)) # occurance of c is sufficient history


                weighted_features = np.zeros((2, len(features)))
                for i in range(len(features)):
                    weighted_features[:, i] = features[i] * w[i]
              
              
                pred_i[:, l] = np.sum(np.array(weighted_features), axis = 1)
            
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


class Bounded_Memory_Map:
    def __init__(self, h):
        self.h = h
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
        num_features = len(self.get_features())
        M = params.reshape((num_features, num_features))
        ll = 0
        for play in dataset: 
            for player in range(2):
                traj = play[player]
                other_traj  = play[1-player] # trajectory of other player
                
                pred_s = self.predict_traj(traj, other_traj, M)  
               
                L = traj.shape[1] #length of trajectory
                for l in range(L):
                    idx = np.where(traj[:, l])
                    ll += np.log(pred_s[:, l][idx][0]+0.00001)


        return -ll # since we are minimizing the negative log likelihood


    def fit(self, dataset):
        num_features = len(self.get_features())

        # additional weights for subhistories

        params = np.random.random(num_features*num_features)
        bnds = [(0, None) for i in range(params.shape[0])]
    

        result = minimize(
            self.log_likelihood ,
            params, 
            bounds = bnds,
            args=(dataset)) 
      
        assert result.status == 0 # make sure the optimization was successful
        return result



    def get_features(self):
        '''
        returns a set of linear_4 like features
        '''

        def strat(a):
            s = np.zeros(2)
            s[a] = 1
            return s

        payoffs = self.payoffs[0]
        max_min = strat(np.argmax(np.min(payoffs, axis=1)))
        max_max = strat(np.argmax(np.max(payoffs, axis=1)))
        min_max = strat(np.argmin(np.max(payoffs, axis=1)))
        eff = strat(np.argmax(np.max(payoffs + payoffs.T, axis=1)))
        fair = strat(np.argmin(np.min( np.abs(payoffs - payoffs.T), axis=1)))

        unif = np.ones(2)/2

        return [max_min, max_max, min_max, eff, fair, unif]


    def hist_to_str(hist):
        hist = hist.astype(str).tolist()
        return [''.join(elmt) for elmt in hist]


    def predict_traj(self, traj, other_traj, M):
        '''
        Returns straetgy predictions against other player
        
        '''


        play_len = traj.shape[1]
        pred_i = np.zeros((2, play_len))

        for l in range(play_len):
            if l == 0:
                pred_i[:, l] = np.ones(2)/2
               
            else:
                p = self.get_probs(other_traj[:, max(l-self.h, 0):l]) # probability of history under each simple strat
                
                w = np.dot(M, p)
              
                features = self.get_features()  
                weighted_features = np.zeros((2, len(features)))

                sum_w = np.sum(w)
                for i in range(len(features)):
                    weighted_features[:, i] = features[i] * (w[i]/sum_w)
              
                pred_i[:, l] = np.sum(np.array(weighted_features), axis = 1)
            
        return pred_i   



    def get_probs(self, hist):
        '''
        Returns the probability of a history 
        '''

        def p_traj_single(a, s):
            '''
            Returns the probability of a sequence of actions
            a being sampled from strategy s.

            Paramters:
                a : (np.Array) 2 x n array of actions chosen
                s : (np.Array) 2 x 1 strategy
            '''
            L = a.shape[1]
            prod = 1
            for l in range(L):
                idx = np.where(a[:, l])
                prod *= s[idx][0]

            return prod

        features = self.get_features()
        probs = np.zeros(len(features))
        for i, s in enumerate(features):
            probs[i] = p_traj_single(hist, s)

        return probs


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