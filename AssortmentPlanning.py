# -*- coding: utf-8 -*-
'''
    File name: staticMNL.py
    Author: Chua Yonzheng Jerry (@lovelydinner)
    Date created: 9/20/2020
    Date last modified: 9/28/2020
    Python Version: 3.7.6
'''


import pandas as pd
import numpy as np
import itertools

class MNL():
    '''
    Class to determine the best assortment of items using
    Multinomial Logit Discrete Choice Model

    Algorithm covered in Rusmevichientong et al. 2010

    '''

    def __init__(self, mean_utility, profits):
        '''
        mean_utility : mean utility (shifted to set outside option to 0), 
                        includes outside option, outside option is last column. 
        cardinality : maximum number of products that can be presented

        cust_pref vector = (e^(mu_i)) for i = {1,...,N}
                         = 1 for i =0

        profit vector = 0 for i = 0
                      = profit for others
        '''
        self.utility = np.asarray(mean_utility)
        self.cust_pref = np.exp(self.utility)
        self.profits = np.asarray(profits)

        assert(self.profits.shape == self.cust_pref.shape)

    def find_intersections(self):
        '''
        finds all the intersection points and sorts them in ascending order

        I(i_t, j_t) are the x coordinates of intersection points to sort

        lambda = I(i,j) = \frac{v_iw_i - v_jw_j}{v_i - v_j}

        to enumerate A(lambda) for all lambda - it is sufficient to enumerate all intersections point of lines

        returns array of all zipped (i,j) and intersection pair
        '''
        intersections = []
        pairs = []

        #O(N^2)
        for i in range(len(self.profits)-1):
            for j in range(i+1, len(self.profits)):
                pairs.append( (i,j) )
                numerator = self.profits[i]*self.cust_pref[i] - self.profits[j]*self.cust_pref[j]
                denominator = self.cust_pref[i] - self.cust_pref[j]
                intersections.append(numerator/denominator)

        inter_pairs = np.asarray(list(zip(pairs,intersections)))
        index_sorting_intersections = np.argsort(inter_pairs[:,1]) 
        inter_pairs = inter_pairs[index_sorting_intersections]

        return inter_pairs

    def staticMNL(self, intersections, constraint):
        '''
        performs staticMNL algorithm, returns collection of assortments

        recall that iterating through intersections is sufficient for all lambda

        \sigma^0 = sorted v in descending order

        for intersection:
            update sigma - transpose i and j for I(i,j)
            update(new) G - top C
            update B - if i==0, add j
            update A - G-B

        return A (outside option is value 0, everything else should +1)


        input:
            intersections: sorted intersections :: list of [(i,j), I(i,j)] \forall interactions
            constraint: constraint for number of items in assortment
        '''

        #initialisation
        A = []
        G = set()
        B = set()
        sigma = np.argsort(-self.cust_pref) #descending order 

        G.update(sigma[:constraint])
        A.append(sigma[:constraint].tolist())

        #iterate interactions
        for t,zipped in enumerate(intersections):
            if zipped[0][1] != len(self.profits) :
                #swap order
                swap_values = zipped[0]
                swap_index = np.argwhere(np.isin(sigma, swap_values)).flatten()
                swap_1, swap_2 = sigma[swap_index[0]], sigma[swap_index[1]]
                sigma[swap_index[0]], sigma[swap_index[1]] = swap_2, swap_1
            else:
                B.add(zipped[0][1])
                print('swapped')

            #update G each iteration
            G.clear()
            G.update(sigma[:constraint])

            #if anything in B is in G, remove
            A_t = G - B
            if A_t:
                A.append(list(A_t))

        # A = np.array(A) + 1

        return A

    def tabulate_profits(self, assortments):
        '''
        tabulate profits for each optimal assortment

        [ [(assortment1), profit_assortment1], [(assortment2),profit_assorment2] ......]

        f(s) = \frac{\sum_{j \in S} w_jv_j}{1 + \sum_{j \in S} v_j}

        where s represents the items in assortment

        assortments does not contain the 0 indexed outside option

        input:
            assortments: list of all optimal assortments

        '''
        profits_ = []
        for assortment in assortments:
            v = self.cust_pref[assortment]
            w = self.profits[assortment]
            numerator = np.dot(v,w)
            denominator = 1 + np.sum(v)
            profits_.append(numerator / denominator)

        assort_profits = list(zip(map(tuple,assortments), profits_))
        
        return assort_profits

    def max_profit(self,all_profits):
        '''
        returns the assortment that gives the maximum profit
        
        outside option is at index 0

        input:
            all_profits: profits of all the optimal assortments in format of []
        '''
        
        profs = np.array(all_profits)
        max_profs_index = np.argmax(profs[:,1])
        
        return profs[max_profs_index]

class ProbitModel(MNL):
    '''
    class to determine best assortments using
    Multinomial Probit Discrete Choice model
    '''
    
    def __init__(self, utility, profits):
        '''
        input:
            initial utility matrix (represented by ratings) - includes outside option
            profits of each item
        '''
        self.initial_utility = utility
        self.profits = np.asarray(profits)
    
    def simulate(self,k=500):
        '''
        simulates/generates k number of samples following
        the multivariate distribution.

        U = V + \epsilon 
        epsilon ~ N(0, \Sigma), and \Sigma represents the correlation
        of items
        let V be the mean utility of items
        
        input:
            k: number of samples to generate, default = 500
        '''
        self.corr_mat = self.initial_utility.corr() #requires utility to be df
        self.V = self.initial_utility.mean(axis = 0)
        generate_samples = np.random.multivariate_normal(self.V, self.corr_mat, size = k)

        return generate_samples

    def empirical_probabilities(self, data):
        '''
        Finds the empirical choice probabilities of each item(returns index of item)

        input:
            data: simulated data (format : user x item of utilities)
        '''
        df = pd.DataFrame(data)
        #determine which item(index) has highest utility
        max_utility = df.apply(np.argmax, axis =1)
        probabilities = max_utility.value_counts() / len(max_utility)
        probabilities = probabilities.reindex(df.columns, fill_value=0)
        self.cust_pref = probabilities
    
        return probabilities

    def get_assortments(self, samples, constraint):
        '''
        exhaustive search, to get all combinations of assortments

        returns list of set of combinations available

        input:
            constraint: represents the maximum assortment size
        '''
        samples_ = pd.DataFrame(samples)
        combis = []
        for k in range(1, constraint):
            combis.extend( list(map(list, itertools.combinations( pd.DataFrame(samples).columns, k) )) )
        return combis
    