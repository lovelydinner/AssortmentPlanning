import numpy as np
import pandas as pd
import random


class MarkovChoiceModel():
    '''
    Craete class for assortment planning for markov chain
    '''
    def __init__(self, ratings):
        '''
        ratings should be a dataframe
        '''
        self.ratings = np.array(ratings)
        self.names = ratings.columns.tolist()
    
    def arrival_proba(self, data):
        '''
        determine the arrival probabilities from entire dataset

        requires:
            ratings value for entire dataset: np array

        return probability of each item being the most preferable product
        '''
        max_ratings = np.argmax(data, axis = 1)
        
        indexes, counts = np.unique(max_ratings, return_counts = True)

        full_index = list(range(data.shape[1]))
        check_in = np.isin(full_index, indexes)
        if any(~check_in):
            for index, i in enumerate(check_in):
                if not i:
                    counts = np.insert(counts, index, 0)
        return counts/counts.sum() 

    def transition_proba(self, data, outside_ratings):
        '''
        find transition probabilities

        Note that \rho_ij refers to the probabilitty of substitutinfg to j from i given that the product i is the most preferable but not available.
        This means that i cannot equal to j since it means it transitions into something that is not available. 

        input:
            outside_ratings: (customer,1) of outside option utilities

        outside option is last index

        if any one of the options are the outside option, it will stay in that state
        
        return transition probabilities and dataset
        '''
        rho = np.zeros((self.ratings.shape[1] + 1, self.ratings.shape[1] + 1))
        base_proba = self.arrival_proba(data)
        
        self.arrival = base_proba

        #transition probabilities are not the same both ways
        for i in range(rho.shape[0]):
            #remove i from dataset
            removed = np.delete(data, i , axis = 1)
            prob_removed = self.arrival_proba(removed) #get probability of each choice given i is removed
            for j in range(rho.shape[1]):
                if (i!=j) and (i!=rho.shape[1]-1):
                    #track which index removed
                    if i<j:
                        j_interested = j-1
                    else:
                        j_interested = j
                    numerator = prob_removed[j_interested] - base_proba[j]
                    denominator = base_proba[i]
                    if denominator == 0: #if base probability of entering i is 0, this is simply the probability where u reenter the system at j
                        rho[i,j] = base_proba[j]
                    else:
                        rho[i,j] = numerator/denominator
                elif (i==j) and (i==(rho.shape[1]-1)):
                    rho[i,j] = 1

        return rho


    def assortment_proba(self, assortment_indexes, rho):
        '''
        determine transition probabilities within the assortment, including outside option
        
        keep only items in the set (assortment)

        input:
            data: dataframe(needed to names of columns), inlcusive of outside option
            assortment_indexes: set of indexes(names) for product
            rho: base transition probability matrix for all N

        return new_rho, index numbers corresponding to the assortment
        '''

        new_rho = rho.copy()
        index_assortment = set()
        for index_i, name_i in enumerate(self.names):
            for index_j, name_j in enumerate(self.names):
                if name_i in assortment_indexes: #absorbing state
                    index_assortment.add(index_i)
                    if name_j!=name_i:
                        new_rho[index_i,index_j] = 0 #can't come out
                    else:
                        new_rho[index_i,index_j] = 1 #absorbing, only goes back to itself

        index_assortment.add(rho.shape[0] -1) #outside option
        index_assortment = sorted(list(index_assortment))
        return new_rho, index_assortment

    def choice_proba(self, arrival_prob, assortment_prob, index_assortment):
        '''
        determine choice probabilities, requires rearrangement of transition submatrices

        input:
            base arrival probabilities : numpy array of probabilities
            assortment probabilities: numpy array of probabilities within the assortment
            index_assortment: list of exact index in entire dataset of assortment, includes outside option, take from assortment_proba

        returns choice probabilities for the assortment
        '''
        I = assortment_prob[index_assortment][:, index_assortment]
        mask = np.isin(list(range(self.ratings.shape[1] + 1)), index_assortment)
        B = assortment_prob[~mask][:,index_assortment]
        O = assortment_prob[index_assortment][:,~mask]
        C = assortment_prob[~mask][:,~mask]
        
        inversed = np.linalg.inv(np.eye(C.shape[0])-C)
        bottomquad = np.dot(inversed, B)

        front_columns = np.vstack((I, bottomquad))
        back_columns = np.vstack((O, np.zeros(shape = (C.shape[0], C.shape[1]))))
        full_mat = np.hstack((front_columns,back_columns))
        choice_probabilities = np.dot(arrival_prob.T, full_mat)


        return np.array(choice_probabilities)


    def fit(self, assortment_indexes,  outside_option = None):
        '''
        fit the data and get choice probabilities for assortment proposed

        input:
            assortment_indexes refers to the data
            outside_option: None, determines if outside option utility is input by self or generated by algorithm. False means generated by algo


        returns choice probability of the assortment

        '''

        #generate outside option utility
        if not outside_option:
            max_ = np.max(self.ratings)
            min_ = np.min(self.ratings)
            length = max_ - min_
            if np.issubdtype(self.ratings[0,0], int):
                outside_option = np.random.randint(length, size = (self.ratings.shape[0],1)) + 1
            else:
                outside_option = length*np.random.random_sample(size = (self.ratings.shape[0],1)) - length

        data = self.ratings.copy()
        #ensure that outside ratings has the same 
        assert outside_option.shape == (data.shape[0],1) , 'Outside Option ratings have to have a shape of (number of customers, 1)'
        data = np.hstack((data, outside_option)) #outside option is last index
        self.names.append('outside')
        
        rho = self.transition_proba(data, outside_option)
        self.updated_rho, index_assortment = self.assortment_proba(assortment_indexes, rho)
        print(index_assortment)
        self.choice_probabilities = self.choice_proba(self.arrival, self.updated_rho, index_assortment)

class AssortmentPlanning():
    '''
    class to solve the assortment plannign problem

    For any assortment planning problem, we seek to find the assortment S that maximises the profit:
    This is equivalent to finding the arrival probability multiplied by the expected revenue a customer arrives in state i when the offer set is S

    This assortment planning solution is detailed in the A Markov chain approximation to Choice Modelling
    '''

    def __init__(self, rho, assortment, choice, revenues):
        '''
        stores:
            rho: transition matrix (np.array)
            assortment: assortment id (set of product ID)
            choice : choice probabilities (np.array)
            revenues/profits of each individual item (np.array)
        '''
        self.assortment = assortment
        self.choice_prob = choice
        self.revenue = revenues.reshape(-1,1) #reshape into one column 
        self.rho = rho

    def compute_revenue(self):
        '''
        algorithm to determine g

        g_i = max_si g_i(s_i)

        converges in polynoimal number of iterations

        returns g - the maximum expected revenue starting from each state

        '''
        #intitialisation
        t = 0
        delta = 1
        #only 2 time states of g required - current and the one before
        g = self.revenue.copy()
        while(delta > 0):
            diag = np.diag(np.diag(self.rho))
            not_included = np.dot(diag, g)
            expected_revenue = np.dot(self.rho, g)
            for_comparison = expected_revenue - not_included
            proposed_g = np.max(np.hstack((g, for_comparison)), axis = 1)
            delta = np.sum(proposed_g - g)
            g = proposed_g
        return g
        


    
        


        
    
        




