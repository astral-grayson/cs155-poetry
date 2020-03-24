########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np
from tqdm import tqdm

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively. 
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]
        
        probs = np.array(probs)
        obs = np.array(self.O)
        tran = np.array(self.A)
      
        # start
        probs[1, :] = np.array(self.A_start) * obs[:,x[0]]
        seqs[1][:] = [str(i) for i in range(self.L)] # lmao is this right
        
        # n is the lengths
        for n in range(2, M+1):
            # m is the state
            for m in range(self.L):
                prob_list = obs[m, x[n-1]] * probs[n-1, :] * tran[:, m]
                index = np.argmax(prob_list)

                probs[n, m] = prob_list[index]
                seqs[n][m] = seqs[n-1][index] + str(m)
                
        max_index = np.argmax(probs[M, :])
        max_seq = seqs[M][max_index]
        
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        
        alphas = np.array(alphas)
        obs = np.array(self.O)
        tran = np.array(self.A)
        
        alphas[1, :] = obs[:, x[0]] * np.array(self.A_start)
        
        for i in range(1, M):
            for m in range(self.L):
                alphas[i + 1, m] = obs[m, x[i]] * np.dot(alphas[i, :], tran[:, m])
            
            if normalize:
                alphas[i + 1, :] = alphas[i + 1, :] / np.sum(alphas[i + 1, :])

        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        obs = np.array(self.O)
        tran = np.array(self.A)
        
        betas = np.array(betas)
        
        betas[M, :] = np.ones(self.L)
        
        for i in range(M-1, 0, -1):
            for m in range(self.L):
                betas[i, m] = np.sum(betas[i+1, :] * tran[m, :] * obs[:, x[i]])
            
            if normalize:
                betas[i, :] = betas[i, :] / np.sum(betas[i, :])

        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        
        for a in range(self.L):
            for b in range(self.L):
                numer_A = 0
                denom_A = 0
                
                # find when y^i = a and y^i+1 = b
                for j in range(len(Y)):
                    for i in range(len(Y[j]) - 1):
                        if Y[j][i] == a:
                            denom_A += 1
                            if Y[j][i+1] == b:
                                numer_A += 1
                
                if denom_A == 0:
                    self.A[a][b] = 0
                else:
                    self.A[a][b] = numer_A / denom_A

        # Calculate each element of O using the M-step formulas.
        
        for z in range(self.L):
            for w in range(self.D):
                numer_O = 0
                denom_O = 0
                
                # find when y^i = z and x^i = w
                for j in range(len(Y)):
                    for i in range(len(Y[j])):
                        if Y[j][i] == z:
                            denom_O += 1
                            if X[j][i] == w:
                                numer_O += 1
                
                if denom_O == 0:
                    self.O[z][w] = 0
                else:
                    self.O[z][w] = numer_O / denom_O


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''
        
        N = len(X)
        
        #for n in tqdm(range(N_iters)):
        for n in range(1):
            # set numerator and denom arrays
            A_num = np.zeros([self.L, self.L])
            A_den = np.zeros([self.L, self.L])
            O_num = np.zeros([self.L, self.D])
            O_den = np.zeros([self.L, self.D])
        
            for j in range(N):
                seq = np.array(X[j]) # for each seq in X
                M = len(seq)
                
                # discard 0th index bc we are indexing at 1 :(
                alphas = self.forward(seq, normalize=True)[1:] 
                betas = self.backward(seq, normalize=True)[1:]
                
                # find and store single margin of P(y = a | x)
                margin = np.zeros([M, self.L])
                for i in range(M):
                    margin[i, :] = alphas[i, :] * betas[i, :]
                    
                    if np.sum(margin[i, :]) != 0:
                        margin[i, :] = margin[i, :] / np.sum(margin[i, :])
                
                joint = np.zeros([self.L, self.L, M])
                
                # find and store joint margin of P(y^i = a, y^i+1 = b | x)
                for i in range(M - 1):
                    for a in range(self.L):
                        for b in range(self.L):
                            joint[a, b, i] = alphas[i, a] * self.A[a][b] * self.O[b][seq[i+1]] * betas[i+1, b]
                    
                    if np.sum(joint[:, :, i]) != 0:
                        joint[:, :, i] = joint[:, :, i] / np.sum(joint[:, :, i])
                
                # find A's and O's by summing over i for a,b/a,w dimensions
                for a in range(self.L):
                    A_den[a, :] += np.sum(margin[:M-1, a])
                    O_den[a, :] += np.sum(margin[:, a])
                    for b in range(self.L):
                        A_num[a, b] += np.sum(joint[a, b, :])
                    for w in range(self.D):
                        x_indices = np.where(seq == w)
                        O_num[a, w] += np.sum(margin[x_indices, a])
            
            # update A and O
            self.A = A_num / A_den
            self.O = O_num / O_den
            


    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []
        
        start = np.random.uniform(0, 1, self.L)
        start /= np.sum(start)
        start_state = np.random.choice(np.arange(self.L), p = start)
        states.append(start_state)
        
        word = np.random.choice(np.arange(self.D), p = self.O[start_state][:])
        emission.append(word)
        
        for i in range(0, M-1):
            state = np.random.choice(np.arange(self.L), p = self.A[states[i]][:])
            states.append(state)
            
            word = np.random.choice(np.arange(self.D), p = self.O[states[i]][:])
            emission.append(word)
            
        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
