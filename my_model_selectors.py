import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """

        # warnings.filterwarnings("ignore", category=DeprecationWarning)
        max_score = float('inf')
        best_model = None
        for n_states in range(self.min_n_components,self.max_n_components + 1):
            model = self.base_model(n_states)
            if model != None:
                try:
                    p = model.n_features
                    N = np.sum(self.lengths)
                    logl = model.score(self.X, self.lengths)
                    bic = -2 * logl * p * math.log(N)

                    if bic < max_score:
                        max_score = bic
                        best_model = model
                except:
                    if self.verbose:
                        print("Not able to score for model on {} in BIC model selection.  "
                              "Skipping model with {} hiddent states".format(self.this_word,n_states))
                    continue
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''


    def compute_sum_of_likelihoods(self,model):
        likelihoods = []
        for key in self.hwords:
            X, lengths = self.hwords[key]
            likelihoods.append(model.score(X,lengths))
        return np.sum(likelihoods)
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        """
        M = number of training words
        Need to compute the log-likihoods of all of the other words besides the current word
        """
        max_score = float('inf')
        best_model = None

        for n_states in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(n_states)
            if model != None:
                try:
                    sum_of_likelihoods = self.compute_sum_of_likelihoods(model)
                    logL = model.score(self.X,self.lengths)
                    M = len(self.hwords)

                    #To get the proper sum, we will subtract the logL from the sum_of_likelihoods value
                    likelihood_sum_value = sum_of_likelihoods - logL
                    DIC = logL - (1/M-1) * likelihood_sum_value

                    if DIC < max_score:
                        max_score = DIC
                        best_model = model
                except:
                    if self.verbose:
                        print("Not able to score for model on {} in DIC model selection. "
                              "Skipping model with {} hidden states".format(self.this_word,n_states))
                    continue

        return best_model




class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_avg = float("-inf")
        best_model = None
        for n_states in range(self.min_n_components, self.max_n_components + 1):
            logl_values = []
            model = self.base_model(n_states)
            if model != None:
                try:
                    split_method = KFold()
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        X,lengths = combine_sequences(cv_train_idx,self.sequences)
                        logl_values.append(model.score(X,lengths))

                    avg_log = np.mean(logl_values)
                    if avg_log > best_avg:
                        best_avg = avg_log
                        best_model = model
                except:
                    if best_model == None:
                        best_model = model
                    if self.verbose:
                        print("Not able to score for model on {} in CV model selection. "
                                "Skipping model with {} hidden states".format(self.this_word,n_states))
                    continue
        return best_model