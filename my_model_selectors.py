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
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        # TODO implement model selection based on BIC scores

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        lowest_bic_score = float('inf')   # lowest wins
        best_hmm_model = None
        log_n = np.log(len((self.lengths)))
        num_features = self.X.shape[1]

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                # get HMM
                hmm_model = self.base_model(num_states).fit(self.X, self.lengths)

                # calculate log likelihood
                log_l = hmm_model.score(self.X, self.lengths)

                # calculate number of free params given transition probabilities and emission
                # probabilities

                # NOTE: below changed per review feedback
                # num_params = num_states + (num_states * (num_states - 1)) + (num_states * num_features * 2)
                # taking it that number of free parameters (p) = m^2 +2mf-1, where:
                #       m is num_states/components
                #       f is num_features
                num_params = (num_states ** 2) + (2 * num_states * num_features) - 1

                # change end

                # calculate BIC score => -2 * logL + p * logN
                bic_score = -2 * log_l + num_params * log_n

            except:
                continue

            # update lowest score/ best model
            if bic_score < lowest_bic_score:
                lowest_bic_score = bic_score
                best_hmm_model = hmm_model

        return best_hmm_model


class SelectorDIC(ModelSelector):
    '''
    select best model based on Discriminative Information Criterion
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        # TODO implement model selection based on DIC scores
        '''
        Unlike BIC, DIC takes goal into account; where
            DIC = log(P(X(i)) - (1/(M-1) * SUM(log(P(X(all but i)))
                        M is number of words
                        log(P(X(i)) = logL(i) -> likelihood
                        log(P(X(all but i) is logL(i) bar entry state -> anti-likelihood
                        anti-likelihood is averaged before being subtracted from likelihood
        '''
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        highest_dic_score = float('-inf')   # highest wins
        best_hmm_model = None

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                # get HMM
                hmm_model = self.base_model(num_states).fit(self.X, self.lengths)

                # calculate log likelihood for current model
                log_l = hmm_model.score(self.X, self.lengths)

                # init anti log_l, word_count for words
                anti_log_l_sum = 0.0
                word_count = 0

                for word in self.hwords:
                    if word != self.this_word:
                        word_x, word_lengths = self.hwords[word]
                        anti_log_l_sum += hmm_model.score(word_x, word_lengths)
                        word_count += 1      # will be M - 1 as this_word not counted

                # calculate DIC score (estimate) for current model
                dic_score = log_l - (anti_log_l_sum / float(word_count))

            except:
                continue

            # update lowest score/ best model
            if dic_score > highest_dic_score:
                highest_dic_score = dic_score
                best_hmm_model = hmm_model

        return best_hmm_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        # TODO implement model selection using CV
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        highest_cv_score = float('-inf')   # highest wins
        best_hmm_model = None

        # ensure there are enough sequences to fold
        if len(self.sequences) < 2:
            return None
        else:
            k_fold = KFold(n_splits=2)

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            log_l_sum = 0
            states_counter = 0

            # Iterate sequences
            for cv_train_idx, cv_test_idx in k_fold.split(self.sequences):
                try:
                    # get train and test sequences
                    cross_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                    cross_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                    # get HMM for training sequence
                    hmm_model = self.base_model(num_states).fit(cross_train, lengths_train)

                    # calculate log likelihood for current model (test)
                    log_l = hmm_model.score(cross_test, lengths_test)

                    # increment divisor for cv score
                    states_counter += 1

                except:
                    log_l = 0

                log_l_sum += log_l

            # Calculate CV score
            cv_score = log_l_sum if states_counter == 0 else (log_l_sum / states_counter)

            # update lowest score/ best model
            if cv_score > highest_cv_score:
                highest_cv_score = cv_score
                best_hmm_model = hmm_model

        return best_hmm_model


if __name__=="__main__":
    from asl_test_model_selectors import TestSelectors
    import unittest
    suite = unittest.TestLoader().loadTestsFromModule(TestSelectors())
    unittest.TextTestRunner().run(suite)