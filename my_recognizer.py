import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """

    # TODO implement the recognizer

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # iterate through test set
    for test_x, test_x_length in list(test_set.get_all_Xlengths().values()):
        words_scored = {}
        # test against each word-model (score)
        for word, hmm_model in models.items():
            try:
                words_scored[word] = hmm_model.score(test_x, test_x_length)
            except:
                continue

        probabilities.append(words_scored)

        # append highest probability word to guesses
        guesses.append(max(words_scored.items(), key=lambda word: word[1])[0])

    return probabilities, guesses

if __name__=="__main__":
    from asl_test_recognizer import TestRecognize
    import unittest
    suite = unittest.TestLoader().loadTestsFromModule(TestRecognize())
    unittest.TextTestRunner().run(suite)
