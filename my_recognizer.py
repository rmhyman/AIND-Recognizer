import warnings
from asl_data import SinglesData

def get_word_index(word,word_list):
    return word_list.index(word)
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
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    ##test_set.wordlist -> List of words
    for i,word in enumerate(test_set.wordlist):
        X,lengths = test_set.get_item_Xlengths(i)
        best_guess,guess_word = float("-inf"), None
        tmp = {}
        for key in models:
            try:
                tmp[key] = models[key].score(X,lengths)
                if tmp[key] > best_guess:
                    best_guess = tmp[key]
                    guess_word = key
            except:
                tmp[key] = float("-inf")
                continue
        probabilities.append(tmp)
        guesses.append(guess_word)
    return probabilities,guesses

    # return probabilities, guesses

