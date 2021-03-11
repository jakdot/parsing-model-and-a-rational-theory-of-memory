"""
The script prepares sentences for parameter estimation. It does various clean-up and adds word frequencies.

Output: corpus_sentences.csv
"""

import pandas as pd
import numpy as np
import math
from nltk import word_tokenize

SEC_IN_YEAR = 365*24*3600
SEC_IN_TIME = 15*SEC_IN_YEAR
USED_WORDS = 112.5

from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import BracketParseCorpusReader
from nltk.tree import Tree

# treebank below can be used to test on selected files
# you have to specify your own folder in which natural_stories file would be located
# the folder has to be found by the nltk utility LazyCorpusLoader
# the folder should include the file all-parses.txt.penn from the Natural Stories Corpus
treebank = LazyCorpusLoader(
    'treebank/natural_stories', BracketParseCorpusReader, r'.*\.penn',
    tagset='wsj', encoding='ascii') #to test on files

# From prepare_sentences.py
def get_frequency(word, df):
    """
    Finds the frequency, based on the dataframe df.
    """
    word_freq = sum(df[df.Word.isin([word])].Frequency)
    if word_freq == 0:
        word_freq = 1
    return float(word_freq)

# From prepare_sentences.py
def get_freq_array(word_freq, real_freq=True):
    """
    Finds the frequency, based on the dataframe df.
    """
    time_interval = SEC_IN_TIME / word_freq
    if real_freq:
        return np.arange(start=-time_interval, stop=-(time_interval*word_freq)-1, step=-time_interval)
    else:
        return np.array([0])


def prepare_sentences(treebank, freq_csv, collected):
    '''
    Covert pandas dataframe with sentences and pos to correct
    sentences.csv file.
    '''
    item = 0
    for parsed, sentence, tagged in list(zip(treebank.parsed_sents(), treebank.sents(), treebank.tagged_sents())):
        tobe_removed = {i for i in range(len(tagged)) if tagged[i][1] == "-NONE-"}
        sentence = [sentence[i].lower() for i in range(len(sentence)) if i not in tobe_removed]
        pos = [tagged[i][1] for i in range(len(tagged)) if i not in tobe_removed]
        for position, word in enumerate(sentence):
            freq = get_frequency(word.lower(), freq_csv)
            word_freq = freq * USED_WORDS/100 
            collected["word"].append(word.lower())
            collected["function"].append(pos[position])
            collected["item"].append(str(item+1))
            collected["position"].append(position+1)
            collected["freq"].append(word_freq)            
            freq_array = 0-get_freq_array(word_freq)
            activation = math.log(np.sum(freq_array ** (-0.5) ))
            collected["activation"].append(activation)
        item += 1
    return collected
        


collected = {"position": [], "word": [], "freq": [], "function": [], "item": [], "activation": []}

freq_csv = pd.read_csv("all.csv", index_col=None, header=0, sep=" ")

final_df = prepare_sentences(treebank, freq_csv, collected)

final_df = pd.DataFrame.from_dict(final_df)

final_df.to_csv("corpus_sentences.csv", sep=",", encoding="utf-8", index=False)
