"""
Estimation of parameters for shift-reduce parser.

This is used on Natural Stories Corpus data.

"""

import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")

import pymc3 as pm
from pymc3 import Gamma, Normal, HalfNormal, Deterministic, Uniform, find_MAP,\
                  Slice, sample, summary, Metropolis, traceplot
from pymc3.backends.base import merge_traces
from pymc3.backends import Text
from pymc3.backends.sqlite import load
import theano
import theano.tensor as tt
from mpi4py import MPI
from theano.compile.ops import as_op

print(theano.__version__)

from matplotlib import pyplot as plt

import run_parser as rp
import utilities as ut

WORDS = rp.WORDS
LABELS = rp.LABELS
ACTIVATIONS = rp.ACTIVATIONS
activations = ut.load_file(ACTIVATIONS, sep=",")

SENTENCES = rp.SENTENCES
parser = rp.parser

WORD_RTs = "processed_wordinfo.tsv"
word_rts = ut.load_file(WORD_RTs, sep="\t")

used_word_rts = word_rts[word_rts.zone.isin([0])] # start an empty word_rts

# remove sentences at the beginning of the stories

activations = activations[~activations.sentence_no.isin(range(1, 11))]
activations = activations[~activations.sentence_no.isin(range(58, 68))]

# collect RTs that we will model
for item in set(activations.item.tolist()):
    small_word_rts = word_rts[word_rts.item.isin([item])]
    used_word_rts = pd.concat([used_word_rts, small_word_rts[small_word_rts.zone.isin(activations[activations.item.isin([item]) & ~activations.position.isin([1])].zone.tolist())]])

used_word_rts[used_word_rts.meanItemRT > 500].meanItemRT = np.NaN

used_rts = used_word_rts.meanItemRT.to_numpy()

#pd.set_option("display.max_rows", None, "display.max_columns", None)
#print(used_word_rts.word, flush=True)

RANKS = {1: range(11, 17), 2: range(17, 24), 3: range(24, 33), 4: range(33, 42), 5: range(42, 52), 6: range(52, 58), 7: range(68, 73), 8: range(73, 79), 9: range(79, 86), 10: range(86, 90), 11: range(90, 95)} #what sentences should ranks use; the max sentence is 94

def run_self_paced_task():
    """
    Run loop of self paced reading.
    """

    stimuli_csv = ut.load_file(SENTENCES, sep=",") #sentences with frequencies
    words = ut.load_file(WORDS, sep="\t")
    labels = ut.load_file(LABELS, sep="\t")
    DM = parser.decmem.copy()

    #prepare dictionaries to calculate spreading activation
    word_freq = {k: sum(g["FREQ"].tolist()) for k,g in words.groupby("WORD")} #this method sums up frequencies of words across all POS
    label_freq = labels.set_index('LABEL')['FREQ'].to_dict()

    sent_nrs = RANKS[rank]

    while True:

        received_list = np.empty(2, dtype=np.float)
        comm.Recv([received_list, MPI.FLOAT], source=0, tag=rank)
        if received_list[0] == -1:
            break
        parser.model_parameters["latency_factor"] = received_list[0]
        parser.model_parameters["latency_exponent"] = received_list[1]
        parser.model_parameters["rule_firing"] = 0.033

        final_times_in_s = np.array([])

        test_len = 0

        for sent_nr in sent_nrs:
            used_activations = activations[activations.sentence_no.isin([str(sent_nr)]) & activations.record_RTs.isin(["yes"])]
            subset_stimuli = stimuli_csv[stimuli_csv.item.isin([str(sent_nr)]) & stimuli_csv.word.isin(used_activations.word.tolist())]
            sentence = used_activations.word.tolist() #words used 
            pos = subset_stimuli.function.tolist() #pos has to be taken from subset_stimuli; it's not used in the actual computation

            test_len += len(sentence)
            try:
                final_times_in_s = np.concatenate((final_times_in_s, rp.read(parser, sentence=sentence, pos=pos, activations=used_activations,\
                    word_freq=word_freq, label_freq=label_freq, weight=10,\
                    decmem=DM, lexical=True, syntactic=True, visual=False, reanalysis=False, prints=False)[1:])) #concatenate final_times_in_s and remove the first word in the sentence
            except:
                final_times_in_s = np.concatenate((final_times_in_s, np.array([10 for _ in sentence[1:]])))

        final_times_in_s = np.concatenate( (np.array([len(final_times_in_s)]), final_times_in_s) ) #store the length of the send array in the first position
        comm.Send([np.array( final_times_in_s ), MPI.FLOAT], dest=0, tag=1) #len_sen - number of items

@as_op(itypes=[tt.dscalar, tt.dscalar], otypes=[tt.dvector])
def actrmodel_latency(lf, le):

    sent_list = np.array([lf, le], dtype = np.float)

    print("SENT LIST", sent_list)

    #get slaves to work
    for i in range(1, comm.Get_size()):
        comm.Send([sent_list, MPI.FLOAT], dest=i, tag=i)

    #print("STARTING", sent_list)

    RT_ms = np.array([])

    for i in range(1, N_GROUPS+1):
        #receive list one by one from slaves - each slave - 8 sentences
        received_list = np.empty(len(used_rts), dtype=np.float) #initialize a large-enough empty array
        comm.Recv([received_list, MPI.FLOAT], i, tag=1)
        np.set_printoptions(threshold=sys.maxsize)
        received_list = received_list[1:int(received_list[0]+1)] #match the length (remove everything beyond the length of the sent list) +1 or +2
        #print("RECEIVED LIST", received_list, flush=True)
        RT_ms = np.concatenate((RT_ms, 1000*received_list))

    return RT_ms

comm = MPI.COMM_WORLD
rank = int(comm.Get_rank())

N_GROUPS = comm.Get_size() - 1 #Groups used for simulation - one less than used cores

if rank == 0: #master
    NDRAWS = int(sys.argv[1])
    CHAIN = int(sys.argv[2])

    # assume testvalues at mean
    testval_le, testval_lf = 0.5, 0.2

    # or try to load past values
    try:
        past_simulations = ut.load_file("natural_stories_20_sen_removed"+str(CHAIN)+"/chain-0.csv", sep=",")
    except:
        pass
    else:
        testval_lf = past_simulations['lf'].iloc[-1]
        testval_le = past_simulations['le'].iloc[-1]

    parser_with_bayes = pm.Model()

    with parser_with_bayes:
        # priors for latency
        lf = Gamma('lf', alpha=2, beta=10, testval=testval_lf)
        le = Gamma('le', alpha=2,beta=4, testval=testval_le)
        # latency likelihood -- this is where pyactr is used
        pyactr_rt = actrmodel_latency(lf, le)
        predicted_mu_rt = Deterministic('predicted_mu_rt', pyactr_rt)
        rt_observed = Normal('rt_observed', mu=predicted_mu_rt, sd=20, observed=used_rts)
        # we start the sampling
        step = Metropolis()
        db = Text('natural_stories_20_sen_removed' + str(CHAIN))
        trace = sample(draws=NDRAWS, trace=db, chains=1, step=step, init='auto', tune=1)
        traceplot(trace)
        plt.savefig("natural_stories_20_sen_removed" + str(CHAIN) + ".pdf")
        plt.savefig("natural_stories_20_sen_removed" + str(CHAIN) + ".png")
        #db.close()

    #stop slaves in their work
    sent_list = np.array([-1, -1, -1], dtype = np.float)
    for i in range(1, comm.Get_size()):
        comm.Send([sent_list, MPI.FLOAT], dest=i, tag=i)

else: #slave
    print(rank)
    run_self_paced_task()
