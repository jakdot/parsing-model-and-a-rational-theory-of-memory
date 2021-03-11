"""
Runs parser based on previously executed actions.
"""

import pandas as pd
import simpy
import re
import sys
import numpy as np

import pyactr as actr

from parser_rules import parser
from parser_dm import environment
from parser_dm import SENTENCES
import parser_rules
from parser_rules import parser
import utilities as ut
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import BracketParseCorpusReader
from nltk.tree import Tree


WORDS = "words.csv"
LABELS = "labels.csv"
ACTIVATIONS = "activations.csv"

actions, word_freq, label_freq = None, None, None

def visual_effect(word, visual=True):
    if visual:
        return len(word)
    else:
        return 5

def read(parser, sentence=None, pos=None, activations=None, word_freq=word_freq, label_freq=label_freq, strength_of_association={}, weight=1, decmem={}, lexical=True, visual=True, syntactic=True, reanalysis=True, prints=True):
    """
    Read a sentence.

    :param sentence: what sentence should be read (list).
    :param pos: what pos should be used (list, matching in length with sentence).
    :param activations: dataframe of activations
    :param condition: name of condition (has to match with what is in ACTIVATIONS)
    :param sent_nr: sent_nr, usually a number (has to match with what is in ACTIVATIONS)
    :param lexical - should lexical information affect reading time?
    :param visual - should visual information affect reading time?
    :param syntactic - should syntactic information affect reading time?
    :param reanalysis - should reanalysis of parse affect reading time?
    """
    if prints:
        print(activations)
    parser.set_decmem(decmem) 
    parser.decmem.activations = decmem.activations
    if prints:
        print(sentence)

    if not lexical:
        for x in parser.decmem:
            parser.decmem.activations[x]=100 #this is to nulify the effect of word retrieval to almost 0

    parser.retrievals = {}
    parser.set_retrieval("retrieval")
    parser.visbuffers = {}
    parser.goals = {}
    parser.set_goal("g")
    parser.set_goal(name="imaginal", delay=0)
    parser.set_goal(name="imaginal_reanalysis", delay=0)
    parser.set_goal("word_info")

    stimuli = [{} for i in range(len(sentence))]
    pos_word = 10
    environment.current_focus = (pos_word + 7+7*visual_effect(sentence[0], visual), 180)
    for x in range(41):
        #this removes any move eyes created previously; we assume that no sentence is longer than 20 words
        parser.productionstring(name="move eyes"+ str(x), string="""
        =g>
        isa         reading
        state       dummy
        ==>
        =g>
        isa         reading
        state       dummy""")
    for i, word in enumerate(sentence):
        pos_word += 7+7*visual_effect(word, visual)
        for j in range(len(stimuli)):
            if j == i:
                stimuli[j].update({i: {'text': word, 'position': (pos_word, 180), 'vis_delay': visual_effect(word, visual)}})
            else:
                stimuli[j].update({i: {'text': "___", 'position': (pos_word, 180), 'vis_delay': 3}})
        
        if i < len(sentence)-3:
            parser.productionstring(name="move eyes" + str(i), string="""
        =g>
        isa             reading
        state            move_eyes
        position        """+str(i)+"""
        ?manual>
        preparation       free
        ==>
        =imaginal>
        isa         action_chunk
        WORD_NEXT0_LEX        """+'"'+str(sentence[i+2])+'"'+"""
        WORD_NEXT0_POS        """+str(pos[i+2])+"""
        WORD_NEXT1_LEX        """+'"'+str(sentence[i+3])+'"'+"""
        WORD_NEXT1_POS        """+str(pos[i+3])+"""
        =g>
        isa             reading
        state   reading_word
        position        """+str(i+1)+"""
        tag             """+str(pos[i+1])+"""
        ?visual_location>
        attended False
        +visual_location>
        isa _visuallocation
        screen_x    """ + str(pos_word+7+7*visual_effect(sentence[i+1], visual)) + """
        screen_y 180
        ~visual>""")
        elif i < len(sentence)-2:
            parser.productionstring(name="move eyes" + str(i), string="""
        =g>
        isa             reading
        state            move_eyes
        position        """+str(i)+"""
        ?manual>
        preparation       free
        ==>
        =imaginal>
        isa         action_chunk
        WORD_NEXT0_LEX        """+'"'+str(sentence[i+2])+'"'+"""
        WORD_NEXT0_POS        """+str(pos[i+2])+"""
        WORD_NEXT1_LEX        None
        =g>
        isa             reading
        state   reading_word
        position        """+str(i+1)+"""
        tag             """+str(pos[i+1])+"""
        ?visual_location>
        attended False
        +visual_location>
        isa _visuallocation
        screen_x    """ + str(pos_word+7+7*visual_effect(sentence[i+1], visual)) + """
        screen_y 180
        ~visual>""")
        elif i < len(sentence)-1:
            parser.productionstring(name="move eyes" + str(i), string="""
        =g>
        isa             reading
        state            move_eyes
        position        """+str(i)+"""
        ?manual>
        preparation       free
        ==>
        =imaginal>
        isa         action_chunk
        WORD_NEXT0_LEX        None
        WORD_NEXT1_LEX        None
        =g>
        isa             reading
        state   reading_word
        position        """+str(i+1)+"""
        tag             """+str(pos[i+1])+"""
        ?visual_location>
        attended False
        +visual_location>
        isa _visuallocation
        screen_x    """ + str(pos_word+7+7*visual_effect(sentence[i+1], visual)) + """
        screen_y 180
        ~visual>""")

    if prints:
        print(sentence)

    parser.goals["g"].add(actr.chunkstring(string="""
    isa             reading
    state           reading_word
    position        0
    tag             """+str(pos[0])))

    parser.goals["imaginal"].add(actr.chunkstring(string="""
    isa             action_chunk
    TREE1_LABEL         NOPOS
    TREE1_HEAD          noword
    TREE2_LABEL         NOPOS
    TREE2_HEAD          noword
    TREE3_LABEL         NOPOS
    TREE3_HEAD          noword
    ANTECEDENT_CARRIED  NO
    WORD_NEXT0_LEX   """+'"'+str(sentence[1])+'"'+"""
    WORD_NEXT0_POS   """+str(pos[1])+"""
    WORD_NEXT1_LEX   """+'"'+str(sentence[2])+'"'+"""
    WORD_NEXT1_POS   """+str(pos[2])))

    # start a dictionary that will collect all created structures, and a list of built constituents
    constituents = {}
    built_constituents = [(Tree("NOPOS", []), (None, "noword")), (Tree("NOPOS", []), (None, "noword")), (Tree("NOPOS", []), (None, "noword"))]
    final_tree = Tree("X", [])

    if prints:
        parser_sim = parser.simulation(realtime=False, gui=False, trace=True, environment_process=environment.environment_process, stimuli=stimuli, triggers='space', times=40)
    else:
        parser_sim = parser.simulation(realtime=False, gui=True, trace=False, environment_process=environment.environment_process, stimuli=stimuli, triggers='space', times=40)

    antecedent_carried = "NO"
    what_antecedent_carried = None

    spr_times = [] #reaction times, recorded and returned

    word_parsed = 0
    last_time = 0

    while True:
        try:
            parser_sim.step()
            #print(parser_sim.current_event)
        except simpy.core.EmptySchedule:
            spr_times = [10 for _ in sentence] #if sth goes wrong, it's probably because it got stuck somewhere; in that case report time-out time per word (40 s) or nan
            break
        if parser_sim.show_time() > 60:
            spr_times = [10 for _ in sentence] #this takes care of looping or excessive time spent - break if you loop (40 s should be definitely enough to move on)
            break
        if parser_sim.current_event.action == "KEY PRESSED: SPACE":
            activation = activations[activations['position'].isin([len(spr_times)+1])]['activation'].to_numpy()[0]
            extra_rule_time = parser.model_parameters["latency_factor"]*np.exp(-parser.model_parameters["latency_exponent"]*(activation*weight))
            # two things play a role - number of matching features; fan of each matching feature; explore these two separately
            #spr_times.append(word_parsed)
            spr_times.append(parser_sim.show_time() + extra_rule_time - last_time)
            last_time = parser_sim.show_time()
            word_parsed += 1

        if word_parsed >= len(sentence):
            break

        #this below - carrying out an action

        if re.search("^RULE FIRED: recall action", parser_sim.current_event.action) or\
                                re.search("^RULE FIRED: move to last action", parser_sim.current_event.action):
            parser_sim.steps(2) #exactly enough steps to make imaginal full
            
            cg = parser.goals["g"].pop()
            wi = parser.goals["word_info"].copy().pop()
            retrieve_wh = activations[activations.position.isin([word_parsed+1])]['retrieve_wh'].to_numpy()[0]
            # make fake wh-antecedent and recall it right away (because the parser always tries to recall WP directly when reading WP (it is guessing we are dealing with subject relative clause)
            if re.search(r"^W", str(wi.cat)):
                parser.decmem.add(actr.chunkstring(string="""
                isa         action_chunk
                TREE0_LABEL     WP
                TREE0_HEAD      wh"""), time=parser_sim.show_time()) # add fake W element
                if str(wi.cat) != "WRB":
                    retrieve_wh = "quick" #we assume (in line with McElree et al. that if you just postulated a wh, you don't need to directly retrieve it, it is still in active memory); all W elements but WRB (where) are retrieved
            if reanalysis:
                reanalysis_value = activations[activations['position'].isin([word_parsed+1])]['reanalysis'].to_numpy()[0]
            else:
                reanalysis_value = "no"
            parser.goals["g"].add(actr.chunkstring(string="""
    isa             reading
    position    """+str(cg.position)+"""
    reanalysis      """+str(reanalysis_value)+"""
    retrieve_wh     """+str(retrieve_wh)+"""
    state           finished_recall"""))

    final_times = spr_times
    if prints:
        print("FINAL TIMES")
        print(final_times)
        # with open('parses/parse_trees.txt', 'a+') as f:
        #     f.write(str(final_tree) + "\n")
        #     f.write("\n")
    return np.array(final_times)
    
if __name__ == "__main__":
    stimuli_csv = ut.load_file(SENTENCES, sep=",") #sentences with frequencies
    words = ut.load_file(WORDS, sep="\t")
    labels = ut.load_file(LABELS, sep="\t")
    activations = ut.load_file(ACTIVATIONS, sep=",")
    DM = parser.decmem.copy()

    #prepare dictionaries to calculate spreading activation
    word_freq = {k: sum(g["FREQ"].tolist()) for k,g in words.groupby("WORD")} #this method sums up frequencies of words across all POS
    label_freq = labels.set_index('LABEL')['FREQ'].to_dict()
        
    parser.model_parameters["latency_factor"] = 0.2
    parser.model_parameters["latency_exponent"] = 0.5
    parser.model_parameters["rule_firing"] = 0.067
    
    for sent_nr in range(1, 2):
        print(sent_nr)
        print(activations)
        # select the right sentence
        used_activations = activations[activations.sentence_no.isin([str(sent_nr)]) & activations.record_RTs.isin(["yes"])]
        #collect stimulus using info from used_activations
        subset_stimuli = stimuli_csv[stimuli_csv.item.isin([str(sent_nr)]) & stimuli_csv.word.isin(used_activations.word.tolist())]
        sentence = subset_stimuli.word.tolist() 
        freqs = subset_stimuli.freq.tolist()
        pos = subset_stimuli.function.tolist()

        final_times = read(parser, sentence=sentence, pos=pos, activations=used_activations, word_freq=word_freq, label_freq=label_freq, weight=1, decmem=DM, lexical=True, syntactic=True, visual=False, reanalysis=True, prints=True)
        print(final_times)
