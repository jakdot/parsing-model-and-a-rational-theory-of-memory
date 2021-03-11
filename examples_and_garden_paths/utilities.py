"""
Utility functions for run_parser.
"""

import math
import pandas as pd
import re
import numpy as np
import pyactr as actr
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tree import Tree

SEC_IN_YEAR = 365*24*3600
SEC_IN_TIME = 15*SEC_IN_YEAR
DFT_STRENGTH_OF_ASSOCIATION = 20
EMPTY = "   "

PHRASES = {"NP": {"N", "-", "P"}, "WHNP": {"N", "W"}, "VP": {"V"}, "ADJP": {"J"}, "WHADJP":
        {"J", "W"}, "ADVP": {"R"}, "WHADVP": {"R", "W"}, "SBAR": {"W", "-", "I"}, "PP": {"I", "T"}, "WHPP":
        {"I", "T", "W"}, "S": {"V"}, "QP": {"C"}, "RRC": {"V"}, "QP": {"C", "N"}, "PRT": {"J", "R"}, "CONJP": {"R", "I"}, "NAC": {"N"}, "NX": {"N", "-"}, "UCP": {"N", "J"}}

lemmatizer = WordNetLemmatizer() 

#following three functions for lemmatization
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def penn_to_wn(tag):
    return get_wordnet_pos(tag)

def lemmatize(word, tag):
    tag = penn_to_wn(tag)
    if tag:
        lemma = lemmatizer.lemmatize(word, tag)
        return lemma
    else:
        return word

def get_pos_info(elem, pos_dict):
    """
    Get pos info for the purposes of lemmatization and word2vec.

    elem: tuple
    
    We check whether elem[0] stores POS and if so, we store that info in small dict pos_dict.
    """
    if elem[0] == "TREE0_LABEL" and "TREE0_HEAD" not in pos_dict:
        pos_dict["TREE0_HEAD"] = elem[1][0]
    elif elem[0] == "TREE0_HEADPOS":
        pos_dict["TREE0_HEAD"] = elem[1][0]
    elif elem[0] == "TREE1_LABEL" and "TREE1_HEAD" not in pos_dict:
        pos_dict["TREE1_HEAD"] = elem[1][0]
    elif elem[0] == "TREE1_HEADPOS":
        pos_dict["TREE1_HEAD"] = elem[1][0]
    elif elem[0] == "TREE2_LABEL" and "TREE2_HEAD" not in pos_dict:
        pos_dict["TREE2_HEAD"] = elem[1][0]
    elif elem[0] == "TREE2_HEADPOS":
        pos_dict["TREE2_HEAD"] = elem[1][0]
    #elif elem[0] == "WORD_NEXT0_POS":
    #    pos_dict["WORD_NEXT0_LEX"] = elem[1][0]
    #elif elem[0] == "WORD_NEXT1_POS":
    #    pos_dict["WORD_NEXT1_LEX"] = elem[1][0]

def load_file(lfile, index_col=None, sep=","):
    """
    Loads file as a list
    """
    csvfile = pd.read_csv(lfile, index_col=index_col, header=0, sep=sep)
    return csvfile

def calculate_S(slot, value, word_freq, label_freq, strength_of_association):
    fan = max(1, word_freq.get(value, 0), label_freq.get(value, 0))
    return max(strength_of_association.get(slot, DFT_STRENGTH_OF_ASSOCIATION) - math.log(fan), 0)

def calculate_spreading(spreading, used_db, word_freq, label_freq, strength_of_association, blind, pos_dict):
    for each in spreading:
        if str(each[1]) != "None" and str(each[0]) not in blind:
            get_pos_info(each, pos_dict)
    for each in spreading:
        if str(each[1]) != "None" and str(each[0]) not in blind:
            string_each1 = str(each[1])
            if each[0] in pos_dict:
                string_each1 = lemmatize(string_each1, pos_dict[each[0]])
            yield np.where(used_db[each[0]].values == string_each1, calculate_S(str(each[0]), string_each1, word_freq, label_freq, strength_of_association), 0)

def recall_action(actions, used_spreading, used_retrieval, critical_rules, recently_retrieved, built_constituents, word_freq, label_freq, strength_of_association={}, number_retrieved=3, postulate_gaps=True, reduce_unary=True, decay=0.5, blind=None, prints=True):
    """
    Recall action from dm using spreading activation and base-level learning.

    :param actions: dataframe of actions
    :param used_spreading: buffer with a chunk used for spreading activation
    :param used_retrieval: what retrieval should be used to guide recall (by checking what word was found
    :param postulate_gaps: should new postulate_gaps be allowed to be found?
    : param number_retrieved: how many actions should be retrieved

    :return dict of collected rules, with their activation and the number (from which parallel process they are coming)
    """
    #print("CRIT RULES", critical_rules)
    if not blind:
        blind = {}
    found = used_retrieval.copy().pop()
    spreading = used_spreading.copy().pop() #chunk ensuring spreading activation

    #use this if you want to restrict search to actions matching in WORD0_POS
    #used_db = actions[actions['TREE0_LABEL'].isin([str(built_constituents[-1][0].label())])]
    #used_db = actions[actions['WORD_NEXT0_POS'].isin([str(spreading.WORD_NEXT0_POS)])]
    try:
        used_db = actions[(actions['WORD_NEXT0_POS'].isin([str(spreading.WORD_NEXT0_POS)])) |  (actions['TREE0_LABEL'].isin([str(built_constituents[-1][0].label())])) | (actions['TREE1_LABEL'].isin([str(built_constituents[-2][0].label())])) | (actions['TREE0_HEADPOS'].isin([str(built_constituents[-1][1][0])]))] #this is to speed up search (we only consider some actions that have match on words or on labels, if the whole is considered it is quite slow)
    except KeyError:
        used_db = actions[(actions['TREE0_LABEL'].isin([str(built_constituents[-1][0].label())])) | (actions['TREE1_LABEL'].isin([str(built_constituents[-2][0].label())])) | (actions['TREE0_HEADPOS'].isin([str(built_constituents[-1][1][0])]))] #this is to speed up search (we only consider some actions that have match on words or on labels, if the whole is considered it is quite slow)

    if critical_rules:
        used_db = used_db[(used_db['ACTION'].isin([critical_rules[0]])) & (used_db['ACTION_RESULT_LABEL'].isin([critical_rules[1]]))]
    
    #use this if you do not want to assume any restrictions (somewhat slower)
    #used_db = actions

    #do not recall reduced_binary, if you have only one tree (otherwise, parsing would crash)
    if built_constituents[-2][0].label() == "None" or built_constituents[-2][0].label() == "NOPOS":
        used_db = used_db[~used_db['ACTION'].isin(["reduce_binary"])]
    #else:
    #    #do not recall reduce_binary with mismatching label on the second tree
    #    used_db2 = used_db[(used_db['ACTION'].isin(["reduce_binary"])) &  (used_db['TREE1_LABEL'].isin([str(built_constituents[-2][0].label())]))]
    #    used_db = used_db[~used_db['ACTION'].isin(["reduce_binary"])]
    #    used_db = used_db.append(used_db2)

    #do not repeatedly recall unary reduction (avoid infinite loops)
    if not reduce_unary:
        used_db = used_db[~used_db['ACTION'].isin(["reduce_unary"])]
    
    #do not repeatedly recall postulate gap (avoid infinite loops)
    if str(spreading.ACTION_PREV) == "postulate_gap":
        used_db = used_db[~used_db['ACTION'].isin(["postulate_gap"])]
    
    #do not recall postulate gap if some restrictions apply (currently at most 3 gaps per word allowed) (avoid infinite loops)
    if not postulate_gaps:
        used_db = used_db[~used_db['ACTION'].isin(["postulate_gap"])]

    
    temp_parser_retrievals = {}

    counter_actions = Counter()
    
    pos_dict = {} #dict for POS

    number_matching_fs = 0
    
    fan_size = 0

    parser_retrievals = [0, {}]

    #start recalling based on activation
    final_array = np.array(used_db['ACTIVATION'])

    for col in calculate_spreading(spreading=spreading, used_db=used_db, word_freq=word_freq, label_freq=label_freq, strength_of_association=strength_of_association, blind=blind, pos_dict=pos_dict):
        final_array += col

    temp_number_retrieved = np.count_nonzero(final_array==np.max(final_array)) #count the number of elements with max activation
    if temp_number_retrieved > number_retrieved:
        number_retrieved = temp_number_retrieved
        #use all elements with max activation

    max_activations_ind = np.argpartition(final_array, range(-number_retrieved, -1))[-number_retrieved:]
    max_activations = np.sort(final_array[max_activations_ind])
    #max_activations = [ np.max(final_array) ] #if only one element is needed, this is easier

    action_freq = 0

    if prints:
        print(reduce_unary)
        print("Imaginal buffer:")
        print(spreading)
    for i in reversed(range(len(max_activations_ind))):
        max_activation = max_activations[i]
        ind = max_activations_ind[i]

        #if activation is above some threshold keep the element
        if max_activation > -50:
            elem = used_db.iloc[ind]
            #elem[elem=="''"] = "None"
            #elem[elem==''] = "None"

            #print what was found
            if prints:
                print(elem.to_dict(), max_activation)
                print("SPREADING FROM:")
                for each in spreading:
                    try:
                        string_each1 = str(each[1].values)
                    except AttributeError:
                        continue
                    if each[0] in pos_dict:
                        string_each1 = lemmatize(string_each1, pos_dict[each[0]])
                    if str(each[0]) not in blind and str(elem[each[0]]) == string_each1:
                        fan = max(1, word_freq.get(string_each1, 0), label_freq.get(string_each1, 0))
                        print(each, strength_of_association.get(str(each[0]), DFT_STRENGTH_OF_ASSOCIATION) - math.log(fan))
            
            action_label_activation = temp_parser_retrievals.get((str(elem.ACTION), str(elem.ACTION_RESULT_LABEL)), 0) + max_activation
            counter_actions.update(((str(elem.ACTION), str(elem.ACTION_RESULT_LABEL)),))
            for each in spreading:
                if str(each[0]) not in blind and str(elem[each[0]]) == str(each[1]):
                    number_matching_fs += 1
                    fan_size += max(1, word_freq.get(each[1].values, 0), label_freq.get(each[1].values, 0))
            temp_parser_retrievals[(str(elem.ACTION), str(elem.ACTION_RESULT_LABEL))] = action_label_activation
            if action_label_activation > parser_retrievals[0]:
                parser_retrievals = [action_label_activation, {'action': str(elem.ACTION), 'action_result_label': (str(elem.ACTION_RESULT_LABEL), action_label_activation)}]

            #if sum(temp_parser_retrievals[str(elem.ACTION)].values()) > parser_retrievals[0]:
                #parser_retrievals[0] = sum(temp_parser_retrievals[str(elem.ACTION)].values())
                #parser_retrievals[1] = {'action': str(elem.ACTION), 'action_result_label': max(temp_parser_retrievals[str(elem.ACTION)].items(), key=lambda x:x[1])}

    parser_retrievals[0] = parser_retrievals[0]/number_retrieved
    
    if prints:
        print("==========COLLECTED ACTIONS===========")
        print(temp_parser_retrievals)
        print(parser_retrievals)

    number_of_matching_actions = counter_actions.most_common()[0][1]
    number_matching_fs = number_matching_fs / number_of_matching_actions
    fan_size = fan_size/number_matching_fs
    
    return parser_retrievals, number_of_matching_actions, number_matching_fs, fan_size

def project_head(projected_label, left_element, *headinfo):
    phrase = re.split("_BAR$", projected_label)[0]
    phrase = re.split("-", phrase)[0]
    possible_heads = PHRASES.get(phrase, set())
    current_head = (None, None)
    if left_element and len(left_element.leaves()) == 1:
        current_head = (left_element.label(), left_element.leaves()[0])
    for head in headinfo:
        if head[0] and head[0][0] in possible_heads:
            current_head = head
    return current_head

def collect_parse(action_dict, built_constituents):
    """
    Collect created parse into a tree.

    :param action: dict describing the current action
    :param built_constituents: list of tuples (tree, (headpos, headlex)) that were parsed or are currently parsed into trees
    """
    # if no action, mark that we are finished building constituents
    if (not action_dict) or action_dict["action"] == 'shift':
        return None

    # binary reduction simplifies built_constituents and creates a new constituent
    if action_dict["action"] == "reduce_binary":
        old_constituent = built_constituents.pop(-1)
        very_old_constituent = built_constituents.pop(-1)
        built_constituents.append((Tree(action_dict['action_result_label'][0], [very_old_constituent[0], old_constituent[0]]), project_head(action_dict['action_result_label'][0], very_old_constituent[0], very_old_constituent[1], old_constituent[1])))
    elif action_dict["action"] == "reduce_unary":
        # otherwise, we can use the last word
        #ignore in-between categories for unary reduction
        #* is used because Tree does not allow stacking of unary trees on top of each other; without * two unary trees would be collapsed; * is ignored in the computation of accuracy
        old_constituent = built_constituents.pop(-1)
        if len(old_constituent[0].leaves()) == 1:
            built_constituents.append((Tree(action_dict["action_result_label"][0], [old_constituent[0], Tree(EMPTY, [EMPTY])]), project_head(action_dict['action_result_label'][0], None, (old_constituent[0].label(), old_constituent[0].leaves()[0]))))
        else:
            built_constituents.append((Tree(action_dict["action_result_label"][0], [old_constituent[0], Tree(EMPTY, [EMPTY])]), project_head(action_dict["action_result_label"][0], None, old_constituent[1])))
    elif action_dict["action"] == "postulate_gap":
        built_constituents.append((Tree("-NONE-", [action_dict["action_result_label"][0]]), (None, action_dict["action_result_label"][0])))
    else:
        raise ValueError("Unknown action")


