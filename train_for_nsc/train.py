"""
Classifier for classifier-based parsing.
"""

from collections import Counter, namedtuple, deque
import math
from nltk import Tree
import io
import csv
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import BracketParseCorpusReader
from nltk.corpus import treebank
from nltk.stem import WordNetLemmatizer
import re
import sys
import pandas as pd
from copy import deepcopy
from nltk.corpus import wordnet

# treebank below can be used to test on selected files
# you have to specify your own folder in which natural_stories file would be located
# the folder has to be found by the nltk utility LazyCorpusLoader
# the folder should include the file all-parses.txt.penn from the Natural Stories Corpus
treebank = LazyCorpusLoader(
    'treebank/natural_stories', BracketParseCorpusReader, r'.*\.penn',
    tagset='wsj', encoding='ascii') #to test on files
    
lemmatizer = WordNetLemmatizer() 

replacing = {'$': 'translated_dollar', "'": "translated_apos"}
replacing = dict((re.escape(k), v) for k, v in replacing.items())
pattern = re.compile("|".join(replacing.keys()))

SEC_IN_YEAR = 365*24*3600
SEC_IN_TIME = 15*SEC_IN_YEAR
DECAY = 0.5

PHRASES = {"NP": {"N", "-", "P"}, "WHNP": {"N", "W"}, "VP": {"V"}, "ADJP": {"J"}, "WHADJP":
        {"J", "W"}, "ADVP": {"R"}, "WHADVP": {"R", "W"}, "SBAR": {"W", "-", "I"}, "PP": {"I", "T"}, "WHPP":
        {"I", "T", "W"}, "S": {"V"}, "QP": {"C"}, "RRC": {"V"}, "QP": {"C", "N"}, "PRT": {"J", "R"}, "CONJP": {"R", "I"}, "NAC": {"N"}, "NX": {"N", "-"}, "UCP": {"N", "J"}}

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

def lemmatize(word, tag, alternative_tag=None):
    if tag and tag != " ":
        tag = penn_to_wn(tag)
    elif alternative_tag and alternative_tag != " ":
        tag = penn_to_wn(alternative_tag)
    if word != " " and tag:
        return lemmatizer.lemmatize(word, tag)
    else:
        return word

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def save_actions(actions, filename="actions.csv"):
    """
    Save rules as a csv.
    """
    # create a dataframe, and convert it to csv
    df = pd.DataFrame(actions)
    #print(df)
    df.to_csv(filename, encoding='utf-8', sep='\t', index=False)

def save_words(words, filename="words.csv"):
    """
    Save words, which is a counter, as a csv.
    """
    # create a dataframe, and convert it to csv
    zips_for_df = zip(*words.items())

    words = zip(*next(zips_for_df))
    freq = next(zips_for_df)

    df = pd.DataFrame({"WORD": next(words), "POS": next(words), "FREQ": freq})
    #print(df)
    df.to_csv(filename, encoding='utf-8', sep='\t', index=False)

def save_labels(labels, words, filename="labels.csv"):
    """
    Save labels, which is a counter, as a csv. Add POS from words to the count.
    """
    # create a dataframe, and convert it to csv
    zips_for_df = zip(*labels.items())

    labels = next(zips_for_df)
    freq = next(zips_for_df)

    pos = Counter()

    for key, val in words.items():
        pos.update({key[1]: val})

    df = pd.DataFrame({"LABEL": labels + tuple(pos.keys()), "FREQ": freq + tuple(pos.values())})
    #print(df)
    df.to_csv(filename, encoding='utf-8', sep='\t', index=False)

class Trainer:
    """
    Create a classifier based on CNF grammar given a corpus name.

    Description:
    The grammar is converted to Chomsky Normal Form (CNF). The tags in this
    grammar are factorized (Roark, 2001).


    Attributes:

    _corpus_name:       The name of the corpus used to obtain the
                        grammar rules.

    _lexicalized:       Can be set to True by setting the lexicalized
                        imput parameter to True. When set to true, the
                        head information will be added to the CNF tags.

    _sentences:         The parsed sentences in their original Form

    _tagged_words:      A list of all words with their corresponding tags

    _cnf_sentences:     The parsed sentences in CNF. factorization
                        for the tags is used according to Roark, 2001.


    Functions:

    def pretty_print(key='showall', prob=False, freq=False)
        Print probability and frequency values in a neat list.

    def words_to_csv(self, filename="words.csv"):
        Store the words with tag in the csv.

    """

    def __init__(self, corpus_name, lexicalized=False, tagset='default'):
        self._corpus_name       = corpus_name
        self._lexicalized       = lexicalized
        self._tagset            = tagset
        self._tagged_words       = Counter()
        self._sentences         = self._get_sentences()
        self.label_counter = Counter()
        self._cnf_sentences     = self._get_cnf_sentences()
        self.action_counter = Counter() #counter for actions
        self.future_included = True #should Trainer use future information (next word?)
        self.current_action_state = namedtuple("current_action_state", "tree0_label, tree0_head, tree0_headpos, tree1_label, tree1_head, tree1_headpos, tree2_label, tree2_head, tree2_headpos, tree3_label, tree3_head, antecedent_carried, next_word0_lex, next_word0_pos, next_word1_lex, next_word1_pos, action, action_result_label, last_action")

    # Class instance representation
    def __repr__(self):
        return 'Grammar("{}")'.format(self._corpus_name)

    def _get_sentences(self):
        """
        This function loads in the parsed sentences from the
        given corpus.
        """
        if self._corpus_name == 'Penn Treebank':

            return treebank.parsed_sents()
        else:
            raise ValueError('No corpus found with'
                             'name {}'.format(self._corpus_name))

    def _get_cnf_sentences(self):
        """
        This function takes as input a list of Tree objects
        and will return a list of Tree objects in CNF
        """
        cnf_sentences = []

        for sentence in self._sentences:
            cnf_sentence = self._rename_tags(sentence) # factorization and cleaning
            cnf_sentences.append(cnf_sentence)

        return cnf_sentences

    def _rename_tags(self, tree):
        """
        Rename the tags and restructure the tree to cnf with
        Roark factorization.
        Find duplicates in leaves and make them unique.
        Add information about heads.
        """

        def _put_head_on_subtree(tree, leaf_nodes):
            """
            Put the head information on the correct subtree labels
            :count int: assign unique ident to duplicated leaves (this is needed later for correct parsing)
            """
            oldlabel = tree.label()
            simplified_label = re.split("-", oldlabel)[0]
            head, head_pos = None, None
            for subtree in tree:
                #subtree is a terminal
                if subtree[0] in leaf_nodes:
                    #create head and make head lowercase
                    temp_head, temp_head_pos = subtree[0].lower(), subtree.label()

                    #check whether the head should project higher up
                    if PHRASES.get(simplified_label) and temp_head_pos and temp_head_pos[0] in PHRASES[simplified_label]:
                        head, head_pos = temp_head, temp_head_pos
                    temp_newlabel = "{}|SPL{}#MID{}|SPL".format(temp_head_pos, temp_head, temp_head_pos)
                    subtree.set_label(temp_newlabel)

                #subtree is not a terminal
                else:
                    temp_head, temp_head_pos = _put_head_on_subtree(subtree, leaf_nodes)

                    if PHRASES.get(simplified_label) and temp_head_pos and temp_head_pos[0] in PHRASES[simplified_label]:
                        head, head_pos = temp_head, temp_head_pos

            #if we found a head in one of the subtrees
            if head:

                #create new label with head info

                newlabel = "{}|SPL{}#MID{}|SPL".format(oldlabel, head, head_pos)
                tree.set_label(newlabel)

            return head, head_pos

        def _recurse_tags(tree, parent, sibling, branches):

            #Obtain the terminal nodes of this tree
            leaves = tree.leaves()

            # Check if the queue of branches that need to be processed
            # is filled. If this is the case, processing the branches
            # takes top priority.

            current_label = tree.label()
            tree.set_label(current_label)

            for subtree in tree:
                try:
                    current_label = subtree.label()
                except AttributeError:
                    pass
                else:
                    subtree.set_label(current_label)


            if branches != []:
                newparent   = str(tree.label())
                newlabel    = "{}^{}".format(parent, sibling)
                lefttree    = branches.pop(0)

                # Construct both branch sides of the tree
                leftside  = _recurse_tags(tree, newparent, None, [])
                rightside = _recurse_tags(lefttree, newlabel,
                                          newparent, branches)

                return "({} {} {})".format(newlabel, leftside, rightside)
            # Else if the current rule in the tree maps to more than
            # 2 children, put the branches in a queue.
            elif len(tree) > 2:
                # branches on the queue are all branches that will be nested:
                # These are the 3rd+ branch in a tree. The new parent is the
                # current tree label.
                branches    = [tree[i] for i in range(2, len(tree))]
                newparent   = str(tree.label())
                newsibling  = str(tree[0].label())

                # The rightmost branch will be written like a normal tree.
                # The left side will get nested and labels will be rewritten.
                leftside  = _recurse_tags(tree[0], newparent, None, [])
                rightside = _recurse_tags(tree[1], newparent,
                                          newsibling, branches)

                # If the current node is on the left hand side or does
                # not have a parent, do not change the label. Else,
                # reformat the label to <parent>^<sibling>
                if parent is None or sibling is None:
                    return "({} {} {})".format(
                                str(tree.label()), leftside, rightside)
                else:
                    newlabel = "{}^{}".format(parent, sibling)
                    return "({}({} {} {}))".format(
                                newlabel, str(tree.label()),
                                leftside, rightside)
            # If a rule is binary, check for whether we are on
            # the right or left branch of the tree.
            elif len(tree) == 2:
                # If on the left branch, the new label will be the
                # current label of the tree.
                if sibling is None:
                    newlabel = str(tree.label())
                    newsibling = str(tree[0].label())

                    # Recursion, change tags of the subtrees
                    leftside  = _recurse_tags(tree[0], newlabel, None, [])
                    rightside = _recurse_tags(tree[1], newlabel, newsibling, [])

                    return "({} {} {})".format(newlabel, leftside, rightside)
                # If on the right branch, the new label will be
                # reformatted to <parent>^<sibling>, which will
                # then be rewritten as the current label.
                else:
                    newlabel = "{}^{}".format(parent, sibling)
                    newparent = str(tree.label())
                    newsibling = str(tree[0].label())

                    # Recursion, change tags of the subtrees
                    leftside  = _recurse_tags(tree[0], newparent, None, [])
                    rightside = _recurse_tags(tree[1], newparent, newsibling, [])
                    return "({} ({} {} {}))".format(
                                newlabel, newparent, leftside, rightside)
            # If a rule is unary, first check whether the rule leads
            # to a nonterminal symbol.
            else:
                # If the unary rule leads to a nonterminal symbol
                if tree[0] not in leaves:
                    subtree = _recurse_tags(tree[0], tree.label(), None, [])
                    if sibling is None:
                        return "({} {})".format(tree.label(), subtree)
                    else:
                        newlabel = "{}^{}".format(parent, sibling)
                        return "({} ({} {}))".format(
                                    newlabel, tree.label(), subtree)
                # Else if the unary rule leads to a terminal symbol
                elif sibling is None:
                    newlabel = tree.label()
                    new_word = tree[0].lower()
                    word_label = re.split("\|SPL", newlabel)[0]
                    lemma = lemmatize(pattern.sub(lambda m: replacing[re.escape(m.group(0))], new_word), word_label)
                    if lemma != pattern.sub(lambda m: replacing[re.escape(m.group(0))], new_word):
                        self._tagged_words.update([(lemma, word_label)]) 
                        self._tagged_words.update([(pattern.sub(lambda m: replacing[re.escape(m.group(0))], new_word), word_label)])
                    else:
                        self._tagged_words.update([(lemma, word_label)]) 
                    if new_word in set_of_dupls:
                        new_word = "".join([new_word, "#", str(unique_counts[0])])
                        unique_counts[0] += 1
                    
                    return "({} {})".format(pattern.sub(lambda m: replacing[re.escape(m.group(0))], newlabel), pattern.sub(lambda m: replacing[re.escape(m.group(0))], new_word))
                else:
                    newlabel = "{}^{}".format(parent, sibling)
                    new_word = tree[0].lower()
                    word_label = re.split("\|SPL", tree.label())[0]
                    lemma = lemmatize(pattern.sub(lambda m: replacing[re.escape(m.group(0))], new_word), word_label)
                    if lemma != pattern.sub(lambda m: replacing[re.escape(m.group(0))], new_word):

                        self._tagged_words.update([(lemma, word_label)]) #split because terminal nodes should not store infor about lex heads
                        self._tagged_words.update([(pattern.sub(lambda m: replacing[re.escape(m.group(0))], new_word), word_label)])
                    else:
                        self._tagged_words.update([(lemma, word_label)]) 
                    if new_word in set_of_dupls:
                        new_word = "".join([new_word, "#", str(unique_counts[0])])
                        unique_counts[0] += 1
                    pattern.sub(lambda m: replacing[re.escape(m.group(0))], new_word)
                    return "({} ({} {}))".format(
                                newlabel, tree.label(), pattern.sub(lambda m: replacing[re.escape(m.group(0))], new_word))

        # Start recursion
        if self._lexicalized == True:
            leaf_nodes = tree.leaves()
            _put_head_on_subtree(tree, leaf_nodes)

        temp_counter =  Counter([x.lower() for x in leaf_nodes])
        set_of_dupls = {x for x in temp_counter if temp_counter[x] > 1}

        unique_counts = [0]

        string = _recurse_tags(tree, None, None, [])

        return Tree.fromstring(string)

            #############################################
            #                                           #
            #   Public functions of the Grammar class   #
            #                                           #
            #############################################

    def parse_sent(self, sent):
        """
        Parse sentence sent.

        :param sent: sentence is nltk.tree as given in _cnf_sentences or _sentences
        """
        def _update_action_counter(action, last_action):
            action, result = re.split("->", str(action))
            last_action, last_result = re.split("->", str(last_action))

            result = re.split("\^", str(result))
            if len(result) > 1:
                result = re.split("\|SPL", result[0])[0] + "_BAR"
            else:
                result = re.split("\|SPL", result[0])[0]

            if result != "''":
                self.label_counter.update([result])

            self.label_counter.update([action])

            last_result = re.split("\^", str(last_result))
            if len(last_result) > 1:
                last_result = re.split("\|SPL", last_result[0])[0] + "_BAR"
            else:
                last_result = re.split("\|SPL", last_result[0])[0]

            temp_tree_stack = [('', '', ''), ('', '', ''), ('', '', ''), ('', '', '')]
            for pos, x in enumerate(tree_stack[-4:]):
                for elem in x[1]:
                    #remove info for coreference
                    elem = "".join(re.split("-[0-9][0-9]*", str(elem)))
                    elem = re.split("\^", str(elem))
                    #check whether this is a BAR (incomplete) category (len(elem)>1); such categories can only be at the currently built (rightmost) tree (pos == 3)
                    if len(elem) > 1 and pos == 3:
                        temp_list = re.split("\|SPL", elem[0])
                        temp_list[0] = temp_list[0] + "_BAR"
                    else:
                        temp_list = re.split("\|SPL", elem[0])
                    leftchild = "".join(re.split("-[0-9][0-9]*", str(x[2][0])))
                    leftchild = re.split("\^", str(leftchild))
                    if len(leftchild) > 1:
                        leftchild = re.split("\|SPL", leftchild[0])[0]
                        leftchild += "_BAR"
                    else:
                        leftchild = re.split("\|SPL", leftchild[0])[0]
                    rightchild = "".join(re.split("-[0-9][0-9]*", str(x[3][0])))
                    rightchild = re.split("\^", str(rightchild))
                    if len(rightchild) > 1:
                        rightchild = re.split("\|SPL", rightchild[0])[0]
                        rightchild += "_BAR"
                    else:
                        rightchild = re.split("\|SPL", rightchild[0])[0]
                    try:
                        tobeappended = (temp_list[0], re.split("\#MID",  temp_list[1])[0], re.split("\#MID",  temp_list[1])[1], leftchild, rightchild)
                        # if treelabel != headpos, append everything, otherwise ignore headpos (we deal with terminal)
                        if tobeappended[0] != tobeappended[2]:
                            temp_tree_stack.append(tobeappended)
                        else:
                            temp_tree_stack.append((tobeappended[0], tobeappended[1], " ", tobeappended[3], tobeappended[4]))
                        #temp_tree_stack.append(("".join([temp_list[0], "".join(temp_list[2:])]), re.split("\#MID",  temp_list[1])[0], re.split("\#MID",  temp_list[1])[1]))
                    except IndexError:
                        temp_tree_stack.append((temp_list[0], " ", " ", leftchild, rightchild))

            if antecedents:
                antecedent_carried = "YES"
            else:
                antecedent_carried = "NO"
            self.label_counter.update([antecedent_carried])

            if self.future_included == True:
                self.action_counter.update([self.current_action_state(tree0_label=temp_tree_stack[-1][0], tree0_head=lemmatize(temp_tree_stack[-1][1], temp_tree_stack[-1][2], temp_tree_stack[-1][0]), tree0_headpos=temp_tree_stack[-1][2],\
                    tree1_label=temp_tree_stack[-2][0], tree1_head=lemmatize(temp_tree_stack[-2][1], temp_tree_stack[-2][2], temp_tree_stack[-2][0]), tree1_headpos=temp_tree_stack[-2][2], \
                    tree2_label=temp_tree_stack[-3][0], tree2_head=lemmatize(temp_tree_stack[-3][1], temp_tree_stack[-3][2], temp_tree_stack[-3][0]), tree2_headpos=temp_tree_stack[-3][2],\
                    tree3_label=temp_tree_stack[-4][0], tree3_head=temp_tree_stack[-4][1],\
                    antecedent_carried=antecedent_carried,\
                    next_word0_lex="".join(re.split("-[0-9][0-9]*", next_word0[0])), next_word0_pos=re.split("\|SPL", next_word0[1])[0],\
                    next_word1_lex="".join(re.split("-[0-9][0-9]*", next_word1[0])), next_word1_pos=re.split("\|SPL", next_word1[1])[0],\
                    action=str(action), action_result_label=str(result), last_action=str(last_action))])
            else:
                self.action_counter.update([self.current_action_state(tree0_label=temp_tree_stack[-1][0], tree0_head=temp_tree_stack[-1][1], tree0_headpos=temp_tree_stack[-1][2], tree0_leftchild=temp_tree_stack[-1][3], tree0_rightchild=temp_tree_stack[-1][4],\
                    tree1_label=temp_tree_stack[-2][0], tree1_head=temp_tree_stack[-2][1], tree1_headpos=temp_tree_stack[-2][2], tree1_leftchild=temp_tree_stack[-2][3], tree1_rightchild=temp_tree_stack[-2][4],\
                    tree2_label=temp_tree_stack[-3][0], tree2_head=temp_tree_stack[-3][1], tree2_headpos=temp_tree_stack[-3][2],\
                    tree3_label=temp_tree_stack[-4][0], tree3_head=temp_tree_stack[-4][1],\
                    antecedent_carried=antecedent_carried,\
                    action=str(action), action_result_label=str(result), last_action=str(last_action))])
                pass

        def _get_antecedent(phrase):
            """
            Remove information about coreference. Store that information separately in a set of antecedents (set antecedents).
            """
            if (re.search("^W", str(phrase)) or re.search("-TPC", str(phrase))) and re.search("-[0-9][0-9]*", str(phrase)):
                antecedent = None
                pot_ant = re.split("\^", str(phrase))[0]
                if re.search("(=|-)[0-9][0-9]*\|SPL", str(pot_ant)):
                    antecedent = re.split("\|SPL", re.search("(=|-)[0-9][0-9]*\|SPL", str(pot_ant)).group())[0]
                elif re.search("(=|-)[0-9][0-9]*$", str(pot_ant)):
                    antecedent = re.search("(=|-)[0-9][0-9]*$", str(pot_ant)).group()
                if antecedent and antecedent not in traces:
                    antecedents.add(antecedent)
                return "".join(re.split("-[0-9][0-9]*", "".join(re.split("-[0-9][0-9]*", str(phrase)))))
            return phrase

        def _reduce(last_action):
            #reduce and record the action(s)
            while True:
                #unary action
                if tree_stack and (tree_stack[i][0], tree_stack[i][1]) in phrases_in_sent:
                    #adjunction is preferred
                    try:
                        new_phrase = phrases_in_sent[(tree_stack[i][0], tree_stack[i][1])].pop(phrases_in_sent[(tree_stack[i][0], tree_stack[i][1])].index(tree_stack[i][1][0]))
                    except ValueError:
                        new_phrase = phrases_in_sent[(tree_stack[i][0], tree_stack[i][1])].pop()
                    if not phrases_in_sent[(tree_stack[i][0], tree_stack[i][1])]:
                        phrases_in_sent.pop((tree_stack[i][0], tree_stack[i][1]))
                    cleaned_phrase = "".join(re.split("=[0-9][0-9]*", "".join(re.split("-[0-9][0-9]*", str(new_phrase)))))
                    _update_action_counter("reduce_unary->{}".format(cleaned_phrase), last_action)
                    _get_antecedent(new_phrase)
                    last_action = "reduce_unary->{}".format(cleaned_phrase)
                    self.label_counter.update(["NOPOS"])
                    tree_stack[i] = (tree_stack[i][0], (new_phrase,), (tree_stack[i][1][0],), ("NOPOS",))
                #binary action
                elif len(tree_stack) > 1 and (tuple(x for tupl in (tree_stack[i-1][0], tree_stack[i][0]) for x in tupl), tuple(x for tupl in (tree_stack[i-1][1], tree_stack[i][1]) for x in tupl)) in phrases_in_sent:
                    new_key =  (tuple(x for tupl in (tree_stack[i-1][0], tree_stack[i][0]) for x in tupl), tuple(x for tupl in (tree_stack[i-1][1], tree_stack[i][1]) for x in tupl))
                    new_phrase = phrases_in_sent[new_key].pop()
                    cleaned_phrase = "".join(re.split("=[0-9][0-9]*", "".join(re.split("-[0-9][0-9]*", str(new_phrase)))))
                    if not phrases_in_sent[new_key]:
                        phrases_in_sent.pop(new_key)
                    _update_action_counter("reduce_binary->{}".format(cleaned_phrase), last_action)
                    _get_antecedent(new_phrase)
                    last_action = "reduce_binary->{}".format(cleaned_phrase)
                    old_phrase = tree_stack.pop()
                    if not re.search("\^", str(old_phrase[1][0])):
                        tree_stack[i] = (new_key[0], (new_phrase,), (tree_stack[i][1][0],), (old_phrase[1][0],))
                    else:
                        tree_stack[i] = (new_key[0], (new_phrase,), (tree_stack[i][1][0],), (old_phrase[2][0],))
                else:
                    break

            return last_action
        
        leaf_nodes = sent.leaves()
        #print(leaf_nodes)
        words = sent.pos()
        sentence_stack = []
        tree_stack = [(('',), ('',), ('',), ('',)), (('',), ('',), ('',), ('',)),(('',), ('',), ('',), ('',)), (('noword',), ('NOPOS|SPLnoword#MIDNOPOS|SPL',), ('',), ('',)) ]
        #tree_stack = [(('noword',), ('NOPOS|SPLnoword#MIDNOPOS|SPL',), ('',), ('',)) for _ in range(4)]
        self.label_counter["NOPOS"] = self.label_counter.get("NOPOS", 0) + 1
        self._tagged_words[("noword", "NOPOS")] = self._tagged_words.get(("noword", "NOPOS"), 0) + 1
        phrases_in_sent = {}
        antecedents = set()
        traces = set() #set of traces to be matched to antecedents
        last_action = "->"
        gaps = {} #gaps to be postulated

        #collect all non-terminal trees (ignore trees of height 2, which are just pos)
        for s in sent.subtrees(lambda t: t.height() != 2):
            key = (tuple(s.leaves()), tuple(t.label() for t in s))
            phrases_in_sent.setdefault(key , []).append(s.label())
            
        i = -1 #to be used for stacks of trees

        #pre-clean the sentence (remove gaps)
        nogap_j = 0
        for j, (word, pos) in enumerate(words):
            next_word = (word, pos)
            if re.split("\|SPL", pos)[0] == "-NONE-":
                gaps.setdefault(nogap_j, []).append([next_word, [word, pos]])
            else:
                sentence_stack.append(next_word)
                nogap_j += 1
        for j, (word, pos) in enumerate(sentence_stack):

            shifted_word0 = re.split("\#.[0-9]*", word)[0] # remove word-by-word numbering
            
            next_word0 = (shifted_word0, pos)

            try:
                shifted_word1 = re.split("\#.[0-9]*", sentence_stack[j+1][0])[0] # remove word-by-word numbering
            except IndexError:
                next_word1 = ("noword", "NOPOS")
            else:
                next_word1 = (shifted_word1, sentence_stack[j+1][1])
            
            try:
                shifted_word2 = re.split("\#.[0-9]*", sentence_stack[j+2][0])[0] # remove word-by-word numbering
            except IndexError:
                next_word2 = ("noword", "NOPOS")
            else:
                next_word2 = (shifted_word2, sentence_stack[j+2][1])
            
            try:
                shifted_word3 = re.split("\#.[0-9]*", sentence_stack[j+3][0])[0] # remove word-by-word numbering
            except IndexError:
                next_word3 = ("noword", "NOPOS")
            else:
                next_word3 = (shifted_word3, sentence_stack[j+3][1])

            last_action = _reduce(last_action)

            while j in gaps:
                full_gap_word, gap_tree = gaps[j].pop(0)
                gap_word = ("".join(re.split("-[0-9][0-9]*", str(full_gap_word[0]))), full_gap_word[1])
                _update_action_counter("postulate_gap->{}".format(re.split("\#.[0-9]*",gap_word[0])[0]), last_action)
                try:
                    trace = re.search("-[0-9][0-9]*$", str(full_gap_word[0])).group()
                except AttributeError:
                    pass
                else:
                    if trace in antecedents:
                        antecedents.remove(trace)
                    else:
                        traces.add(trace)
                last_action = "postulate_gap->{}".format(re.split("\#.[0-9]*",gap_word[0])[0])
                tree_stack.append(((gap_tree[0],), (gap_tree[1],), ("",), ("",)))
                last_action = _reduce(last_action)
                if gaps[j] == []:
                    gaps.pop(j)

            _update_action_counter("shift->''", last_action)
            last_action = "shift->''"
            tree_stack.append(((word,), (pos,), ("",), ("",)))

        #last word (usually, the period) is reduced outside of the loop
        _reduce(last_action)

        #for x in self.action_counter:
        #    print(x, self.action_counter[x])


        #check that every phrase was used (nothing was forgotten)
        #print(phrases_in_sent.values())
        assert not(any(phrases_in_sent.values())), "Not all phrases were consumed in the process"


    def collect_parses(self):
        """
        Collect parses for all sentences in _cnf_sentences.
        """
        total = len(self._cnf_sentences)
        rulecounter = 0
        badcounter = 0

        for sentence in ptb_grammar._cnf_sentences:

            rulecounter += 1
            progress(rulecounter, total) # Progress bar
            #ptb_grammar.parse_sent(sentence)
            #sentence.draw()
            try:
                ptb_grammar.parse_sent(sentence)
            except (AssertionError, ValueError, IndexError) as e:
                print(sentence.leaves())
                print(e)
                badcounter += 1

        all_actions = []

        for action in self.action_counter:
            if self.future_included:
                all_actions.append({'TREE0_LABEL': action.tree0_label, 'TREE0_HEAD': action.tree0_head, 'TREE0_HEADPOS': action.tree0_headpos,\
                    'TREE1_LABEL': action.tree1_label, 'TREE1_HEAD': action.tree1_head, 'TREE1_HEADPOS': action.tree1_headpos,\
                    'TREE2_LABEL': action.tree2_label, 'TREE2_HEAD': action.tree2_head, 'TREE2_HEADPOS': action.tree2_headpos,\
                    'TREE3_LABEL': action.tree3_label, 'TREE3_HEAD': action.tree3_head,\
                    'ANTECEDENT_CARRIED': action.antecedent_carried,\
                    'WORD_NEXT0_LEX': action.next_word0_lex, 'WORD_NEXT0_POS': action.next_word0_pos,\
                    'WORD_NEXT1_LEX': action.next_word1_lex, 'WORD_NEXT1_POS': action.next_word1_pos,\
                    'ACTION': action.action, 'ACTION_RESULT_LABEL': action.action_result_label,\
                    'ACTION_PREV': action.last_action,\
                    #'ACTION_RESULT_LABEL_PREV': action.last_action_result_label,\
                    "FREQ": self.action_counter[action],
                    "ACTIVATION": math.log(self.action_counter[action]/(1-DECAY)) - DECAY*math.log(SEC_IN_TIME * self.action_counter[action])})
                #print(all_actions)
            else:
                all_actions.append({'TREE0_LABEL': action.tree0_label, 'TREE0_HEAD': action.tree0_head, 'TREE0_HEADPOS': action.tree0_headpos, 'TREE0_LEFTCHILD': action.tree0_leftchild, 'TREE0_RIGHTCHILD': action.tree0_rightchild,\
                    'TREE1_LABEL': action.tree1_label, 'TREE1_HEAD': action.tree1_head, 'TREE1_HEADPOS': action.tree1_headpos, 'TREE1_LEFTCHILD': action.tree1_leftchild, 'TREE1_RIGHTCHILD': action.tree1_rightchild,\
                    'TREE2_LABEL': action.tree2_label, 'TREE2_HEAD': action.tree2_head, 'TREE2_HEADPOS': action.tree2_headpos,\
                    'TREE3_LABEL': action.tree3_label, 'TREE3_HEAD': action.tree3_head,\
                    'ANTECEDENT_CARRIED': action.antecedent_carried,\
                    'ACTION': action.action, 'ACTION_RESULT_LABEL': action.action_result_label,\
                    'ACTION_PREV': action.last_action,\
                    #'ACTION_RESULT_LABEL_PREV': action.last_action_result_label,\
                    "FREQ": self.action_counter[action],
                    "ACTIVATION": math.log(self.action_counter[action]/(1-DECAY)) - DECAY*math.log(SEC_IN_TIME * self.action_counter[action])})
                #print(all_actions)

        print(badcounter, total)

        return all_actions

# Construct the Penn Treebank Grammar
ptb_grammar = Trainer('Penn Treebank', lexicalized=True, tagset='lexicalized')

ptb_grammar.future_included = True
#ptb_grammar.current_action_state = namedtuple("current_action_state", "tree0_label, tree0_head, tree0_headpos, tree1_label, tree1_head, tree1_headpos, tree2_label, tree2_head, tree2_headpos, tree3_label, tree3_head, antecedent_carried, action, action_result_label, last_action")

print("Loading done.")

#ptb_grammar.parse_sent(ptb_grammar._cnf_sentences[28])
actions = ptb_grammar.collect_parses()
save_words(ptb_grammar._tagged_words)
save_labels(ptb_grammar.label_counter, ptb_grammar._tagged_words)

save_actions(actions, filename="actions.csv")
