"""
A bottom-up parser.
"""

import pyactr as actr

from parser_dm import parser

parser = parser

parser.productionstring(name="find first word", string="""
    =g>
    isa             parsing_goal
    task            reading_word
    ?visual_location>
    buffer          empty
    ==>
    ?visual_location>
    attended False
    +visual_location>
    isa _visuallocation
    screen_x lowest
    """)

parser.productionstring(name="find word", string="""
    =g>
    isa             reading
    state            reading_word
    ?visual_location>
    buffer          full
    =visual_location>
    isa    _visuallocation
    ?visual>
    state   free
    buffer  empty
    ==>
    =g>
    isa             reading
    state            reading_word
    +visual>
    isa     _visual
    cmd     move_attention
    screen_pos =visual_location""") #move attention to a located word

parser.productionstring(name="look back at word", string="""
    =g>
    isa         reading
    state       reading_word
    word        None
    ?visual>
    buffer  full
    =visual_location>
    isa    _visuallocation
    =visual>
    isa     _visual
    value   ___
    ==>
    ~visual>
    =g>
    isa         reading
    state       reading_word
    +visual>
    isa     _visual
    cmd     move_attention
    screen_pos =visual_location""") #move attention to a located word

parser.productionstring(name="recall word", string="""
    =g>
    isa         reading
    state       reading_word
    word        None
    ?visual>
    buffer  full
    =visual>
    isa     _visual
    value   =w
    value   ~___
    ?retrieval>
    state       free
    ==>
    =g>
    isa         reading
    state       recall_word
    retrieve_wh     None
    ~visual>
    +retrieval>
    isa         word
    form        =w""")

parser.productionstring(name="done recalling", string="""
    =g>
    isa         reading
    state       recall_word
    ?retrieval>
    state       free
    buffer      full
    =retrieval>
    isa         word
    form        ~None
    form        =w
    cat         =f
    =imaginal>
    isa         action_chunk
    WORD_NEXT0_LEX ~None
    WORD_NEXT0_LEX =nwl
    ==>
    =g>
    isa         reading
    state       done_recalling_word
    =imaginal>
    isa         action_chunk
    TREE0_HEAD  =w
    TREE0_LABEL =f
    +word_info>
    isa         word
    form        =w
    cat         =f
    """)

parser.productionstring(name="finish sentence", string="""
    =g>
    isa         reading
    state       recall_word
    ?retrieval>
    state       free
    buffer      full
    =retrieval>
    isa         word
    form        ~None
    form        =w
    cat         =f
    =imaginal>
    isa         action_chunk
    WORD_NEXT0_LEX None
    ==>
    =g>
    isa         reading
    state       move_to_last_action
    =imaginal>
    isa         action_chunk
    TREE0_HEAD  =w
    TREE0_LABEL =f
    +word_info>
    isa         word
    form        =w
    cat         =f
    """)

parser.productionstring(name="recall action", string="""
    =g>
    isa         reading
    state       done_recalling_word
    reanalysis  ~yes
    ==>
    =g>
    isa         reading
    state       recall_action
    ~visual>
    """)

parser.productionstring(name="reanalyse", string="""
    =g>
    isa         reading
    state       done_recalling_word
    reanalysis  yes
    ==>
    =g>
    isa         reading
    reanalysis  None
    """)

#for last word
parser.productionstring(name="move to last action", string="""
    =g>
    isa         reading
    state       move_to_last_action
    ==>
    =g>
    isa         reading
    state       recall_action
    =imaginal>
    isa         action_chunk
    WORD_NEXT0_POS   None
    """)

parser.productionstring(name="postulate_wh", string="""
    =g>
    isa             reading
    state           finished_recall
    retrieve_wh     quick
    ==>
    =g>
    isa             reading
    state           finished_recall
    retrieve_wh     None
""")

parser.productionstring(name="retrieve_wh", string="""
    =g>
    isa             reading
    state           finished_recall
    retrieve_wh     yes
    what_retrieve   =c
    ==>
    =g>
    isa             reading
    state           finished_recall
    retrieve_wh     None
    +retrieval>
    isa             action_chunk
    TREE0_LABEL     =c
""")

parser.productionstring(name="press spacebar", string="""
    =g>
    isa             reading
    state           finished_recall
    retrieve_wh     ~yes
    ?manual>
    state           free
    ?retrieval>
    state           free
    ==>
    =g>
    isa             reading
    state            move_eyes
    +manual>
    isa             _manual
    cmd             'press_key'
    key             'space'
""")

