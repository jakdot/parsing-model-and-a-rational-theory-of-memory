#### Info

This repo provides the code for the experiments in our paper:

Jakub Dotlacil, Puck de Haan - Parsing model and a rational theory of memory (2021)

#### Structure

All required packages can be found in ``requirements.txt``. The ``pyactr`` package can be installed with pip together with the other requirements, or directly from the source files in the **pyactr** folder. Instructions to do so can be found at <https://github.com/jakdot/pyactr>.

The folder **/examples_and_garden_paths** includes code to recreate the results of the parser on the garden path sentences as depicted in the paper. The parser can be run by executing the file ``run_parser_act.py``. All necessary data is present in the folder. Code for creating the plots can be found in ``plotting.rnw``.

The folder **/model_nsc_reading_data** includes code to recreate the experiments done with the NSC. To parse a sentence with the ACT-R based parsing model, one can first run the script ``run_parser.py`` -- which builds parse trees and generates activations. Then, it is possible to run the script ``run_parser_act.py``, which generates reading times given the data. Parameter estimation can be done by running the script ``parallel_estimation_smaller.py``. Estimation is done in a parallelized manner to speed up the process, we recommend the user to make use of this feature, since it decreases the runtime significantly. The subfolder **/generate_plots** contains the script to plot the results: ``plotting_csv.rnw`` and the resulting figures used in the paper. The other subfolder, **/parses**, contains the PTB parses, parse trees and sentence that have been excluded from the experiments.  It should be noted that the content of the ``actions.csv`` and ``blind_actions.csv`` files are based only on a small subset of the Penn Treebank (PTB) that is publicly available in the ``nltk`` package. If you require the entire PTB dataset, you can contact us.

The folder **/train_for_nsc** includes code to train the parsing model on PTB data.It should be noted that training is optional, since all data needed to recreate the experiments is already present in the **/model_nsc_reading_data** folder. 



