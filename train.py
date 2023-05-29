#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Training module of the gbdt4hi package

This module contains the main function to train a gradient boosting
decision trees model for a hyperspectral image dataset.

This module can be imported as a part of the gbdt4hi package, but it
can also be launched from command line, as a script. For that, use the
`-h` option to see the required arguments.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# Local imports
if '.' in __name__:
    
    # To run as a module
    from .lib import config
    from .lib.data import get_dataset
    from .lib.model import get_model
    from .lib.gbdt import GBDT

else:
    
    # To run as an script
    from lib import config
    from lib.data import get_dataset
    from lib.model import get_model
    from lib.gbdt import GBDT

# PARAMETERS
# =============================================================================

def _parse_args():
    """Analyses the received parameters and returns them organised.
    
    Takes the list of strings received at sys.argv and generates a
    namespace assigning them to objects.
    
    Returns
    -------
    out : namespace
        The namespace with the values of the received parameters
        assigned to objects.
    """
    
    # Generate the parameter analyser
    parser = ArgumentParser(description = __doc__,
                            formatter_class = RawDescriptionHelpFormatter)
    
    # Add arguments
    parser.add_argument("name",
                        choices=config.DATASETS_LIST,
                        help="Abbreviated name of the dataset.")
    
    # Return the analysed parameters
    return parser.parse_args()

# MAIN FUNCTION
# =============================================================================

def train(name):
    """Trains a GBDT model for a hyperspectral image dataset"""
    
    # CONFIGURATION (extracted here as variables just for code clarity)
    # -------------------------------------------------------------------------
    
    # Input, output and dataset references
    d_path = config.DATA_PATH
    base_output_dir = config.MODELS_DIR
    datasets = config.DATASETS
    
    # DATASET INFORMATION
    # -------------------------------------------------------------------------
    
    dataset = datasets[name]
    
    # Extract dataset characteristics and model parameters
    num_classes = dataset['num_classes']
    num_features = dataset['num_features']
    p_train = dataset['p']
    n_estimators = dataset['n_estimators']
    min_child_samples = dataset['min_child_samples']
    max_depth = dataset['max_depth']
    
    # Generate output dir
    output_dir = f"{name}_{p_train}train"
    output_dir = os.path.join(base_output_dir, output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # GET DATA
    # -------------------------------------------------------------------------
    
    # Get dataset
    X_train, y_train, _, _ = get_dataset(dataset, d_path, p_train)
    
    # GET MODEL
    # -------------------------------------------------------------------------
    
    # Get model
    model = get_model(n_estimators, min_child_samples, max_depth)
    
    # TRAIN MODEL
    # -------------------------------------------------------------------------
    
    # Train
    model.fit(X_train, y_train)
    
    # SAVE MODEL
    # ---------------------------------------------------------------------
    
    # Save txt booster
    model.booster_.save_model(f"{output_dir}/final.txt")
    
    # Save trees list representation
    columns = ['tree_index', 'split_feature', 'threshold', 'value']
    dataframe = model.booster_.trees_to_dataframe()[columns]
    gbdt_manager = GBDT(num_classes, dataframe)
    gbdt_manager.save(f"{output_dir}/final.json")

if __name__ == "__main__":
    
    # Parse args
    args = _parse_args()
    
    # Launch main function
    train(args.name)

