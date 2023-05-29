#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import os
import json
import numpy as np
import lightgbm as lgb
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# Local imports
if '.' in __name__:
    
    # To run as a module
    from .lib import config
    from .lib.data import get_map, get_image
    from .lib.plot import plot_maps

else:
    
    # To run as an script
    from lib import config
    from lib.data import get_map, get_image
    from lib.plot import plot_maps

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

def test_map(name, groups=10):
    """
    """
    
    # CONFIGURATION (extracted here as variables just for code clarity)
    # -------------------------------------------------------------------------
    
    # Input, output and dataset references
    d_path = config.DATA_PATH
    base_dir = config.MODELS_DIR
    datasets = config.DATASETS
    output_dir = config.TEST_DIR
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Maps colours
    colours = config.MAP_COLOURS
    
    # DATASET INFORMATION
    # -------------------------------------------------------------------------
    
    dataset = datasets[name]
    
    # Extract dataset classes
    num_classes = dataset['num_classes']
    num_features = dataset['num_features']
    p_train = dataset['p']
    
    # Get model dir
    model_dir = f"{name}_{p_train}train"
    model_dir = os.path.join(base_dir, model_dir)
    
    # GET DATA
    # -------------------------------------------------------------------------
    
    # Get dataset
    X, y, shape = get_map(dataset, d_path)
    
    # GENERATE OR LOAD MAP PREDICTIONS AND UNCERTAINTY
    # -------------------------------------------------------------------------
    
    # If prediction file already exist
    pred_map_file = os.path.join(model_dir, "pred_map.npy")
    if os.path.isfile(pred_map_file):
        
        # Load message
        print("\n### Loading {} map test".format(name))
        print('#'*80)
        print("\nMODEL DIR: {}".format(model_dir), flush=True)
        
        # Load them
        pred_map = np.load(pred_map_file)
    
    else:
        
        # Test model
        # ---------------------------------------------------------------------
        
        # Load model parameters
        model_file = os.path.join(model_dir, f"final.txt")
        model = lgb.Booster(model_file=model_file)
        
        # Map test message
        print("\n### Starting {} map test".format(name))
        print('#'*80)
        print("\nMODEL DIR: {}".format(model_dir))
        
        # Prediction
        prediction = model.predict(X)
        
        # Liberate model
        del model
        
        # Prediction map
        pred_map = prediction.argmax(axis=1)
        
        # Save prediction file
        np.save(os.path.join(model_dir, "pred_map"), pred_map)
    
    # PLOT MAP
    # -------------------------------------------------------------------------
    
    # Get image and wavelengths
    img, _ = get_image(dataset, d_path)
    wl = dataset['wl']
    
    # Plot
    plot_maps(output_dir, name, shape, num_classes, wl, img, y, pred_map,
              colours)

if __name__ == "__main__":
    
    # Parse args
    args = _parse_args()
    
    # Launch main function
    test_map(args.name)

