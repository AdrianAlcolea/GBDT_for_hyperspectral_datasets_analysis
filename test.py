#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Testing module of the gbdt4hi package

"""

import os
import lightgbm as lgb
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# Local imports
if '.' in __name__:
    
    # To run as a module
    from .lib import config
    from .lib.data import get_dataset

else:
    
    # To run as an script
    from lib import config
    from lib.data import get_dataset

# PREDICT FUNCTION
# =============================================================================

def predict(name, model, X_test, y_test):
    """Predicts the test evaluation data of the lightgbm model."""
    
    # Predict with test data
    test_pred = model.predict(X_test)
    
    # Get accuracy
    test_ok = (test_pred.argmax(axis=1) == y_test)
    test_accuracy = test_ok.sum() / test_ok.size
    
    # Write results messages
    print(f"{name} accuracy: {test_accuracy:.3f}")
    
    # Return prediction
    return test_pred

# MAIN FUNCTION
# =============================================================================

def test():
    """Bla bla"""
    
    # CONFIGURATION (extracted here as variables just for code clarity)
    # -------------------------------------------------------------------------
    
    # Input, output and dataset references
    d_path = config.DATA_PATH
    base_dir = config.MODELS_DIR
    datasets = config.DATASETS
    output_dir = config.TEST_DIR
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # FOR EVERY DATASET
    # -------------------------------------------------------------------------
    
    for name, dataset in datasets.items():
        
        # DATASET INFORMATION
        # ---------------------------------------------------------------------
        
        # Extract dataset characteristics and model parameters
        num_classes = dataset['num_classes']
        num_features = dataset['num_features']
        p_train = dataset['p']
        
        # Get model dir
        model_dir = f"{name}_{p_train}train"
        model_dir = os.path.join(base_dir, model_dir)
        
        # GET DATA
        # ---------------------------------------------------------------------
        
        # Get dataset
        _, _, X_test, y_test = get_dataset(dataset, d_path, p_train)
        
        # TEST MODEL (From lightgbm booster)
        # ---------------------------------------------------------------------
        
        # Load model
        model_file = os.path.join(model_dir, f"final.txt")
        model = lgb.Booster(model_file=model_file)
        
        # Prediction
        prediction = predict(name, model, X_test, y_test)
        
        del model

if __name__ == "__main__":
    
    # Launch main function
    test()

