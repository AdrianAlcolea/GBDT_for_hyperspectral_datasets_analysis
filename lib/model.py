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
import lightgbm as lgb

# MODEL FUNCTION
# =============================================================================

def get_model(n_estimators, min_child_samples, max_depth):
    """Returns the lightgbm model"""
    
    # Generate model
    model = lgb.LGBMClassifier(objective='multiclass',
                               class_weight='balanced',
                               n_estimators=n_estimators,
                               min_child_samples=min_child_samples,
                               max_depth=max_depth,
                               num_leaves=2^max_depth)
    
    # Return the model
    return model
