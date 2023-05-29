#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot module of the bnn4hi package

The functions of this module are used to generate plots using the
results of the analysed bayesian predictions.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import os
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from .HSI2RGB import HSI2RGB

# MAP FUNCTIONS
# =============================================================================

def _map_to_img(prediction, shape, colours, metric=None, th=0.0, bg=(0, 0, 0)):
    """Generates an RGB image from `prediction` and `colours`
    
    The prediction itself should represent the index of its
    correspondent color.
    
    Parameters
    ----------
    prediction : array_like
        Array with the values to represent.
    shape : tuple of ints
        Original shape to reconstruct the image (without channels, just
        height and width).
    colours : list of RGB tuples
        List of colours for the RGB image representation.
    metric : array_like, optional (Default: None)
        Array with the same length of `prediction` to determine a
        metric for plotting or not each `prediction` value according to
        a threshold.
    th : float, optional (Default: 0.0)
        Threshold value to compare with each `metric` value if defined.
    bg : RGB tuple, optional (Default: (0, 0, 0))
        Background color used for the pixels not represented according
        to `metric`.
    
    Returns
    -------
    img : ndarray
        RGB image representation of `prediction` colouring each group
        according to `colours`.
    """
    
    # Generate RGB image shape
    img_shape = (shape[0], shape[1], 3)
    
    if metric is not None:
        
        # Coloured RGB image that only shows those values where metric
        # is lower to threshold
        return np.reshape([colours[int(p)] if m < th else bg
                           for p, m in zip(prediction, metric)], img_shape)
    else:
        
        # Coloured RGB image of the entire prediction
        return np.reshape([colours[int(p)] for p in prediction], img_shape)

# PLOT FUNCTIONS
# =============================================================================

def plot_reliability_diagram(output_dir, data, w, h, colours, num_groups=10):
    """Generates and saves the `reliability diagram` plot
    
    It saves the plot in `output_dir` in pdf format with the name
    `reliability_diagram.pdf`.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    data : dict
        It contains the `reliability diagram` data of each dataset. The
        key must be the dataset name abbreviation.
    w : int
        Width of the plot.
    h : int
        Height of the plot.
    colours : dict
        It contains the HEX value of the RGB color of each dataset. The
        key must be the dataset name abbreviation.
    num_groups : int, optional (default: 10)
        Number of groups to divide xticks labels.
    """
    
    # Generate x axis labels and data for the optimal calibration curve
    p_groups = np.linspace(0.0, 1.0, num_groups + 1)
    center = (p_groups[1] - p_groups[0]) / 2
    optimal = (p_groups + center)[:-1]
    if num_groups <= 10:
        labels = ["{:.1f}-{:.1f}".format(p_groups[i], p_groups[i + 1])
                  for i in range(num_groups)]
    else:
        labels = ["{:.2f}-{:.2f}".format(p_groups[i], p_groups[i + 1])
                  for i in range(num_groups)]
    
    # Xticks
    xticks = np.arange(len(labels))
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    for img_name in colours.keys():
        ax.plot(xticks[:len(data[img_name])], data[img_name], label=img_name,
                color=colours[img_name])
    ax.plot(xticks, optimal, label="Optimal calibration", color='black',
            linestyle='dashed')
    
    # Axes labels
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    
    # Y axis limit
    ax.set_ylim((0, 1))
    
    # X axis limit
    ax.set_xlim((xticks[0], xticks[-1]))
    
    # Rotate X axis labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Grid
    ax.grid(axis='y')
    
    # Legend
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    
    # Save
    file_name = "reliability_diagram.pdf"
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print("Saved {} in {}".format(file_name, output_dir), flush=True)

def plot_maps(output_dir, name, shape, num_classes, wl, img, y, pred_map,
              colours):
    """Generates and saves the `map` plot of a dataset
    
    This plot shows an RGB representation of the hyperspectral image,
    the ground truth and the prediction map.
    
    It saves the plot in `output_dir` in pdf format with the name
    `<NAME>_map.pdf`, where <NAME> is the abbreviation of the dataset
    name.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    name : str
        The abbreviation of the dataset name.
    shape : tuple of ints
        Original shape to reconstruct the image (without channels, just
        height and width).
    num_classes : int
        Number of classes of the dataset.
    wl : list of floats
        Selected wavelengths of the hyperspectral image for RGB
        representation.
    img : ndarray
        Flattened list of the hyperspectral image pixels normalised.
    y : ndarray
        Flattened ground truth pixels of the hyperspectral image.
    pred_map : ndarray
        Array with the averages of the bayesian predictions.
    colours : list of RGB tuples
        List of colours for the prediction map classes.
    """
    
    # PREPARE FIGURE
    # -------------------------------------------------------------------------
    
    # Select shape and size depending on the dataset
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(3*shape[1]/96, shape[0]/96)
    
    # Remove axis
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    
    # RGB IMAGE GENERATION
    #     Using HSI2RGB algorithm from paper:
    #         M. Magnusson, J. Sigurdsson, S. E. Armansson, M. O.
    #         Ulfarsson, H. Deborah and J. R. Sveinsson, "Creating RGB
    #         Images from Hyperspectral Images Using a Color Matching
    #         Function," IGARSS 2020 - 2020 IEEE International
    #         Geoscience and Remote Sensing Symposium, 2020,
    #         pp. 2045-2048, doi: 10.1109/IGARSS39084.2020.9323397.
    #     HSI2RGB code from:
    #         https://github.com/JakobSig/HSI2RGB
    # -------------------------------------------------------------------------
    
    # Create and show RGB image (D65 illuminant and 0.002 threshold)
    RGB_img = HSI2RGB(wl, img, shape[0], shape[1], 65, 0.002)
    ax1.imshow(RGB_img)
    
    # GROUND TRUTH GENERATION
    # -------------------------------------------------------------------------
    
    # Generate and show coloured ground truth
    gt = _map_to_img(y, shape, [(0, 0, 0)] + colours[:num_classes])
    ax2.imshow(gt)
    
    # PREDICTION MAP GENERATION
    # -------------------------------------------------------------------------
    
    # Generate and show coloured prediction map
    pred_H_img = _map_to_img(pred_map, shape, colours[:num_classes])
    ax3.imshow(pred_H_img)
    
    # PLOT COMBINED IMAGE
    # -------------------------------------------------------------------------
    
    # Adjust layout between images
    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    
    # Save
    file_name = f"{name}_map.pdf"
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print(f"Saved {file_name} in {output_dir}", flush=True)

