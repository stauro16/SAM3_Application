# =============================================================================
# CELL 1A — DEPENDENCY INSTALL
# =============================================================================
# EXPLANATION:
# This cell installs the Python packages required for the SAM3 notebook
# to run correctly in Databricks.
#
# WHAT THIS CELL DOES:
#   1) installs core SAM3-related dependencies
#   2) installs / reinstalls OpenCV and NumPy with a compatible version
#   3) prepares the Python environment before model imports
#
# IMPORTANT:
# - this cell should be run only once per cluster environment
# - a Python restart is required after installation
# - do not mix imports into this cell
# - later cells assume these packages are already installed
# =============================================================================

%pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    iopath timm decord pycocotools ftfy

%pip install opencv-python numpy==1.26.4 --force-reinstall


# =============================================================================
# CELL 1B — HARD RESTART PYTHON (MANDATORY)
# =============================================================================
# EXPLANATION:
# Databricks must restart the Python process so the newly installed packages
# are correctly loaded into the notebook session.
#
# IMPORTANT:
# - run this immediately after CELL 1A
# - re-run the notebook from CELL 2 after restart
# =============================================================================

dbutils.library.restartPython()



# =============================================================================
# CELL 2 — ENVIRONMENT + HUGGING FACE CACHE
# =============================================================================
# EXPLANATION:
# This cell configures the Hugging Face cache locations used by the notebook.
#
# WHAT THIS CELL DOES:
#   1) points HF_HUB_CACHE to a persistent Databricks Volume location
#   2) optionally keeps HF_HOME aligned with that cache root
#   3) defines a reusable CACHE_DIR variable for later cells
#
# IMPORTANT:
# - run this after the Python restart from CELL 1B
# - must be executed before any Hugging Face / model loading imports that use the cache
# - the Unity Catalog volume must already exist
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Configure Hugging Face cache paths
# -----------------------------------------------------------------------------
import os

# Replace these with your real UC volume path components.
VOLUME_ROOT = "/Volumes/<catalog>/<schema>/<volume>"

CACHE_DIR = os.path.join(VOLUME_ROOT, "hf_cache")
HUB_CACHE_DIR = os.path.join(CACHE_DIR, "hub")

# Optional:
# HF_HOME is the general Hugging Face home directory.
# Note: token files may also default under HF_HOME unless HF_TOKEN_PATH is set separately.
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = HUB_CACHE_DIR

# -----------------------------------------------------------------------------
# 2. Ensure cache directories exist
# -----------------------------------------------------------------------------
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)

# -----------------------------------------------------------------------------
# 3. Print configured cache paths
# -----------------------------------------------------------------------------
print("HF_HOME     :", os.environ["HF_HOME"])
print("HF_HUB_CACHE:", os.environ["HF_HUB_CACHE"])
print("CACHE_DIR   :", CACHE_DIR)




# =============================================================================
# CELL 3A — CORE IMPORTS + SYSTEM CHECKS
# =============================================================================
# EXPLANATION:
# This cell loads the core Python libraries needed for the notebook and performs
# a quick runtime sanity check.
#
# WHAT THIS CELL DOES:
#   1) imports standard library, data, image, and ML packages
#   2) applies global warning settings
#   3) confirms installed runtime versions
#   4) verifies whether CUDA / GPU is available
#
# IMPORTANT:
# - run this after CELL 2
# - this is an early sanity check before model setup
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Standard library imports
# -----------------------------------------------------------------------------
import os
import gc
import re
import io
import json
import time
import glob
import math
import shutil
import random
import zipfile
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# -----------------------------------------------------------------------------
# 2. Data handling imports
# -----------------------------------------------------------------------------
import pandas as pd

# -----------------------------------------------------------------------------
# 3. Data science and visualisation imports
# -----------------------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL
from PIL import Image, ImageOps

# -----------------------------------------------------------------------------
# 4. Machine learning imports
# -----------------------------------------------------------------------------
import torch
import torchvision

# -----------------------------------------------------------------------------
# 5. Global warning behaviour
# -----------------------------------------------------------------------------
# Optional: keep only if you intentionally want a quieter notebook.
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 6. Print runtime version information
# -----------------------------------------------------------------------------
print("PyTorch version     :", torch.__version__)
print("Torch CUDA build    :", torch.version.cuda)
print("TorchVision version :", torchvision.__version__)
print("NumPy version       :", np.__version__)
print("OpenCV version      :", cv2.__version__)
print("Pillow version      :", PIL.__version__)

# -----------------------------------------------------------------------------
# 7. Verify GPU / CUDA availability
# -----------------------------------------------------------------------------
print("CUDA available      :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA/GPU is not available. Please attach this notebook to a GPU cluster before running SAM3."
    )

print("CUDA device count   :", torch.cuda.device_count())
print("Current CUDA device :", torch.cuda.current_device())
print("CUDA device name    :", torch.cuda.get_device_name(0))




# =============================================================================
# CELL 3B — GLOBAL CONSTANTS + NOTEBOOK CONFIG
# =============================================================================
# EXPLANATION:
# This cell defines notebook-wide constants used across the Databricks SAM3
# workflow.
#
# WHAT THIS CELL DOES:
#   1) defines runtime / reproducibility constants
#   2) defines general debug controls
#   3) defines shared global SAM3 thresholds
#   4) defines pole-detection prompts and post-processing settings
#   5) defines production pole-gallery settings
#   6) defines single-image crop-box debug settings (CELL 15A)
#   7) defines fixed-canvas pole-top ROI settings (CELL 15B)
#   8) defines crossarm debug / filtering / display settings (CELL 16A)
#   9) defines shared visual and file-naming settings
#  10) creates a shared SAM3_TASK_CONFIG dictionary
#
# IMPORTANT:
# - run this after CELL 3A
# - later inference cells should read these values from globals()
# - path definitions remain in their existing path-specific cells
# - this is intended to be the master config cell for the notebook
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Runtime / reproducibility
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
DEVICE = "cuda"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# -----------------------------------------------------------------------------
# 2. General debug / notebook controls
# -----------------------------------------------------------------------------
# EXPLANATION:
# These control general notebook debug behaviour and image selection.
# -----------------------------------------------------------------------------
MAX_IMAGES_FOR_DEBUG_GALLERY = 8
DEBUG_ROW_INDEX = 4

# -----------------------------------------------------------------------------
# 3. Shared global SAM3 thresholds
# -----------------------------------------------------------------------------
# EXPLANATION:
# Keep one shared threshold definition so later cells do not hardcode values.
# TEXT threshold filters prompt matches.
# MASK threshold filters pixel masks during post-processing.
# -----------------------------------------------------------------------------
GLOBAL_TEXT_SCORE_THRESHOLD = 0.30
MASK_THRESHOLD = 0.50

# -----------------------------------------------------------------------------
# 4. Pole prompts + pole-detection config
# -----------------------------------------------------------------------------
# EXPLANATION:
# POLE_PROMPTS is used in the multi-image debug pole cells.
# POLE_PROMPT is the single production prompt for CELL 14C.
# -----------------------------------------------------------------------------
POLE_PROMPTS = [
    "utility pole",
]

POLE_PROMPT = "timber power pole"
POLE_TEXT_THRESHOLD = GLOBAL_TEXT_SCORE_THRESHOLD

# -----------------------------------------------------------------------------
# 5. Pole post-processing constants
# -----------------------------------------------------------------------------
# EXPLANATION:
# These are used for:
# - prefiltering weak / tiny / short / wide / non-vertical candidates
# - ranking the remaining candidates
# - shaft penalty handling
# -----------------------------------------------------------------------------
POLE_MIN_SCORE = 0.25
POLE_MIN_AREA_FRAC = 0.005      # 0.5% of image area
POLE_MIN_HEIGHT_FRAC = 0.15     # 15% of image height
POLE_MIN_ASPECT = 1.80          # bbox_h / bbox_w
POLE_MAX_WIDTH_FRAC = 0.08      # max 8% of image width
POLE_MAX_BOX_W_PX = 400         # absolute width guard

# Shaft-penalty constants
SHAFT_WIDTH_FRAC_THRESHOLD = 0.12
SHAFT_PENALTY_FACTOR = 0.40

# Final ranking weights
W_X_CENTER = 0.45
W_HEIGHT   = 0.30
W_AREA     = 0.10
W_CONF     = 0.10
W_EDGE     = 0.05

# -----------------------------------------------------------------------------
# 6. Production pole gallery controls (CELL 14C)
# -----------------------------------------------------------------------------
POLE_GALLERY_COUNT = 6

# -----------------------------------------------------------------------------
# 7. Single-image crop-box debug settings (CELL 15A)
# -----------------------------------------------------------------------------
# EXPLANATION:
# These control the one-image crop-box tuning view after pole selection.
# -----------------------------------------------------------------------------
POLE_ROI_DEBUG_ROW_INDEX = 0

EXPANDED_BOX_WIDTH_FACTOR_FROM_POLE_HEIGHT = 0.90
MIN_EXPANDED_BOX_WIDTH = 600

TOP_EXTRA_FACTOR_FROM_POLE_HEIGHT = 0.10
BOTTOM_EXTRA_FACTOR_FROM_POLE_HEIGHT = 0.20

MIN_TOP_EXTRA_PIXELS = 40
MIN_BOTTOM_EXTRA_PIXELS = 10

# -----------------------------------------------------------------------------
# 8. Fixed-canvas pole-top ROI settings (CELL 15B)
# -----------------------------------------------------------------------------
# EXPLANATION:
# These define the fixed-size saved ROI and padding behaviour for the Silver ROI
# generation stage.
# -----------------------------------------------------------------------------
FIXED_ROI_WIDTH = 2600
FIXED_ROI_HEIGHT = 2600
POLE_TOP_BUFFER_ABOVE = 350
PAD_RGB = (0, 0, 0)
POLE_ROI_GALLERY_COUNT = 6

# -----------------------------------------------------------------------------
# 9. Crossarm debug / inference prompt settings (CELL 16A)
# -----------------------------------------------------------------------------
# EXPLANATION:
# These define the single-prompt crossarm debug run on the saved pole ROI crop.
# -----------------------------------------------------------------------------
CROSSARM_ROI_DEBUG_ROW_INDEX = 0
CROSSARM_PROMPT_TEXT = "utility pole crossarm"
CROSSARM_TEXT_THRESHOLD = 0.30
RUN_PLOT_RESULTS_DIAGNOSTIC_CROSSARM = False

# -----------------------------------------------------------------------------
# 10. Crossarm filtering settings (CELL 16A)
# -----------------------------------------------------------------------------
# EXPLANATION:
# These drive:
# - containment suppression
# - main-cluster filtering
# - pole-mask overlap filtering
# - structure filtering
# - level dedupe
# - optional PCA screening
# -----------------------------------------------------------------------------

# Containment suppression
CONTAINMENT_THRESHOLD = 0.80
MIN_AREA_RATIO = 1.20
MIN_SCORE_ADVANTAGE = 0.0

# Main-cluster filtering
CENTER_DIST_FACTOR = 2.75

# Pole-mask overlap filtering
POLE_MASK_FILTER_ENABLED = True
POLE_OVERLAP_MIN_FRACTION = 0.05

# Structure filtering
CROSSARM_STRUCTURE_FILTER_ENABLED = True
CROSSARM_MIN_ASPECT_RATIO = 1.50
POLE_ATTACH_MARGIN_PX = 120
MIN_RELATIVE_WIDTH_TO_MAX = 0.55

# Level dedupe filtering
CROSSARM_LEVEL_FILTER_ENABLED = True
CROSSARM_LEVEL_BAND_FACTOR = 0.60
MAX_BOX_H_TO_MEDIAN_RATIO = 1.80
KEEP_PER_LEVEL = 1

# Optional PCA filtering
CROSSARM_PCA_FILTER_ENABLED = True
PCA_SUSPICIOUS_ASPECT_MAX = 2.20
PCA_SUSPICIOUS_HEIGHT_TO_MEDIAN_MIN = 1.15
PCA_SUSPICIOUS_REL_WIDTH_MAX = 0.85
PCA_MIN_MASK_PIXELS = 80
PCA_MIN_PC1_RATIO = 0.85
PCA_MIN_ANISOTROPY = 4.00

# -----------------------------------------------------------------------------
# 11. Crossarm display / stage-grid settings (CELL 16A)
# -----------------------------------------------------------------------------
CROSSARM_MASK_ALPHA = 0.40
POLE_MASK_ALPHA = 0.30
LABEL_BG = "#1E90FF"

SHOW_STAGE_GRID = True
GRID_FIGSIZE = (20, 10)

# -----------------------------------------------------------------------------
# 12. Visual troubleshooting constants
# -----------------------------------------------------------------------------
# EXPLANATION:
# These support the 3-stage pole overlay:
# - raw      -> yellow
# - kept     -> cyan
# - selected -> red
# -----------------------------------------------------------------------------
STAGE_COLORS = {
    "raw": "yellow",
    "kept": "cyan",
    "selected": "red",
}

STAGE_LINEWIDTHS = {
    "raw": 2,
    "kept": 2,
    "selected": 3,
}

# -----------------------------------------------------------------------------
# 13. Shared image-extension constants
# -----------------------------------------------------------------------------
VALID_IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
)

# -----------------------------------------------------------------------------
# 14. Image-id naming constants
# -----------------------------------------------------------------------------
IMAGE_ID_PREFIX = "img"

# -----------------------------------------------------------------------------
# 15. Shared task config dictionary
# -----------------------------------------------------------------------------
SAM3_TASK_CONFIG = {
    "runtime": {
        "random_seed": RANDOM_SEED,
        "device": DEVICE,
    },

    "debug": {
        "max_images_for_debug_gallery": MAX_IMAGES_FOR_DEBUG_GALLERY,
        "debug_row_index": DEBUG_ROW_INDEX,
        "pole_gallery_count": POLE_GALLERY_COUNT,
        "pole_roi_debug_row_index": POLE_ROI_DEBUG_ROW_INDEX,
        "pole_roi_gallery_count": POLE_ROI_GALLERY_COUNT,
        "crossarm_roi_debug_row_index": CROSSARM_ROI_DEBUG_ROW_INDEX,
        "show_stage_grid": SHOW_STAGE_GRID,
        "grid_figsize": GRID_FIGSIZE,
        "run_plot_results_diagnostic_crossarm": RUN_PLOT_RESULTS_DIAGNOSTIC_CROSSARM,
    },

    "thresholds": {
        "text_score_threshold": GLOBAL_TEXT_SCORE_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
        "pole_text_threshold": POLE_TEXT_THRESHOLD,
        "crossarm_text_threshold": CROSSARM_TEXT_THRESHOLD,
    },

    "pole_detection": {
        "prompts": POLE_PROMPTS,
        "prompt": POLE_PROMPT,
        "text_score_threshold": POLE_TEXT_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
    },

    "pole_postprocess": {
        "min_score": POLE_MIN_SCORE,
        "min_area_frac": POLE_MIN_AREA_FRAC,
        "min_height_frac": POLE_MIN_HEIGHT_FRAC,
        "min_aspect": POLE_MIN_ASPECT,
        "max_width_frac": POLE_MAX_WIDTH_FRAC,
        "max_box_w_px": POLE_MAX_BOX_W_PX,
        "shaft_width_frac_threshold": SHAFT_WIDTH_FRAC_THRESHOLD,
        "shaft_penalty_factor": SHAFT_PENALTY_FACTOR,
        "weights": {
            "x_center": W_X_CENTER,
            "height": W_HEIGHT,
            "area": W_AREA,
            "conf": W_CONF,
            "edge": W_EDGE,
        },
    },

    "pole_roi_debug": {
        "expanded_box_width_factor_from_pole_height": EXPANDED_BOX_WIDTH_FACTOR_FROM_POLE_HEIGHT,
        "min_expanded_box_width": MIN_EXPANDED_BOX_WIDTH,
        "top_extra_factor_from_pole_height": TOP_EXTRA_FACTOR_FROM_POLE_HEIGHT,
        "bottom_extra_factor_from_pole_height": BOTTOM_EXTRA_FACTOR_FROM_POLE_HEIGHT,
        "min_top_extra_pixels": MIN_TOP_EXTRA_PIXELS,
        "min_bottom_extra_pixels": MIN_BOTTOM_EXTRA_PIXELS,
    },

    "pole_roi_fixed_canvas": {
        "fixed_roi_width": FIXED_ROI_WIDTH,
        "fixed_roi_height": FIXED_ROI_HEIGHT,
        "pole_top_buffer_above": POLE_TOP_BUFFER_ABOVE,
        "pad_rgb": PAD_RGB,
    },

    "crossarm_detection": {
        "prompt_text": CROSSARM_PROMPT_TEXT,
        "text_threshold": CROSSARM_TEXT_THRESHOLD,
    },

    "crossarm_filters": {
        "containment_threshold": CONTAINMENT_THRESHOLD,
        "min_area_ratio": MIN_AREA_RATIO,
        "min_score_advantage": MIN_SCORE_ADVANTAGE,
        "center_dist_factor": CENTER_DIST_FACTOR,
        "pole_mask_filter_enabled": POLE_MASK_FILTER_ENABLED,
        "pole_overlap_min_fraction": POLE_OVERLAP_MIN_FRACTION,
        "structure_filter_enabled": CROSSARM_STRUCTURE_FILTER_ENABLED,
        "crossarm_min_aspect_ratio": CROSSARM_MIN_ASPECT_RATIO,
        "pole_attach_margin_px": POLE_ATTACH_MARGIN_PX,
        "min_relative_width_to_max": MIN_RELATIVE_WIDTH_TO_MAX,
        "level_filter_enabled": CROSSARM_LEVEL_FILTER_ENABLED,
        "crossarm_level_band_factor": CROSSARM_LEVEL_BAND_FACTOR,
        "max_box_h_to_median_ratio": MAX_BOX_H_TO_MEDIAN_RATIO,
        "keep_per_level": KEEP_PER_LEVEL,
        "pca_filter_enabled": CROSSARM_PCA_FILTER_ENABLED,
        "pca_suspicious_aspect_max": PCA_SUSPICIOUS_ASPECT_MAX,
        "pca_suspicious_height_to_median_min": PCA_SUSPICIOUS_HEIGHT_TO_MEDIAN_MIN,
        "pca_suspicious_rel_width_max": PCA_SUSPICIOUS_REL_WIDTH_MAX,
        "pca_min_mask_pixels": PCA_MIN_MASK_PIXELS,
        "pca_min_pc1_ratio": PCA_MIN_PC1_RATIO,
        "pca_min_anisotropy": PCA_MIN_ANISOTROPY,
    },

    "visual_debug": {
        "stage_colors": STAGE_COLORS,
        "stage_linewidths": STAGE_LINEWIDTHS,
        "crossarm_mask_alpha": CROSSARM_MASK_ALPHA,
        "pole_mask_alpha": POLE_MASK_ALPHA,
        "label_bg": LABEL_BG,
    },

    "files": {
        "valid_image_extensions": VALID_IMAGE_EXTENSIONS,
    },

    "naming": {
        "image_id_prefix": IMAGE_ID_PREFIX,
    },
}

# -----------------------------------------------------------------------------
# 16. Print summary
# -----------------------------------------------------------------------------
print("Global constants loaded.\n")

print("=" * 90)
print("RUNTIME / GENERAL")
print("=" * 90)
print(f"DEVICE                              : {DEVICE}")
print(f"RANDOM_SEED                         : {RANDOM_SEED}")
print(f"MAX_IMAGES_FOR_DEBUG_GALLERY        : {MAX_IMAGES_FOR_DEBUG_GALLERY}")
print(f"DEBUG_ROW_INDEX                     : {DEBUG_ROW_INDEX}")
print(f"GLOBAL_TEXT_SCORE_THRESHOLD         : {GLOBAL_TEXT_SCORE_THRESHOLD}")
print(f"MASK_THRESHOLD                      : {MASK_THRESHOLD}")

print("\n" + "=" * 90)
print("POLE DETECTION / POST-PROCESS")
print("=" * 90)
print(f"POLE_PROMPTS                        : {POLE_PROMPTS}")
print(f"POLE_PROMPT                         : {POLE_PROMPT}")
print(f"POLE_TEXT_THRESHOLD                 : {POLE_TEXT_THRESHOLD}")
print(f"POLE_MIN_SCORE                      : {POLE_MIN_SCORE}")
print(f"POLE_MIN_AREA_FRAC                  : {POLE_MIN_AREA_FRAC}")
print(f"POLE_MIN_HEIGHT_FRAC                : {POLE_MIN_HEIGHT_FRAC}")
print(f"POLE_MIN_ASPECT                     : {POLE_MIN_ASPECT}")
print(f"POLE_MAX_WIDTH_FRAC                 : {POLE_MAX_WIDTH_FRAC}")
print(f"POLE_MAX_BOX_W_PX                   : {POLE_MAX_BOX_W_PX}")
print(f"SHAFT_WIDTH_FRAC_THRESHOLD          : {SHAFT_WIDTH_FRAC_THRESHOLD}")
print(f"SHAFT_PENALTY_FACTOR                : {SHAFT_PENALTY_FACTOR}")
print(f"W_X_CENTER                          : {W_X_CENTER}")
print(f"W_HEIGHT                            : {W_HEIGHT}")
print(f"W_AREA                              : {W_AREA}")
print(f"W_CONF                              : {W_CONF}")
print(f"W_EDGE                              : {W_EDGE}")
print(f"POLE_GALLERY_COUNT                  : {POLE_GALLERY_COUNT}")

print("\n" + "=" * 90)
print("CELL 15A — SINGLE-IMAGE CROP BOX")
print("=" * 90)
print(f"POLE_ROI_DEBUG_ROW_INDEX            : {POLE_ROI_DEBUG_ROW_INDEX}")
print(f"EXPANDED_BOX_WIDTH_FACTOR_FROM_POLE_HEIGHT : {EXPANDED_BOX_WIDTH_FACTOR_FROM_POLE_HEIGHT}")
print(f"MIN_EXPANDED_BOX_WIDTH              : {MIN_EXPANDED_BOX_WIDTH}")
print(f"TOP_EXTRA_FACTOR_FROM_POLE_HEIGHT   : {TOP_EXTRA_FACTOR_FROM_POLE_HEIGHT}")
print(f"BOTTOM_EXTRA_FACTOR_FROM_POLE_HEIGHT: {BOTTOM_EXTRA_FACTOR_FROM_POLE_HEIGHT}")
print(f"MIN_TOP_EXTRA_PIXELS                : {MIN_TOP_EXTRA_PIXELS}")
print(f"MIN_BOTTOM_EXTRA_PIXELS             : {MIN_BOTTOM_EXTRA_PIXELS}")

print("\n" + "=" * 90)
print("CELL 15B — FIXED POLE-TOP ROI")
print("=" * 90)
print(f"FIXED_ROI_WIDTH                     : {FIXED_ROI_WIDTH}")
print(f"FIXED_ROI_HEIGHT                    : {FIXED_ROI_HEIGHT}")
print(f"POLE_TOP_BUFFER_ABOVE               : {POLE_TOP_BUFFER_ABOVE}")
print(f"PAD_RGB                             : {PAD_RGB}")
print(f"POLE_ROI_GALLERY_COUNT              : {POLE_ROI_GALLERY_COUNT}")

print("\n" + "=" * 90)
print("CELL 16A — CROSSARM DEBUG / FILTERS")
print("=" * 90)
print(f"CROSSARM_ROI_DEBUG_ROW_INDEX        : {CROSSARM_ROI_DEBUG_ROW_INDEX}")
print(f"CROSSARM_PROMPT_TEXT                : {CROSSARM_PROMPT_TEXT}")
print(f"CROSSARM_TEXT_THRESHOLD             : {CROSSARM_TEXT_THRESHOLD}")
print(f"RUN_PLOT_RESULTS_DIAGNOSTIC_CROSSARM: {RUN_PLOT_RESULTS_DIAGNOSTIC_CROSSARM}")
print(f"CONTAINMENT_THRESHOLD               : {CONTAINMENT_THRESHOLD}")
print(f"MIN_AREA_RATIO                      : {MIN_AREA_RATIO}")
print(f"MIN_SCORE_ADVANTAGE                 : {MIN_SCORE_ADVANTAGE}")
print(f"CENTER_DIST_FACTOR                  : {CENTER_DIST_FACTOR}")
print(f"POLE_MASK_FILTER_ENABLED            : {POLE_MASK_FILTER_ENABLED}")
print(f"POLE_OVERLAP_MIN_FRACTION           : {POLE_OVERLAP_MIN_FRACTION}")
print(f"CROSSARM_STRUCTURE_FILTER_ENABLED   : {CROSSARM_STRUCTURE_FILTER_ENABLED}")
print(f"CROSSARM_MIN_ASPECT_RATIO           : {CROSSARM_MIN_ASPECT_RATIO}")
print(f"POLE_ATTACH_MARGIN_PX               : {POLE_ATTACH_MARGIN_PX}")
print(f"MIN_RELATIVE_WIDTH_TO_MAX           : {MIN_RELATIVE_WIDTH_TO_MAX}")
print(f"CROSSARM_LEVEL_FILTER_ENABLED       : {CROSSARM_LEVEL_FILTER_ENABLED}")
print(f"CROSSARM_LEVEL_BAND_FACTOR          : {CROSSARM_LEVEL_BAND_FACTOR}")
print(f"MAX_BOX_H_TO_MEDIAN_RATIO           : {MAX_BOX_H_TO_MEDIAN_RATIO}")
print(f"KEEP_PER_LEVEL                      : {KEEP_PER_LEVEL}")
print(f"CROSSARM_PCA_FILTER_ENABLED         : {CROSSARM_PCA_FILTER_ENABLED}")
print(f"PCA_SUSPICIOUS_ASPECT_MAX           : {PCA_SUSPICIOUS_ASPECT_MAX}")
print(f"PCA_SUSPICIOUS_HEIGHT_TO_MEDIAN_MIN : {PCA_SUSPICIOUS_HEIGHT_TO_MEDIAN_MIN}")
print(f"PCA_SUSPICIOUS_REL_WIDTH_MAX        : {PCA_SUSPICIOUS_REL_WIDTH_MAX}")
print(f"PCA_MIN_MASK_PIXELS                 : {PCA_MIN_MASK_PIXELS}")
print(f"PCA_MIN_PC1_RATIO                   : {PCA_MIN_PC1_RATIO}")
print(f"PCA_MIN_ANISOTROPY                  : {PCA_MIN_ANISOTROPY}")
print(f"CROSSARM_MASK_ALPHA                 : {CROSSARM_MASK_ALPHA}")
print(f"POLE_MASK_ALPHA                     : {POLE_MASK_ALPHA}")
print(f"LABEL_BG                            : {LABEL_BG}")
print(f"SHOW_STAGE_GRID                     : {SHOW_STAGE_GRID}")
print(f"GRID_FIGSIZE                        : {GRID_FIGSIZE}")

print("\n" + "=" * 90)
print("FILES / NAMING / VISUALS")
print("=" * 90)
print(f"VALID_IMAGE_EXTENSIONS              : {VALID_IMAGE_EXTENSIONS}")
print(f"IMAGE_ID_PREFIX                     : {IMAGE_ID_PREFIX}")
print(f"STAGE_COLORS                        : {STAGE_COLORS}")
print(f"STAGE_LINEWIDTHS                    : {STAGE_LINEWIDTHS}")



# =============================================================================
# CELL 4 — ADD SAM3 REPO TO PYTHONPATH
# =============================================================================
# EXPLANATION:
# This cell makes the local SAM3 codebase importable inside the notebook.
#
# WHAT THIS CELL DOES:
#   1) defines the SAM3 repository root
#   2) validates that the repository root exists
#   3) adds the repository root to Python's import path
#   4) imports the sam3 package
#   5) prints the resolved sam3 package location for verification
#
# IMPORTANT:
# - this cell does not yet load model weights
# - this is required because the SAM3 code lives in a Databricks Volume,
#   not in a standard pip-installed package
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Standard import needed for path injection
# -----------------------------------------------------------------------------
import sys

# -----------------------------------------------------------------------------
# 2. Define and validate SAM3 repository root
# -----------------------------------------------------------------------------
# EXPLANATION:
# Python needs the repo root (the parent folder that contains the "sam3"
# package directory) in sys.path so that "import sam3" works correctly.
# -----------------------------------------------------------------------------
SAM3_REPO_ROOT = (
    "/Volumes/repos/sam3"
)

if not os.path.isdir(SAM3_REPO_ROOT):
    raise FileNotFoundError(
        f"SAM3 repository root not found: {SAM3_REPO_ROOT}"
    )

# -----------------------------------------------------------------------------
# 3. Add SAM3 repository root to PYTHONPATH
# -----------------------------------------------------------------------------
# EXPLANATION:
# Insert at the front so the local Databricks repo copy takes priority over any
# other sam3 package that might exist in the environment.
# -----------------------------------------------------------------------------
if SAM3_REPO_ROOT not in sys.path:
    sys.path.insert(0, SAM3_REPO_ROOT)

# -----------------------------------------------------------------------------
# 4. Import sam3 package
# -----------------------------------------------------------------------------
# EXPLANATION:
# This verifies that the local SAM3 code is visible to Python.
# -----------------------------------------------------------------------------
import sam3

# -----------------------------------------------------------------------------
# 5. Print resolved package location
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is a quick sanity check to confirm the notebook is importing SAM3 from
# the expected Databricks Volume path.
# -----------------------------------------------------------------------------
print("SAM3_REPO_ROOT  :", SAM3_REPO_ROOT)
print("sam3 loaded from:", sam3.__file__)



# =============================================================================
# CELL 5 — SAM3 IMPORTS
# =============================================================================
# EXPLANATION:
# This cell imports the SAM3-specific classes and helper functions used later
# for model creation, image processing, and visualisation.
#
# WHAT THIS CELL DOES:
#   1) validates that the SAM3 repo path setup has already run
#   2) imports the SAM3 model builder
#   3) imports the SAM3 processor used for image + prompt inference
#   4) imports helper functions for box conversion and visualisation
#
# IMPORTANT:
# - run this after CELL 4
# - this cell imports SAM3 code only; it does not yet build the model
# - general libraries such as PIL and matplotlib were already imported in CELL 3A
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Safety checks
# -----------------------------------------------------------------------------
if "SAM3_REPO_ROOT" not in globals():
    raise NameError(
        "SAM3_REPO_ROOT not found.\n"
        "Please run CELL 4 first."
    )

# -----------------------------------------------------------------------------
# 1. Import SAM3 model builder and processor
# -----------------------------------------------------------------------------
# EXPLANATION:
# These are the main SAM3 components used later to:
# - build the image model
# - prepare images for inference
# - apply text prompts
# -----------------------------------------------------------------------------
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# -----------------------------------------------------------------------------
# 2. Import SAM3 helper utilities
# -----------------------------------------------------------------------------
# EXPLANATION:
# These helper utilities are useful later for bounding box conversion,
# coordinate normalisation, and visualisation.
# -----------------------------------------------------------------------------
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import (
    draw_box_on_image,
    normalize_bbox,
    plot_results,
)

# -----------------------------------------------------------------------------
# 3. Print import confirmation
# -----------------------------------------------------------------------------
# EXPLANATION:
# This provides a simple sanity check that the key SAM3 components are available.
# -----------------------------------------------------------------------------
print("SAM3 imports ready.")
print("build_sam3_image_model loaded")
print("Sam3Processor loaded")
print("SAM3 utility functions loaded")



# =============================================================================
# CELL 6 — GPU SETTINGS + RUNTIME BEHAVIOUR
# =============================================================================
# EXPLANATION:
# This cell applies the GPU runtime settings used by the notebook.
#
# WHAT THIS CELL DOES:
#   1) validates that PyTorch is already available
#   2) enables TF32 for Ampere-class GPUs such as the A10
#   3) keeps inference in the default float32 mode
#   4) avoids precision changes that could affect model behaviour
#
# IMPORTANT:
# - run this after CELL 5
# - this cell does not build the SAM3 model yet
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Safety checks
# -----------------------------------------------------------------------------
if "torch" not in globals():
    raise NameError(
        "torch not found.\n"
        "Please run CELL 3A first."
    )

# -----------------------------------------------------------------------------
# 1. Enable TF32 on supported GPUs
# -----------------------------------------------------------------------------
# EXPLANATION:
# TF32 can improve matrix math performance on Ampere GPUs while keeping the
# notebook in standard float32 inference mode.
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------------------------------------------------------------------
# 2. Print runtime precision settings
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is a quick sanity check so you can confirm the runtime is configured
# the way you expect before model creation.
# -----------------------------------------------------------------------------
print("TF32 matmul enabled :", torch.backends.cuda.matmul.allow_tf32)
print("TF32 cuDNN enabled  :", torch.backends.cudnn.allow_tf32)
print("Default dtype       :", torch.get_default_dtype())



# =============================================================================
# CELL 7 — SAM3 PATH DEFINITIONS + FILE CHECKS
# =============================================================================
# EXPLANATION:
# This cell defines the key filesystem paths needed to build the SAM3 model.
#
# WHAT THIS CELL DOES:
#   1) validates that the SAM3 repo root from CELL 4 exists
#   2) derives the inner SAM3 code root from that repo root
#   3) defines the BPE tokenizer vocab path
#   4) defines the checkpoint / weight file path
#   5) checks that the required files actually exist
#
# IMPORTANT:
# - run this after CELL 6
# - this cell does not build the model yet
# - later cells assume these path variables are already defined
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Safety checks
# -----------------------------------------------------------------------------
required_globals = [
    "SAM3_REPO_ROOT",
]

missing_globals = [name for name in required_globals if name not in globals()]

if missing_globals:
    raise NameError(
        "CELL 7 cannot run because some required variables are missing.\n"
        "Please run the required earlier cells first.\n"
        f"Missing globals: {missing_globals}"
    )

# -----------------------------------------------------------------------------
# 1. Define SAM3 code root
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is the inner SAM3 package folder that contains assets such as the BPE
# vocabulary file used for text prompts.
# Deriving it from SAM3_REPO_ROOT keeps the notebook path setup consistent.
# -----------------------------------------------------------------------------
SAM3_CODE_ROOT = os.path.join(SAM3_REPO_ROOT, "sam3")

# -----------------------------------------------------------------------------
# 2. Define BPE vocab path
# -----------------------------------------------------------------------------
# EXPLANATION:
# This file is required because you are using text prompts with SAM3.
# -----------------------------------------------------------------------------
BPE_PATH = os.path.join(
    SAM3_CODE_ROOT,
    "assets",
    "bpe_simple_vocab_16e6.txt.gz",
)

# -----------------------------------------------------------------------------
# 3. Define model checkpoint path
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is the SAM3 weight file stored in the shared Databricks model cache.
# -----------------------------------------------------------------------------
CHECKPOINT_PATH = (
    "/Volumes/advanalytics_dev_catalog/"
    "models/hf_cache/hub/sam3/sam3.pt"
)

# -----------------------------------------------------------------------------
# 4. Validate required files
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is a quick sanity check before model creation so file path issues are
# caught early and clearly.
# -----------------------------------------------------------------------------
print("SAM3_CODE_ROOT exists :", os.path.isdir(SAM3_CODE_ROOT))
print("BPE_PATH exists       :", os.path.exists(BPE_PATH))
print("CHECKPOINT_PATH exists:", os.path.exists(CHECKPOINT_PATH))

if not os.path.isdir(SAM3_CODE_ROOT):
    raise FileNotFoundError(f"SAM3 code root folder not found: {SAM3_CODE_ROOT}")

if not os.path.exists(BPE_PATH):
    raise FileNotFoundError(f"BPE file not found: {BPE_PATH}")

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint file not found: {CHECKPOINT_PATH}")

# -----------------------------------------------------------------------------
# 5. Backward-compatible aliases
# -----------------------------------------------------------------------------
# EXPLANATION:
# Keep these aliases for any later cells that still reference the older names.
# This lets you clean up downstream cells gradually without breaking the notebook.
# -----------------------------------------------------------------------------
sam3_root = SAM3_CODE_ROOT
bpe_path = BPE_PATH
checkpoint_path = CHECKPOINT_PATH

# -----------------------------------------------------------------------------
# 6. Print summary
# -----------------------------------------------------------------------------
print("SAM3 path definitions ready.")
print(f"SAM3_REPO_ROOT : {SAM3_REPO_ROOT}")
print(f"SAM3_CODE_ROOT : {SAM3_CODE_ROOT}")
print(f"BPE_PATH       : {BPE_PATH}")
print(f"CHECKPOINT_PATH: {CHECKPOINT_PATH}")




# =============================================================================
# CELL 8 — BUILD SAM3 MODEL
# =============================================================================
# EXPLANATION:
# This cell builds the SAM3 image model using the previously defined
# tokenizer vocabulary path and checkpoint path.
#
# WHAT THIS CELL DOES:
#   1) validates that model-build inputs already exist
#   2) clears any stale CUDA memory
#   3) builds the SAM3 model from local code + local weights
#   4) moves the model to the configured device
#   5) switches the model into evaluation mode
#
# IMPORTANT:
# - run this after CELL 7
# - this cell builds the model only once per notebook session
# - this cell does not yet load an image or apply prompts
# - this uses the default float32 behaviour to stay aligned with your earlier setup
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Safety checks
# -----------------------------------------------------------------------------
required_globals = [
    "build_sam3_image_model",
    "BPE_PATH",
    "CHECKPOINT_PATH",
    "DEVICE",
]

missing_globals = [name for name in required_globals if name not in globals()]

if missing_globals:
    raise NameError(
        "CELL 8 cannot run because some required variables are missing.\n"
        "Please run the required earlier cells first.\n"
        f"Missing globals: {missing_globals}"
    )

# -----------------------------------------------------------------------------
# 1. Clear any stale CUDA cache
# -----------------------------------------------------------------------------
# EXPLANATION:
# This helps reduce the chance of memory fragmentation before model creation.
# -----------------------------------------------------------------------------
torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# 2. Build SAM3 image model
# -----------------------------------------------------------------------------
# EXPLANATION:
# The SAM3 model is created from:
# - local SAM3 Python code
# - local BPE vocab file
# - local checkpoint weights
# -----------------------------------------------------------------------------
model = build_sam3_image_model(
    bpe_path=BPE_PATH,
    checkpoint_path=CHECKPOINT_PATH
)

# -----------------------------------------------------------------------------
# 3. Move model to configured device
# -----------------------------------------------------------------------------
# EXPLANATION:
# This makes device placement explicit so the notebook does not rely on the
# builder function to place the model on GPU implicitly.
# -----------------------------------------------------------------------------
model = model.to(DEVICE)

# -----------------------------------------------------------------------------
# 4. Set model to evaluation mode
# -----------------------------------------------------------------------------
# EXPLANATION:
# Evaluation mode disables training-specific behaviours and is the correct mode
# for inference.
# -----------------------------------------------------------------------------
model.eval()

# -----------------------------------------------------------------------------
# 5. Print model build confirmation
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is a quick sanity check so you know the model is ready before moving on
# to image processing and prompts.
# -----------------------------------------------------------------------------
print("SAM3 model built successfully.")
print("Model type           :", type(model))
print("Model eval mode      :", not model.training)
print("Model parameter dtype:", next(model.parameters()).dtype)
print("Model device         :", next(model.parameters()).device)




# =============================================================================
# CELL 9 — BUILD SAM3 PROCESSOR
# =============================================================================
# EXPLANATION:
# This cell creates the SAM3 processor object used to prepare images and text
# prompts for inference.
#
# WHAT THIS CELL DOES:
#   1) validates that Sam3Processor has already been imported
#   2) validates that the SAM3 model has already been built
#   3) creates the processor object
#   4) prints a small confirmation
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Safety checks
# -----------------------------------------------------------------------------
if "Sam3Processor" not in globals():
    raise NameError(
        "Sam3Processor class not found.\n"
        "Please run CELL 5 first."
    )

if "model" not in globals():
    raise NameError(
        "SAM3 model not found.\n"
        "Please run CELL 8 first."
    )

if model is None:
    raise ValueError(
        "model exists but is None.\n"
        "Please rebuild the SAM3 model in CELL 8."
    )

# -----------------------------------------------------------------------------
# 1. Build processor
# -----------------------------------------------------------------------------
processor = Sam3Processor(model)

# -----------------------------------------------------------------------------
# 2. Print confirmation
# -----------------------------------------------------------------------------
print("SAM3 processor created successfully.")
print("Processor type:", type(processor))




# =============================================================================
# CELL 10 — WORKSPACE + DIRECTORY STRUCTURE (DATABRICKS)
# =============================================================================
# EXPLANATION:
# This cell defines the Databricks project workspace and creates the directory
# structure used by the notebook.
#
# WHAT THIS CELL DOES:
#   1) validates that core earlier setup has already run
#   2) defines the project root inside a Databricks Volume
#   3) defines state, artifact, Bronze, Silver, and Gold folders
#   4) creates the full project directory tree
#   5) prints a summary so the structure can be verified before ingestion
#
# IMPORTANT:
# - run this after CELL 9
# - this cell does not yet ingest images
# - path definitions remain centralized here rather than in CELL 3B
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Safety checks
# -----------------------------------------------------------------------------
required_globals = [
    "SAM3_TASK_CONFIG",
]

missing_globals = [name for name in required_globals if name not in globals()]

if missing_globals:
    raise NameError(
        "CELL 10 cannot run because some required variables are missing.\n"
        "Please run the required earlier cells first.\n"
        f"Missing globals: {missing_globals}"
    )

# -----------------------------------------------------------------------------
# 1. Define project workspace root
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is the main root folder for the SAM3 project inside your Databricks
# Volume.
# -----------------------------------------------------------------------------
WORK_DIR = (
    "/Volumes/"
    "sam3_project"
)

# -----------------------------------------------------------------------------
# 2. Define state / artifact folders
# -----------------------------------------------------------------------------
# EXPLANATION:
# These folders hold notebook state, saved tabular manifests, and general
# output artifacts.
# -----------------------------------------------------------------------------
STATE_DIR = os.path.join(WORK_DIR, "state")
DF_DIR    = os.path.join(STATE_DIR, "dataframes")
ART_DIR   = os.path.join(WORK_DIR, "artifacts")

# -----------------------------------------------------------------------------
# 3. Define Bronze layer folders
# -----------------------------------------------------------------------------
# EXPLANATION:
# Bronze holds raw input images before any processing or cropping.
# -----------------------------------------------------------------------------
BRONZE_ROOT          = os.path.join(WORK_DIR, "bronze")
BRONZE_SOURCE_IMAGES = os.path.join(BRONZE_ROOT, "source_images")

# -----------------------------------------------------------------------------
# 4. Define Silver layer folders
# -----------------------------------------------------------------------------
# EXPLANATION:
# Silver holds shared pole ROIs and intermediate branch outputs.
# -----------------------------------------------------------------------------
SILVER_ROOT      = os.path.join(WORK_DIR, "silver")
SILVER_POLE_ROIS = os.path.join(SILVER_ROOT, "pole_rois")

# Silver branch 1: asset detection candidates
SILVER_ASSET_DETECTION_CANDIDATES = os.path.join(
    SILVER_ROOT,
    "asset_detection_candidates",
)
SILVER_ASSET_PROMPT_RUNS = os.path.join(
    SILVER_ASSET_DETECTION_CANDIDATES,
    "prompt_runs",
)
SILVER_ASSET_OVERLAYS = os.path.join(
    SILVER_ASSET_DETECTION_CANDIDATES,
    "overlays",
)
SILVER_ASSET_MASKS = os.path.join(
    SILVER_ASSET_DETECTION_CANDIDATES,
    "masks",
)

# Silver branch 2: crossarm detection
SILVER_CROSSARM_DETECTION  = os.path.join(SILVER_ROOT, "crossarm_detection")
SILVER_CROSSARM_CANDIDATES = os.path.join(SILVER_CROSSARM_DETECTION, "candidates")
SILVER_CROSSARM_PROCESSING = os.path.join(SILVER_CROSSARM_DETECTION, "processing")
SILVER_CROSSARM_REVIEW     = os.path.join(SILVER_CROSSARM_DETECTION, "review")

# -----------------------------------------------------------------------------
# 5. Define Gold layer folders
# -----------------------------------------------------------------------------
# EXPLANATION:
# Gold holds final cleaned outputs ready for downstream analysis or export.
# -----------------------------------------------------------------------------
GOLD_ROOT                = os.path.join(WORK_DIR, "gold")
GOLD_ASSET_DETECTIONS    = os.path.join(GOLD_ROOT, "asset_detections")
GOLD_CROSSARM_DETECTIONS = os.path.join(GOLD_ROOT, "crossarm_detections")

# -----------------------------------------------------------------------------
# 6. Collect all directories to create
# -----------------------------------------------------------------------------
# EXPLANATION:
# Keeping the directory list in one place makes this cell easier to inspect,
# extend, and debug later.
# -----------------------------------------------------------------------------
DIRECTORIES_TO_CREATE = [
    WORK_DIR,
    STATE_DIR,
    DF_DIR,
    ART_DIR,
    BRONZE_ROOT,
    BRONZE_SOURCE_IMAGES,
    SILVER_ROOT,
    SILVER_POLE_ROIS,
    SILVER_ASSET_DETECTION_CANDIDATES,
    SILVER_ASSET_PROMPT_RUNS,
    SILVER_ASSET_OVERLAYS,
    SILVER_ASSET_MASKS,
    SILVER_CROSSARM_DETECTION,
    SILVER_CROSSARM_CANDIDATES,
    SILVER_CROSSARM_PROCESSING,
    SILVER_CROSSARM_REVIEW,
    GOLD_ROOT,
    GOLD_ASSET_DETECTIONS,
    GOLD_CROSSARM_DETECTIONS,
]

# -----------------------------------------------------------------------------
# 7. Create directory tree
# -----------------------------------------------------------------------------
# EXPLANATION:
# This ensures the full project folder structure exists before later cells try
# to write files into it.
# -----------------------------------------------------------------------------
for d in DIRECTORIES_TO_CREATE:
    os.makedirs(d, exist_ok=True)

# -----------------------------------------------------------------------------
# 8. Verify directory creation
# -----------------------------------------------------------------------------
# EXPLANATION:
# Fail early if any expected directory still does not exist after creation.
# -----------------------------------------------------------------------------
missing_dirs_after_create = [d for d in DIRECTORIES_TO_CREATE if not os.path.isdir(d)]

if missing_dirs_after_create:
    raise RuntimeError(
        "Some project directories were not created successfully.\n"
        f"Missing directories: {missing_dirs_after_create}"
    )

# -----------------------------------------------------------------------------
# 9. Print workspace summary
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is a quick sanity check to confirm the project structure is ready.
# -----------------------------------------------------------------------------
print("Databricks SAM3 project workspace ready.")
print(f"WORK_DIR                    : {WORK_DIR}")
print(f"STATE_DIR                   : {STATE_DIR}")
print(f"DF_DIR                      : {DF_DIR}")
print(f"ART_DIR                     : {ART_DIR}")
print(f"BRONZE_SOURCE_IMAGES        : {BRONZE_SOURCE_IMAGES}")
print(f"SILVER_POLE_ROIS            : {SILVER_POLE_ROIS}")
print(f"SILVER_ASSET_PROMPT_RUNS    : {SILVER_ASSET_PROMPT_RUNS}")
print(f"SILVER_ASSET_OVERLAYS       : {SILVER_ASSET_OVERLAYS}")
print(f"SILVER_ASSET_MASKS          : {SILVER_ASSET_MASKS}")
print(f"SILVER_CROSSARM_CANDIDATES  : {SILVER_CROSSARM_CANDIDATES}")
print(f"SILVER_CROSSARM_PROCESSING  : {SILVER_CROSSARM_PROCESSING}")
print(f"SILVER_CROSSARM_REVIEW      : {SILVER_CROSSARM_REVIEW}")
print(f"GOLD_ASSET_DETECTIONS       : {GOLD_ASSET_DETECTIONS}")
print(f"GOLD_CROSSARM_DETECTIONS    : {GOLD_CROSSARM_DETECTIONS}")



# -----------------------------------------------------------------------------
# Run controls
# -----------------------------------------------------------------------------
# EXPLANATION:
# Keep the Bronze overwrite control outside the main cell body so it can be
# changed easily before rerunning ingestion.
# -----------------------------------------------------------------------------
OVERWRITE_BRONZE = True   # set to False to protect Bronze


# =============================================================================
# CELL 11 — DATA INGESTION: SCAN VOLUME INTO BRONZE
# =============================================================================
# EXPLANATION:
# This cell performs Databricks-native raw image ingestion from a source Volume.
#
# WHAT THIS CELL DOES:
#   1) validates that the project workspace has already been created
#   2) validates that shared image extensions are already defined
#   3) defines the source image folder inside a Databricks Volume
#   4) scans the source folder recursively for supported image files
#   5) clears and rebuilds Bronze/source_images for a clean ingest
#   6) copies source images into Bronze while preserving folder structure
#   7) builds images_df as the raw image manifest for downstream processing
#
# IMPORTANT:
# - Volumes are treated as persistent storage
# - Bronze should contain only the raw working copy for this project
# - this cell is intentionally responsible for source ingestion path definition
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Safety checks
# -----------------------------------------------------------------------------
# EXPLANATION:
# This cell depends on the project workspace variables created in CELL 10
# and the shared file-extension constants created in CELL 3B.
# If they are missing, fail early with a clear message.
# -----------------------------------------------------------------------------
required_globals = [
    "SAM3_TASK_CONFIG",
    "WORK_DIR",
    "BRONZE_ROOT",
    "BRONZE_SOURCE_IMAGES",
    "VALID_IMAGE_EXTENSIONS",
]

missing_globals = [name for name in required_globals if name not in globals()]

if missing_globals:
    raise NameError(
        "CELL 11 cannot run because some required variables are missing.\n"
        "Please run the required earlier cells first.\n"
        f"Missing globals: {missing_globals}"
    )

# -----------------------------------------------------------------------------
# 1. Define source image folder
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is the external source folder that contains the raw images you want to
# ingest into the Bronze layer for this project.
# -----------------------------------------------------------------------------
SOURCE_IMAGE_ROOT = (
    "/Volumes/"
    "test_images"
)

if not os.path.isdir(SOURCE_IMAGE_ROOT):
    raise FileNotFoundError(
        f"Source image folder does not exist:\n{SOURCE_IMAGE_ROOT}"
    )

print("SOURCE_IMAGE_ROOT:", SOURCE_IMAGE_ROOT)

# -----------------------------------------------------------------------------
# 2. Reuse shared supported image extensions
# -----------------------------------------------------------------------------
# EXPLANATION:
# Use the single shared source of truth from CELL 3B so file-type support is
# defined in one place only.
# -----------------------------------------------------------------------------
IMAGE_EXTS_LOWER = tuple(sorted({ext.lower() for ext in VALID_IMAGE_EXTENSIONS}))

# -----------------------------------------------------------------------------
# 3. Read overwrite control
# -----------------------------------------------------------------------------
# EXPLANATION:
# Use globals().get(...) so reruns behave predictably even if the control was
# not manually redefined in the current session.
# -----------------------------------------------------------------------------
OVERWRITE_BRONZE = bool(globals().get("OVERWRITE_BRONZE", False))

# -----------------------------------------------------------------------------
# 4. Guard Bronze overwrite, then clear and recreate Bronze/source_images
# -----------------------------------------------------------------------------
# EXPLANATION:
# Bronze/source_images should reflect the current ingest only.
# We guard destructive rebuilds so CELL 11 is not accidentally rerun
# mid-pipeline.
# -----------------------------------------------------------------------------
if os.path.isdir(BRONZE_SOURCE_IMAGES):
    existing_bronze_items = os.listdir(BRONZE_SOURCE_IMAGES)

    if existing_bronze_items and not OVERWRITE_BRONZE:
        raise RuntimeError(
            "BRONZE_SOURCE_IMAGES already contains files.\n"
            "Re-running CELL 11 would overwrite the Bronze working copy.\n"
            "If you really want to rebuild Bronze, set:\n"
            "OVERWRITE_BRONZE = True\n"
            "and then run CELL 11 again."
        )

    if existing_bronze_items and OVERWRITE_BRONZE:
        shutil.rmtree(BRONZE_SOURCE_IMAGES)

os.makedirs(BRONZE_SOURCE_IMAGES, exist_ok=True)

# -----------------------------------------------------------------------------
# 5. Recursively discover source image files
# -----------------------------------------------------------------------------
# EXPLANATION:
# The source folder may contain nested folders, so we scan recursively and
# collect all supported image files.
# -----------------------------------------------------------------------------
source_image_files = []

for root, _, files in os.walk(SOURCE_IMAGE_ROOT):
    for fn in files:
        if fn.lower().endswith(IMAGE_EXTS_LOWER):
            source_image_files.append(os.path.join(root, fn))

source_image_files = sorted(source_image_files)

if len(source_image_files) == 0:
    raise ValueError(
        "No image files were found in the source image folder.\n"
        "Check the folder path and supported file extensions."
    )

print(f"Discovered source image count: {len(source_image_files)}")

# -----------------------------------------------------------------------------
# 6. Copy images into Bronze while preserving relative folder structure
# -----------------------------------------------------------------------------
# EXPLANATION:
# Preserving relative subfolders avoids accidental filename collisions when
# different source folders contain files with the same basename.
# -----------------------------------------------------------------------------
bronze_image_paths = []

for src_path in source_image_files:
    rel_path = os.path.relpath(src_path, SOURCE_IMAGE_ROOT)
    dst_path = os.path.join(BRONZE_SOURCE_IMAGES, rel_path)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)

    bronze_image_paths.append(dst_path)

print(f"Copied image count into Bronze: {len(bronze_image_paths)}")

# -----------------------------------------------------------------------------
# 7. Verify copied Bronze files
# -----------------------------------------------------------------------------
# EXPLANATION:
# Fail early if any copied output file is missing after the ingest step.
# -----------------------------------------------------------------------------
missing_bronze_files = [p for p in bronze_image_paths if not os.path.exists(p)]

if missing_bronze_files:
    raise RuntimeError(
        "Some Bronze image files were not copied successfully.\n"
        f"Missing copied files: {missing_bronze_files[:10]}"
    )

# -----------------------------------------------------------------------------
# 8. Build images_df manifest
# -----------------------------------------------------------------------------
# EXPLANATION:
# images_df becomes the raw image tracking table for downstream pipeline steps.
# At this stage, it contains one row per Bronze image and basic file metadata.
# -----------------------------------------------------------------------------
images_df = pd.DataFrame({
    "source_image_path": source_image_files,
    "image_path": bronze_image_paths,
})

images_df["relative_image_path"] = images_df["source_image_path"].map(
    lambda x: os.path.relpath(x, SOURCE_IMAGE_ROOT)
)
images_df["file_name"] = images_df["image_path"].map(os.path.basename)
images_df["stem"] = images_df["file_name"].map(lambda x: os.path.splitext(x)[0])
images_df["ext"] = images_df["file_name"].map(lambda x: os.path.splitext(x)[1])
images_df["source_layer"] = "bronze"
images_df["source_root"] = SOURCE_IMAGE_ROOT
images_df["bronze_root"] = BRONZE_SOURCE_IMAGES

# -----------------------------------------------------------------------------
# 9. Print summary and preview
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is a quick sanity check before later cells begin pole detection or
# image-level SAM3 processing.
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("BRONZE INGESTION SUMMARY (DATABRICKS)")
print("=" * 80)
print(f"SOURCE_IMAGE_ROOT    : {SOURCE_IMAGE_ROOT}")
print(f"BRONZE_SOURCE_IMAGES : {BRONZE_SOURCE_IMAGES}")
print(f"OVERWRITE_BRONZE     : {OVERWRITE_BRONZE}")
print(f"images_df shape      : {images_df.shape}")

display(images_df.head())