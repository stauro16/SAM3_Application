PROD CODE SAM3
=====================
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
# - later cells assume these packages are already installed
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Install SAM3 runtime dependencies
# -----------------------------------------------------------------------------
# EXPLANATION:
# These packages support:
# - model utilities
# - image/video decoding
# - mask / detection utilities
# - text processing helpers used by the SAM3 codebase
# -----------------------------------------------------------------------------
%pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    iopath timm decord pycocotools ftfy

# -----------------------------------------------------------------------------
# 2. Install compatible OpenCV + NumPy versions
# -----------------------------------------------------------------------------
# EXPLANATION:
# NumPy and OpenCV version mismatches can cause ABI / import issues.
# We force a known compatible combination here.
# -----------------------------------------------------------------------------
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
#   1) points HF_HOME to the shared Databricks Volume cache
#   2) points HF_HUB_CACHE to the model hub cache location
#   3) defines a reusable CACHE_DIR variable for later cells
#
# IMPORTANT:
# - run this after the Python restart from CELL 1
# - must be executed before any model loading logic
# - does not install anything
# - keeps model artifacts in persistent Volume storage
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Configure Hugging Face cache paths
# -----------------------------------------------------------------------------
# EXPLANATION:
# These environment variables tell Hugging Face where to store and read cached
# model artifacts, configs, and downloaded files.
# -----------------------------------------------------------------------------
import os

os.environ["HF_HOME"] = (
    "/Volumes/models/hf_cache"
)

os.environ["HF_HUB_CACHE"] = (
    "/Volumes/models/hf_cache/hub"
)

# -----------------------------------------------------------------------------
# 2. Ensure cache directories exist
# -----------------------------------------------------------------------------
# EXPLANATION:
# Create the cache folders if they do not already exist so later model download
# and loading steps can safely use them.
# -----------------------------------------------------------------------------
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)

# -----------------------------------------------------------------------------
# 3. Define reusable cache directory variable
# -----------------------------------------------------------------------------
# EXPLANATION:
# CACHE_DIR is a simple convenience variable that later cells can reuse.
# -----------------------------------------------------------------------------
CACHE_DIR = os.environ["HF_HOME"]

# -----------------------------------------------------------------------------
# 4. Print configured cache paths
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is a quick sanity check so you can confirm the notebook is pointing to
# the expected persistent Databricks Volume locations.
# -----------------------------------------------------------------------------
print("HF_HOME    :", os.environ["HF_HOME"])
print("HF_HUB_CACHE:", os.environ["HF_HUB_CACHE"])
print("CACHE_DIR  :", CACHE_DIR)


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
#   1) defines runtime constants
#   2) defines notebook-specific constants
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
# 1. Runtime
# -----------------------------------------------------------------------------
DEVICE = "cuda"

# -----------------------------------------------------------------------------
# 2. Shared global SAM3 thresholds
# -----------------------------------------------------------------------------
# EXPLANATION:
# Keep one shared threshold definition so later cells do not hardcode values.
# TEXT threshold filters prompt matches.
# MASK threshold filters pixel masks during post-processing.
# -----------------------------------------------------------------------------
GLOBAL_TEXT_SCORE_THRESHOLD = 0.30
MASK_THRESHOLD = 0.50

# -----------------------------------------------------------------------------
# 3. Pole prompts + pole-detection config
# -----------------------------------------------------------------------------
# EXPLANATION:
# POLE_PROMPTS is used in the multi-image debug pole cells.
# POLE_PROMPT is the single production prompt for CELL 14C.
# -----------------------------------------------------------------------------
POLE_PROMPT_TEXT = ["utility pole"]
POLE_TEXT_THRESHOLD = GLOBAL_TEXT_SCORE_THRESHOLD

# -----------------------------------------------------------------------------
# 4. Pole post-processing constants
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

# ======================================================================================================================================
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
CROSSARM_PROMPT_TEXT = ["utility pole crossarm"]
CROSSARM_TEXT_THRESHOLD = GLOBAL_TEXT_SCORE_THRESHOLD
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

# ======================================================================================================================
# -----------------------------------------------------------------------------
# 15. Shared task config dictionary
# -----------------------------------------------------------------------------
SAM3_TASK_CONFIG = {
    "runtime": {
        "device": DEVICE,
    },

    "thresholds": {
        "text_score_threshold": GLOBAL_TEXT_SCORE_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
        "pole_text_threshold": POLE_TEXT_THRESHOLD,
        "crossarm_text_threshold": CROSSARM_TEXT_THRESHOLD,
    },

    "pole_detection": {
        "prompts": POLE_PROMPT_TEXT,
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
# ========================================================================================================================
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
print(f"GLOBAL_TEXT_SCORE_THRESHOLD         : {GLOBAL_TEXT_SCORE_THRESHOLD}")
print(f"MASK_THRESHOLD                      : {MASK_THRESHOLD}")

print("\n" + "=" * 90)
print("POLE DETECTION / POST-PROCESS")
print("=" * 90)
print(f"POLE_PROMPT_TEXT                    : {POLE_PROMPT_TEXT}")
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
# - this cell imports SAM3 code only; it does not yet build the model
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
# This is a quick sanity check so we can confirm the runtime is configured
# the way we expect before model creation.
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
    "/Volumes/models/hf_cache/hub/sam3/sam3.pt"
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
# 5. Print summary
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
# - this cell builds the model only once per notebook session
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
# This is a quick sanity check so we know the model is ready before moving on
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
    "/Volumes"
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
SILVER_POLE_SELECTION = os.path.join(SILVER_ROOT, "pole_selection")
SILVER_POLE_SELECTION_OVERLAYS = os.path.join(SILVER_ROOT,"pole_selection_overlays")

# Silver branch 1: asset detection candidates
SILVER_ASSET_DETECTION_CANDIDATES = os.path.join(SILVER_ROOT,"asset_detection_candidates")
SILVER_ASSET_PROMPT_RUNS = os.path.join(SILVER_ASSET_DETECTION_CANDIDATES,"prompt_runs")
SILVER_ASSET_OVERLAYS = os.path.join(SILVER_ASSET_DETECTION_CANDIDATES,"overlays")
SILVER_ASSET_MASKS = os.path.join(SILVER_ASSET_DETECTION_CANDIDATES,"masks")

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
    "sam3_project/"
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
# OVERWRITE_BRONZE = bool(globals().get("OVERWRITE_BRONZE", False))

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



# =============================================================================
# CELL 12 — PREPARE IMAGE DATAFRAME FOR PIPELINE
# =============================================================================
# EXPLANATION:
# This cell does NOT run SAM3 yet.
#
# It prepares the Bronze image manifest for downstream production pipeline
# processing.
#
# PURPOSE:
# - start from images_df created in CELL 11
# - validate that the required columns exist
# - create a clean working copy
# - ensure useful helper columns are present
# - sort images into a stable processing order
# - create stable image_id values for downstream tracking
#
# WHY THIS MATTERS:
# Later cells should not work directly on the raw images_df if we want:
# - a clean production working table
# - stable ordering across reruns
# - consistent IDs for saved outputs and joins
#
# OUTPUT:
# - run_images_df : cleaned production working image table
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Safety checks
# -----------------------------------------------------------------------------
# EXPLANATION:
# images_df should already exist from CELL 11.
# IMAGE_ID_PREFIX should already exist from CELL 3B.
# BRONZE_SOURCE_IMAGES should already exist from CELL 10.
# -----------------------------------------------------------------------------
required_globals = [
    "images_df",
    "IMAGE_ID_PREFIX",
    "BRONZE_SOURCE_IMAGES",
]

missing_globals = [name for name in required_globals if name not in globals()]

if missing_globals:
    raise NameError(
        "CELL 12 cannot run because some required variables are missing.\n"
        "Please run the required earlier cells first.\n"
        f"Missing globals: {missing_globals}"
    )

if not isinstance(images_df, pd.DataFrame):
    raise TypeError(
        "images_df exists but is not a pandas DataFrame."
    )

if images_df.empty:
    raise ValueError(
        "images_df exists but is empty.\n"
        "Please check CELL 11."
    )

# -----------------------------------------------------------------------------
# 1. Validate required columns
# -----------------------------------------------------------------------------
# EXPLANATION:
# image_path is the essential column because later cells use it to load images
# from Bronze for inference and output generation.
# -----------------------------------------------------------------------------
required_cols = ["image_path"]
missing_cols = [c for c in required_cols if c not in images_df.columns]

if missing_cols:
    raise ValueError(
        f"images_df is missing required columns: {missing_cols}"
    )

# -----------------------------------------------------------------------------
# 2. Create working copy
# -----------------------------------------------------------------------------
# EXPLANATION:
# Do not mutate the raw Bronze manifest directly.
# -----------------------------------------------------------------------------
run_images_df = images_df.copy()

# -----------------------------------------------------------------------------
# 3. Ensure helper columns exist
# -----------------------------------------------------------------------------
# EXPLANATION:
# These columns make later naming, sorting, tracking, and exports easier.
# -----------------------------------------------------------------------------
if "file_name" not in run_images_df.columns:
    run_images_df["file_name"] = run_images_df["image_path"].map(os.path.basename)

if "stem" not in run_images_df.columns:
    run_images_df["stem"] = run_images_df["file_name"].map(
        lambda x: os.path.splitext(x)[0] if isinstance(x, str) else None
    )

if "ext" not in run_images_df.columns:
    run_images_df["ext"] = run_images_df["file_name"].map(
        lambda x: os.path.splitext(x)[1] if isinstance(x, str) else None
    )

if "relative_image_path" not in run_images_df.columns:
    run_images_df["relative_image_path"] = run_images_df["image_path"].map(
        lambda x: os.path.relpath(x, BRONZE_SOURCE_IMAGES) if isinstance(x, str) else None
    )

# -----------------------------------------------------------------------------
# 4. Sort into stable processing order
# -----------------------------------------------------------------------------
# EXPLANATION:
# This makes production runs easier to compare across reruns.
# -----------------------------------------------------------------------------
run_images_df = run_images_df.sort_values(
    ["relative_image_path", "file_name", "image_path"]
).reset_index(drop=True)

# -----------------------------------------------------------------------------
# 5. Create stable unique image_id if missing
# -----------------------------------------------------------------------------
# EXPLANATION:
# image_id should remain unique even when different folders contain files with
# the same filename, so we append the row index.
# -----------------------------------------------------------------------------
if "image_id" not in run_images_df.columns:
    safe_stems = (
        run_images_df["stem"]
        .fillna("image")
        .astype(str)
        .str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    run_images_df["image_id"] = [
        f"{IMAGE_ID_PREFIX}_{stem}_{idx}"
        for idx, stem in enumerate(safe_stems)
    ]

# -----------------------------------------------------------------------------
# 6. Reorder key columns for readability
# -----------------------------------------------------------------------------
# EXPLANATION:
# Put the most commonly used tracking columns first.
# -----------------------------------------------------------------------------
preferred_front_cols = [
    "image_id",
    "file_name",
    "stem",
    "ext",
    "relative_image_path",
    "image_path",
    "source_image_path",
    "source_layer",
    "source_root",
    "bronze_root",
]

existing_front_cols = [c for c in preferred_front_cols if c in run_images_df.columns]
remaining_cols = [c for c in run_images_df.columns if c not in existing_front_cols]

run_images_df = run_images_df[existing_front_cols + remaining_cols]

# -----------------------------------------------------------------------------
# 7. Print summary
# -----------------------------------------------------------------------------
# EXPLANATION:
# This provides a quick overview before later pipeline inference cells.
# -----------------------------------------------------------------------------
print("Production image manifest preparation complete.\n")
print(f"  run_images_df rows : {len(run_images_df)}")
print(f"  IMAGE_ID_PREFIX    : {IMAGE_ID_PREFIX}")

# -----------------------------------------------------------------------------
# 8. Preview table
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is a quick sanity check before later cells begin SAM3 inference.
# -----------------------------------------------------------------------------
print("\nrun_images_df preview:")
display(run_images_df.head(10))



# =============================================================================
# CELL 13 — PRODUCTION POLE SELECTION + SAVED OVERLAYS
# =============================================================================
# EXPLANATION:
# This cell runs production pole selection across all images in run_images_df
# and saves one final pole overlay image per input image.
#
# WHAT THIS CELL DOES:
#   1) validates that the production inputs already exist
#   2) reads the production pole prompt list and pole post-processing settings
#   3) runs SAM3 on every image in run_images_df
#   4) collects all raw pole candidates across all configured pole prompts
#   5) normalises boxes, scores, and masks into a stable production format
#   6) computes per-image pole ranking features
#   7) prefilters weak / tiny / short / wide / non-vertical candidates
#   8) scores remaining candidates using shaft-penalty weighting
#   9) selects one best pole per image when available
#  10) builds and saves one final pole overlay image per input image
#  11) saves production outputs for downstream ROI and crossarm stages
#
# OUTPUTS:
# - pole_candidates_df : all scored pole candidates across all run images
# - pole_selection_df  : one row per image, with either a selected pole
#                        or a no-reliable-pole status row
# - pole_mask_lookup   : mask lookup keyed by (image_id, prompt, det_idx)
#
# IMPORTANT:
# - this is the production replacement for the old debug pole cells
# - this cell does NOT use debug_images_df
# - this cell does NOT produce galleries, troubleshooting overlays, or
#   plot_results(...) diagnostics
# - this cell now saves a final selected-pole overlay image for every input image
# - POLE_TEXT_THRESHOLD is kept as config for consistency / persistence;
#   the processor confidence threshold set earlier is what actually governs
#   low-confidence filtering in this stateful flow
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Safety checks
# -----------------------------------------------------------------------------
required_globals = [
    "run_images_df",
    "model",
    "processor",
    "DEVICE",
    "SILVER_ROOT",
    "GLOBAL_TEXT_SCORE_THRESHOLD",
    "POLE_PROMPT_TEXT",
    "POLE_TEXT_THRESHOLD",
    "POLE_MIN_SCORE",
    "POLE_MIN_AREA_FRAC",
    "POLE_MIN_HEIGHT_FRAC",
    "POLE_MIN_ASPECT",
    "POLE_MAX_WIDTH_FRAC",
    "POLE_MAX_BOX_W_PX",
    "SHAFT_WIDTH_FRAC_THRESHOLD",
    "SHAFT_PENALTY_FACTOR",
    "W_X_CENTER",
    "W_HEIGHT",
    "W_AREA",
    "W_CONF",
    "W_EDGE",
]

missing_globals = [name for name in required_globals if name not in globals()]

if missing_globals:
    raise NameError(
        "CELL 13 cannot run because some required variables are missing.\n"
        "Please run the required earlier cells first.\n"
        f"Missing globals: {missing_globals}"
    )

if not isinstance(run_images_df, pd.DataFrame):
    raise TypeError("run_images_df exists but is not a pandas DataFrame.")

if run_images_df.empty:
    raise ValueError("run_images_df is empty. Please check CELL 12.")

# -----------------------------------------------------------------------------
# 1. Production config
# -----------------------------------------------------------------------------
RUN_DEVICE = DEVICE
POLE_MASK_ALPHA = float(globals().get("POLE_MASK_ALPHA", 0.30))

# EXPLANATION:
# Keep the pole prompt config as a list so additional prompts can be added later.
# Normalize safely in case a single string is ever passed by mistake.
if isinstance(POLE_PROMPT_TEXT, (list, tuple)):
    pole_prompt_texts = [str(p).strip() for p in POLE_PROMPT_TEXT if str(p).strip()]
else:
    pole_prompt_texts = [str(POLE_PROMPT_TEXT).strip()]

if len(pole_prompt_texts) == 0:
    raise ValueError(
        "POLE_PROMPT_TEXT does not contain any usable prompt strings."
    )

POLE_TEXT_THRESHOLD = float(POLE_TEXT_THRESHOLD)

POLE_MIN_SCORE = float(POLE_MIN_SCORE)
POLE_MIN_AREA_FRAC = float(POLE_MIN_AREA_FRAC)
POLE_MIN_HEIGHT_FRAC = float(POLE_MIN_HEIGHT_FRAC)
POLE_MIN_ASPECT = float(POLE_MIN_ASPECT)
POLE_MAX_WIDTH_FRAC = float(POLE_MAX_WIDTH_FRAC)
POLE_MAX_BOX_W_PX = float(POLE_MAX_BOX_W_PX)

SHAFT_WIDTH_FRAC_THRESHOLD = float(SHAFT_WIDTH_FRAC_THRESHOLD)
SHAFT_PENALTY_FACTOR = float(SHAFT_PENALTY_FACTOR)

W_X_CENTER = float(W_X_CENTER)
W_HEIGHT   = float(W_HEIGHT)
W_AREA     = float(W_AREA)
W_CONF     = float(W_CONF)
W_EDGE     = float(W_EDGE)

# -----------------------------------------------------------------------------
# 2. Overlay output folders
# -----------------------------------------------------------------------------
# EXPLANATION:
# Use explicit folder globals if they already exist from CELL 10.
# Otherwise derive a clean Silver folder path here.
# -----------------------------------------------------------------------------
POLE_SELECTION_ROOT = globals().get(
    "SILVER_POLE_SELECTION",
    os.path.join(SILVER_ROOT, "pole_selection"),
)

POLE_SELECTION_OVERLAY_DIR = globals().get(
    "SILVER_POLE_SELECTION_OVERLAYS",
    os.path.join(POLE_SELECTION_ROOT, "overlays"),
)

os.makedirs(POLE_SELECTION_ROOT, exist_ok=True)
os.makedirs(POLE_SELECTION_OVERLAY_DIR, exist_ok=True)

print("Production pole-selection config:\n")
print(f"  RUN_DEVICE                  : {RUN_DEVICE}")
print(f"  POLE_PROMPT_TEXT            : {pole_prompt_texts}")
print(f"  POLE_TEXT_THRESHOLD         : {POLE_TEXT_THRESHOLD}")
print(f"  POLE_MIN_SCORE              : {POLE_MIN_SCORE}")
print(f"  POLE_MIN_AREA_FRAC          : {POLE_MIN_AREA_FRAC}")
print(f"  POLE_MIN_HEIGHT_FRAC        : {POLE_MIN_HEIGHT_FRAC}")
print(f"  POLE_MIN_ASPECT             : {POLE_MIN_ASPECT}")
print(f"  POLE_MAX_WIDTH_FRAC         : {POLE_MAX_WIDTH_FRAC}")
print(f"  POLE_MAX_BOX_W_PX           : {POLE_MAX_BOX_W_PX}")
print(f"  SHAFT_WIDTH_FRAC_THRESHOLD  : {SHAFT_WIDTH_FRAC_THRESHOLD}")
print(f"  SHAFT_PENALTY_FACTOR        : {SHAFT_PENALTY_FACTOR}")
print(f"  POLE_MASK_ALPHA             : {POLE_MASK_ALPHA}")
print(f"  POLE_SELECTION_OVERLAY_DIR  : {POLE_SELECTION_OVERLAY_DIR}")
print(f"  run_images_df rows          : {len(run_images_df)}")

# -----------------------------------------------------------------------------
# 3. Helpers
# -----------------------------------------------------------------------------
def _to_numpy_safe(x):
    """
    Convert tensors / arrays safely to numpy.

    Args:
        x:
            Tensor / array-like object or None.

    Returns:
        Numpy array or None.
    """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _candidate_key(image_id, prompt, det_idx):
    """
    Build a stable lookup key for one detection.

    Args:
        image_id:
            Stable image identifier.
        prompt:
            Prompt string used for the detection.
        det_idx:
            Detection index within the prompt run.

    Returns:
        Tuple key suitable for mask lookup dictionaries.
    """
    return (str(image_id), str(prompt), int(det_idx))

def _infer_num_detections(raw_boxes, raw_scores, raw_masks):
    """
    Infer how many detections are present from boxes, scores, or masks.

    Args:
        raw_boxes:
            Raw boxes object from processor state.
        raw_scores:
            Raw scores object from processor state.
        raw_masks:
            Raw masks object from processor state.

    Returns:
        Integer number of detections inferred from available outputs.
    """
    boxes_arr = _to_numpy_safe(raw_boxes)
    scores_arr = _to_numpy_safe(raw_scores)

    if boxes_arr is not None:
        if boxes_arr.ndim == 2 and boxes_arr.shape[1] == 4:
            return int(boxes_arr.shape[0])
        if boxes_arr.ndim == 1 and boxes_arr.size == 4:
            return 1

    if scores_arr is not None:
        scores_arr = scores_arr.reshape(-1)
        if scores_arr.size > 0:
            return int(scores_arr.size)

    if raw_masks is not None:
        if isinstance(raw_masks, (list, tuple)):
            return len(raw_masks)

        masks_arr = _to_numpy_safe(raw_masks)
        if masks_arr is not None:
            if masks_arr.ndim == 2:
                return 1
            if masks_arr.ndim >= 3:
                return int(masks_arr.shape[0])

    return 0

def _normalize_boxes_local(raw_boxes, num_detections):
    """
    Normalize raw boxes into a stable (N, 4) float32 array.

    Args:
        raw_boxes:
            Raw boxes object from processor state.
        num_detections:
            Expected number of detections.

    Returns:
        Float32 numpy array of shape (num_detections, 4).
    """
    if num_detections <= 0:
        return np.zeros((0, 4), dtype=np.float32)

    if raw_boxes is None:
        return np.zeros((num_detections, 4), dtype=np.float32)

    arr = _to_numpy_safe(raw_boxes).astype(np.float32)

    if arr.ndim == 1 and arr.size == 4:
        arr = arr.reshape(1, 4)

    if arr.ndim != 2 or arr.shape[1] != 4:
        return np.zeros((num_detections, 4), dtype=np.float32)

    if arr.shape[0] < num_detections:
        pad = np.zeros((num_detections - arr.shape[0], 4), dtype=np.float32)
        arr = np.vstack([arr, pad])

    return arr[:num_detections]

def _normalize_scores_local(raw_scores, num_detections):
    """
    Normalize raw scores into a stable (N,) float32 array.

    Args:
        raw_scores:
            Raw scores object from processor state.
        num_detections:
            Expected number of detections.

    Returns:
        Float32 numpy array of shape (num_detections,).
    """
    if num_detections <= 0:
        return np.zeros((0,), dtype=np.float32)

    if raw_scores is None:
        return np.zeros((num_detections,), dtype=np.float32)

    arr = _to_numpy_safe(raw_scores).astype(np.float32).reshape(-1)

    if arr.size < num_detections:
        pad = np.zeros((num_detections - arr.size,), dtype=np.float32)
        arr = np.concatenate([arr, pad])

    return arr[:num_detections]

def _normalize_masks_local(raw_masks, num_detections, image_h, image_w):
    """
    Normalize raw masks into a list of 2D boolean masks.

    Args:
        raw_masks:
            Raw masks object from processor state.
        num_detections:
            Expected number of detections.
        image_h:
            Image height.
        image_w:
            Image width.

    Returns:
        List of length num_detections containing either 2D boolean masks or None.
    """
    if num_detections <= 0:
        return []

    if raw_masks is None:
        return [None] * num_detections

    if isinstance(raw_masks, (list, tuple)):
        mask_items = list(raw_masks)
    else:
        arr = _to_numpy_safe(raw_masks)

        if arr is None:
            return [None] * num_detections

        if arr.ndim == 2:
            mask_items = [arr]
        elif arr.ndim == 3:
            if arr.shape[0] == num_detections:
                mask_items = [arr[i] for i in range(arr.shape[0])]
            else:
                mask_items = [arr[i] for i in range(min(arr.shape[0], num_detections))]
        elif arr.ndim == 4:
            if arr.shape[0] == num_detections:
                mask_items = [arr[i] for i in range(arr.shape[0])]
            else:
                mask_items = [arr[i] for i in range(min(arr.shape[0], num_detections))]
        else:
            return [None] * num_detections

    norm_masks = []

    for det_idx in range(num_detections):
        if det_idx >= len(mask_items):
            norm_masks.append(None)
            continue

        m = _to_numpy_safe(mask_items[det_idx])

        if m is None:
            norm_masks.append(None)
            continue

        m = np.squeeze(m)

        if m.ndim != 2:
            norm_masks.append(None)
            continue

        if m.shape != (image_h, image_w):
            norm_masks.append(None)
            continue

        mask_bool = m.copy() if m.dtype == bool else (m > 0)

        if mask_bool.sum() == 0:
            norm_masks.append(None)
        else:
            norm_masks.append(mask_bool)

    if len(norm_masks) < num_detections:
        norm_masks.extend([None] * (num_detections - len(norm_masks)))

    return norm_masks[:num_detections]

def _clip_box_to_image(x1, y1, x2, y2, image_w, image_h):
    """
    Clip and sort box coordinates to stay inside image bounds.

    Args:
        x1, y1, x2, y2:
            Raw box coordinates.
        image_w:
            Image width.
        image_h:
            Image height.

    Returns:
        Clipped and ordered box coordinates.
    """
    x1 = float(np.clip(x1, 0, max(image_w - 1, 0)))
    y1 = float(np.clip(y1, 0, max(image_h - 1, 0)))
    x2 = float(np.clip(x2, 0, max(image_w - 1, 0)))
    y2 = float(np.clip(y2, 0, max(image_h - 1, 0)))

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    return x1, y1, x2, y2

def _build_pole_overlay_label(prompt, score, final_score):
    """
    Build the exact pole overlay label text to save for later reuse.

    Args:
        prompt:
            Selected pole prompt string.
        score:
            Raw SAM3 score.
        final_score:
            Final ranking score after post-processing.

    Returns:
        Overlay label string.
    """
    label_bits = ["POLE"]

    if prompt is not None and str(prompt).strip():
        label_bits.append(str(prompt).strip())

    if pd.notna(score):
        label_bits.append(f"score={float(score):.3f}")

    if pd.notna(final_score):
        label_bits.append(f"final={float(final_score):.3f}")

    return " | ".join(label_bits)

def _build_no_reliable_overlay_label():
    """
    Build the overlay label used when no reliable pole was found.

    Returns:
        Overlay label string.
    """
    return "NO RELIABLE POLE FOUND"

def _render_pole_overlay_rgb(image_rgb, selected_row=None, mask_2d=None, label_text=None):
    """
    Render the final pole overlay image.

    Args:
        image_rgb:
            RGB image as numpy array.
        selected_row:
            Selected pole row, or None if no pole was selected.
        mask_2d:
            Selected pole mask, or None.
        label_text:
            Overlay label text.

    Returns:
        RGB overlay image as numpy array.
    """
    out = image_rgb.copy().astype(np.float32)

    # -------------------------------------------------------------------------
    # Red mask overlay
    # -------------------------------------------------------------------------
    if isinstance(mask_2d, np.ndarray) and mask_2d.ndim == 2 and mask_2d.shape == image_rgb.shape[:2]:
        red_rgb = np.array([255.0, 0.0, 0.0], dtype=np.float32)
        out[mask_2d] = ((1.0 - POLE_MASK_ALPHA) * out[mask_2d]) + (POLE_MASK_ALPHA * red_rgb)

    out = np.clip(out, 0, 255).astype(np.uint8)

    line_color = (255, 0, 0)
    label_bg = (255, 0, 0)
    text_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    text_thickness = 1
    box_thickness = 3

    h, w = out.shape[:2]

    # -------------------------------------------------------------------------
    # Red box
    # -------------------------------------------------------------------------
    if selected_row is not None:
        x1 = int(round(float(selected_row["x1"])))
        y1 = int(round(float(selected_row["y1"])))
        x2 = int(round(float(selected_row["x2"])))
        y2 = int(round(float(selected_row["y2"])))

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(out, (x1, y1), (x2, y2), line_color, box_thickness)

    # -------------------------------------------------------------------------
    # Label text
    # -------------------------------------------------------------------------
    if label_text is not None and str(label_text).strip():
        label_text = str(label_text).strip()
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text,
            font,
            font_scale,
            text_thickness,
        )

        if selected_row is not None:
            x1 = int(round(float(selected_row["x1"])))
            y1 = int(round(float(selected_row["y1"])))

            box_x1 = max(0, min(w - 1, x1))
            box_y1 = max(0, y1 - text_h - baseline - 10)

            # If there is not enough room above the box, place label just below.
            if box_y1 <= 4:
                box_y1 = min(h - (text_h + baseline + 8), y1 + 6)

        else:
            box_x1 = 8
            box_y1 = 8

        box_x2 = min(w - 1, box_x1 + text_w + 8)
        box_y2 = min(h - 1, box_y1 + text_h + baseline + 8)

        cv2.rectangle(out, (box_x1, box_y1), (box_x2, box_y2), label_bg, -1)

        text_org = (
            box_x1 + 4,
            min(h - 1, box_y2 - baseline - 3),
        )

        cv2.putText(
            out,
            label_text,
            text_org,
            font,
            font_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )

    return out

def _build_overlay_output_path(row, image_id):
    """
    Build the output path for the saved pole overlay image.

    Args:
        row:
            Current run_images_df row.
        image_id:
            Stable image identifier.

    Returns:
        Absolute output path for the overlay image.
    """
    relative_image_path = row.get("relative_image_path", None)

    if isinstance(relative_image_path, str) and len(relative_image_path.strip()) > 0:
        relative_dir = os.path.dirname(relative_image_path)
    else:
        relative_dir = ""

    if relative_dir in ("", "."):
        target_dir = POLE_SELECTION_OVERLAY_DIR
    else:
        target_dir = os.path.join(POLE_SELECTION_OVERLAY_DIR, relative_dir)

    os.makedirs(target_dir, exist_ok=True)

    overlay_file_name = f"{image_id}_pole_overlay.png"
    return os.path.join(target_dir, overlay_file_name)

def _save_overlay_image(image_rgb, output_path):
    """
    Save an RGB overlay image to disk.

    Args:
        image_rgb:
            RGB numpy array.
        output_path:
            Target file path.

    Returns:
        The saved output path.
    """
    Image.fromarray(image_rgb).save(output_path)
    return output_path

def _make_no_reliable_pole_row(
    image_id,
    file_name,
    image_path,
    image_w,
    image_h,
    n_raw_candidates,
    n_kept_candidates,
    fallback_triggered,
    overlay_label_text,
    overlay_image_path,
):
    """
    Build a consistent no-reliable-pole selection row.

    Args:
        image_id:
            Stable image identifier.
        file_name:
            Image filename.
        image_path:
            Image path.
        image_w:
            Image width.
        image_h:
            Image height.
        n_raw_candidates:
            Number of raw candidates found for the image.
        n_kept_candidates:
            Number of candidates kept after prefilter.
        fallback_triggered:
            Whether fallback-to-all-candidates logic was triggered.
        overlay_label_text:
            Saved overlay label text.
        overlay_image_path:
            Saved overlay image path.

    Returns:
        Dict with the full production selection schema.
    """
    return {
        "image_id": image_id,
        "file_name": file_name,
        "image_path": image_path,
        "image_w": int(image_w),
        "image_h": int(image_h),
        "selection_status": "no_reliable_pole_found",
        "selection_mode": "no_reliable_pole_found",
        "n_raw_candidates": int(n_raw_candidates),
        "n_kept_candidates": int(n_kept_candidates),
        "prompt": None,
        "det_idx": None,
        "score": np.nan,
        "x1": np.nan,
        "y1": np.nan,
        "x2": np.nan,
        "y2": np.nan,
        "box_w": np.nan,
        "box_h": np.nan,
        "box_area": np.nan,
        "pole_cx": np.nan,
        "pole_cy": np.nan,
        "x_center_dist_norm": np.nan,
        "width_frac": np.nan,
        "height_frac": np.nan,
        "area_frac": np.nan,
        "aspect_ratio": np.nan,
        "shaft_penalty": np.nan,
        "final_score": np.nan,
        "has_mask": False,
        "fallback_triggered": bool(fallback_triggered),
        "overlay_label_text": overlay_label_text,
        "overlay_image_path": overlay_image_path,
    }

# -----------------------------------------------------------------------------
# 4. Production run across all images
# -----------------------------------------------------------------------------
candidate_frames = []
selection_rows = []
pole_mask_lookup = {}

for row_idx in range(len(run_images_df)):
    row = run_images_df.iloc[row_idx]

    image_path = row["image_path"]

    image_id = row.get("image_id", None)
    if pd.isna(image_id):
        image_id = None

    file_name = row.get("file_name", None)
    if pd.isna(file_name) or not isinstance(file_name, str) or len(file_name.strip()) == 0:
        file_name = os.path.basename(image_path)

    if not isinstance(image_path, str) or len(image_path.strip()) == 0:
        raise ValueError(f"Invalid image_path: {image_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if image_id is None:
        raise ValueError(
            "image_id is None for one of the production rows.\n"
            "Please check CELL 12 ran correctly and image_id was created."
        )

    # -------------------------------------------------------------------------
    # Load image
    # -------------------------------------------------------------------------
    with Image.open(image_path) as img:
        original_mode = img.mode
        if original_mode != "RGB":
            image = img.convert("RGB")
        else:
            image = img.copy()
        image.load()

    image_rgb = np.array(image)
    image_w, image_h = image.size
    image_cx = image_w / 2.0

    # -------------------------------------------------------------------------
    # Prepare processor state
    # -------------------------------------------------------------------------
    if hasattr(processor, "device"):
        processor.device = RUN_DEVICE

    state = {}
    state = processor.set_image(image, state=state)

    # -------------------------------------------------------------------------
    # Run all configured pole prompts and collect raw candidates
    # -------------------------------------------------------------------------
    raw_rows = []

    for prompt in pole_prompt_texts:
        reset_result = processor.reset_all_prompts(state)
        if reset_result is not None:
            state = reset_result

        state = processor.set_text_prompt(prompt, state)

        raw_boxes = state.get("boxes", None)
        raw_scores = state.get("scores", None)
        raw_masks = state.get("masks", None)

        num_detections = _infer_num_detections(raw_boxes, raw_scores, raw_masks)
        boxes = _normalize_boxes_local(raw_boxes, num_detections)
        scores = _normalize_scores_local(raw_scores, num_detections)
        masks_2d = _normalize_masks_local(raw_masks, num_detections, image_h, image_w)

        for det_idx in range(num_detections):
            x1, y1, x2, y2 = [float(v) for v in boxes[det_idx]]
            x1, y1, x2, y2 = _clip_box_to_image(x1, y1, x2, y2, image_w, image_h)

            box_w = max(1.0, x2 - x1)
            box_h = max(1.0, y2 - y1)
            box_area = box_w * box_h

            mask_2d = masks_2d[det_idx] if det_idx < len(masks_2d) else None
            has_mask = isinstance(mask_2d, np.ndarray) and mask_2d.ndim == 2 and mask_2d.sum() > 0

            key = _candidate_key(image_id, prompt, det_idx)
            if has_mask:
                pole_mask_lookup[key] = mask_2d

            raw_rows.append({
                "image_id": image_id,
                "file_name": file_name,
                "image_path": image_path,
                "image_w": int(image_w),
                "image_h": int(image_h),
                "prompt": prompt,
                "det_idx": int(det_idx),
                "score": float(scores[det_idx]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "box_w": box_w,
                "box_h": box_h,
                "box_area": box_area,
                "has_mask": bool(has_mask),
            })

    raw_df = pd.DataFrame(raw_rows)
    n_raw_candidates = int(len(raw_df))

    # -------------------------------------------------------------------------
    # No-detection case for this image
    # -------------------------------------------------------------------------
    if raw_df.empty:
        overlay_label_text = _build_no_reliable_overlay_label()
        overlay_rgb = _render_pole_overlay_rgb(
            image_rgb=image_rgb,
            selected_row=None,
            mask_2d=None,
            label_text=overlay_label_text,
        )
        overlay_image_path = _build_overlay_output_path(row=row, image_id=image_id)
        _save_overlay_image(overlay_rgb, overlay_image_path)

        selection_rows.append(
            _make_no_reliable_pole_row(
                image_id=image_id,
                file_name=file_name,
                image_path=image_path,
                image_w=image_w,
                image_h=image_h,
                n_raw_candidates=0,
                n_kept_candidates=0,
                fallback_triggered=False,
                overlay_label_text=overlay_label_text,
                overlay_image_path=overlay_image_path,
            )
        )
        continue

    # -------------------------------------------------------------------------
    # Add per-image production features
    # -------------------------------------------------------------------------
    scored_df = raw_df.copy()

    scored_df["pole_cx"] = (scored_df["x1"] + scored_df["x2"]) / 2.0
    scored_df["pole_cy"] = (scored_df["y1"] + scored_df["y2"]) / 2.0
    scored_df["image_area"] = scored_df["image_w"] * scored_df["image_h"]

    scored_df["area_frac"] = (
        scored_df["box_area"] / scored_df["image_area"].clip(lower=1.0)
    )
    scored_df["height_frac"] = (
        scored_df["box_h"] / scored_df["image_h"].clip(lower=1.0)
    )
    scored_df["width_frac"] = (
        scored_df["box_w"] / scored_df["image_w"].clip(lower=1.0)
    )
    scored_df["aspect_ratio"] = (
        scored_df["box_h"] / scored_df["box_w"].clip(lower=1.0)
    )

    scored_df["x_center_dist_norm"] = (
        np.abs(scored_df["pole_cx"] - image_cx) / max(image_w / 2.0, 1.0)
    )
    scored_df["x_center_score"] = 1.0 - np.clip(
        scored_df["x_center_dist_norm"], 0.0, 1.0
    )

    # IMPORTANT:
    # Per-image normalization only, matching the earlier pole-selection logic.
    max_h = max(float(scored_df["box_h"].max()), 1.0)
    max_a = max(float(scored_df["box_area"].max()), 1.0)

    scored_df["height_score"] = scored_df["box_h"] / max_h
    scored_df["area_score"] = scored_df["box_area"] / max_a
    scored_df["conf_score"] = scored_df["score"]

    edge_margin = np.minimum.reduce([
        scored_df["x1"].values,
        scored_df["y1"].values,
        (scored_df["image_w"] - scored_df["x2"]).values,
        (scored_df["image_h"] - scored_df["y2"]).values,
    ])

    edge_norm_denom = 0.05 * np.minimum(
        scored_df["image_w"],
        scored_df["image_h"]
    )
    edge_norm_denom = edge_norm_denom.clip(lower=1.0)

    scored_df["edge_margin"] = edge_margin
    scored_df["edge_score"] = np.clip(
        scored_df["edge_margin"] / edge_norm_denom,
        0.0,
        1.0
    )

    # -------------------------------------------------------------------------
    # Prefilter candidates
    # -------------------------------------------------------------------------
    scored_df["keep_score"] = scored_df["score"] >= POLE_MIN_SCORE
    scored_df["keep_area"] = scored_df["area_frac"] >= POLE_MIN_AREA_FRAC
    scored_df["keep_height"] = scored_df["height_frac"] >= POLE_MIN_HEIGHT_FRAC
    scored_df["keep_aspect"] = scored_df["aspect_ratio"] >= POLE_MIN_ASPECT
    scored_df["keep_width_frac"] = scored_df["width_frac"] <= POLE_MAX_WIDTH_FRAC
    scored_df["keep_width_px"] = scored_df["box_w"] <= POLE_MAX_BOX_W_PX

    scored_df["is_kept_after_prefilter"] = (
        scored_df["keep_score"] &
        scored_df["keep_area"] &
        scored_df["keep_height"] &
        scored_df["keep_aspect"] &
        scored_df["keep_width_frac"] &
        scored_df["keep_width_px"]
    )

    n_kept_candidates = int(scored_df["is_kept_after_prefilter"].sum())

    scored_df["selection_mode"] = "not_kept"
    scored_df["final_score"] = np.nan
    scored_df["is_selected_pole"] = False

    kept_df = scored_df[
        scored_df["is_kept_after_prefilter"] == True
    ].copy()

    fallback_triggered = kept_df.empty

    if fallback_triggered:
        kept_df = scored_df.copy()
        kept_df["selection_mode"] = "fallback_all_candidates"
    else:
        kept_df["selection_mode"] = "prefilter_kept"

    kept_df["shaft_penalty"] = np.where(
        kept_df["width_frac"] > SHAFT_WIDTH_FRAC_THRESHOLD,
        SHAFT_PENALTY_FACTOR,
        1.0
    )

    kept_df["final_score"] = (
        (
            W_X_CENTER * kept_df["x_center_score"] +
            W_HEIGHT   * kept_df["height_score"] +
            W_AREA     * kept_df["area_score"] +
            W_CONF     * kept_df["conf_score"] +
            W_EDGE     * kept_df["edge_score"]
        )
        * kept_df["shaft_penalty"]
    )

    kept_df = kept_df.sort_values(
        by=["final_score", "score", "box_h", "x_center_score"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    if len(kept_df) > 0:
        kept_df.loc[0, "is_selected_pole"] = True

    score_cols = [
        "image_id",
        "prompt",
        "det_idx",
        "selection_mode",
        "final_score",
        "is_selected_pole",
        "shaft_penalty",
    ]

    scored_df = scored_df.drop(
        columns=["selection_mode", "final_score", "is_selected_pole", "shaft_penalty"],
        errors="ignore"
    )

    scored_df = scored_df.merge(
        kept_df[score_cols],
        on=["image_id", "prompt", "det_idx"],
        how="left",
    )

    scored_df["selection_mode"] = scored_df["selection_mode"].fillna("not_kept")
    scored_df["is_selected_pole"] = scored_df["is_selected_pole"].fillna(False)
    scored_df["shaft_penalty"] = scored_df["shaft_penalty"].fillna(np.nan)
    scored_df["fallback_triggered"] = bool(fallback_triggered)
    scored_df["selection_status"] = np.where(
        scored_df["is_selected_pole"].astype(bool),
        "selected",
        "not_selected",
    )

    candidate_frames.append(scored_df)

    selected_df = scored_df[
        scored_df["is_selected_pole"] == True
    ].copy()

    if len(selected_df) > 1:
        raise RuntimeError(f"Multiple poles selected for image_id={image_id}")

    if len(selected_df) == 1:
        best_row = selected_df.iloc[0]

        overlay_label_text = _build_pole_overlay_label(
            prompt=best_row["prompt"],
            score=best_row["score"],
            final_score=best_row["final_score"],
        )

        selected_mask_key = _candidate_key(
            image_id=image_id,
            prompt=best_row["prompt"],
            det_idx=int(best_row["det_idx"]),
        )
        selected_mask = pole_mask_lookup.get(selected_mask_key, None)

        overlay_rgb = _render_pole_overlay_rgb(
            image_rgb=image_rgb,
            selected_row=best_row,
            mask_2d=selected_mask,
            label_text=overlay_label_text,
        )
        overlay_image_path = _build_overlay_output_path(row=row, image_id=image_id)
        _save_overlay_image(overlay_rgb, overlay_image_path)

        selection_rows.append({
            "image_id": image_id,
            "file_name": file_name,
            "image_path": image_path,
            "image_w": int(image_w),
            "image_h": int(image_h),
            "selection_status": "selected",
            "selection_mode": str(best_row["selection_mode"]),
            "n_raw_candidates": n_raw_candidates,
            "n_kept_candidates": n_kept_candidates,
            "prompt": str(best_row["prompt"]),
            "det_idx": int(best_row["det_idx"]),
            "score": float(best_row["score"]),
            "x1": float(best_row["x1"]),
            "y1": float(best_row["y1"]),
            "x2": float(best_row["x2"]),
            "y2": float(best_row["y2"]),
            "box_w": float(best_row["box_w"]),
            "box_h": float(best_row["box_h"]),
            "box_area": float(best_row["box_area"]),
            "pole_cx": float(best_row["pole_cx"]),
            "pole_cy": float(best_row["pole_cy"]),
            "x_center_dist_norm": float(best_row["x_center_dist_norm"]),
            "width_frac": float(best_row["width_frac"]),
            "height_frac": float(best_row["height_frac"]),
            "area_frac": float(best_row["area_frac"]),
            "aspect_ratio": float(best_row["aspect_ratio"]),
            "shaft_penalty": float(best_row["shaft_penalty"])
                if pd.notna(best_row["shaft_penalty"]) else np.nan,
            "final_score": float(best_row["final_score"])
                if pd.notna(best_row["final_score"]) else np.nan,
            "has_mask": bool(best_row["has_mask"]),
            "fallback_triggered": bool(fallback_triggered),
            "overlay_label_text": overlay_label_text,
            "overlay_image_path": overlay_image_path,
        })
    else:
        overlay_label_text = _build_no_reliable_overlay_label()
        overlay_rgb = _render_pole_overlay_rgb(
            image_rgb=image_rgb,
            selected_row=None,
            mask_2d=None,
            label_text=overlay_label_text,
        )
        overlay_image_path = _build_overlay_output_path(row=row, image_id=image_id)
        _save_overlay_image(overlay_rgb, overlay_image_path)

        selection_rows.append(
            _make_no_reliable_pole_row(
                image_id=image_id,
                file_name=file_name,
                image_path=image_path,
                image_w=image_w,
                image_h=image_h,
                n_raw_candidates=n_raw_candidates,
                n_kept_candidates=n_kept_candidates,
                fallback_triggered=fallback_triggered,
                overlay_label_text=overlay_label_text,
                overlay_image_path=overlay_image_path,
            )
        )

# -----------------------------------------------------------------------------
# 5. Combine outputs
# -----------------------------------------------------------------------------
pole_candidates_df = (
    pd.concat(candidate_frames, ignore_index=True)
    if len(candidate_frames) > 0 else
    pd.DataFrame()
)

pole_selection_df = pd.DataFrame(selection_rows)

# -----------------------------------------------------------------------------
# 6. Reorder key columns for readability
# -----------------------------------------------------------------------------
candidate_front_cols = [
    "image_id",
    "file_name",
    "image_path",
    "prompt",
    "det_idx",
    "score",
    "x1",
    "y1",
    "x2",
    "y2",
    "box_w",
    "box_h",
    "box_area",
    "pole_cx",
    "pole_cy",
    "x_center_dist_norm",
    "area_frac",
    "height_frac",
    "width_frac",
    "aspect_ratio",
    "has_mask",
    "is_kept_after_prefilter",
    "shaft_penalty",
    "final_score",
    "selection_mode",
    "selection_status",
    "is_selected_pole",
]

if not pole_candidates_df.empty:
    candidate_existing_front_cols = [c for c in candidate_front_cols if c in pole_candidates_df.columns]
    candidate_remaining_cols = [c for c in pole_candidates_df.columns if c not in candidate_existing_front_cols]
    pole_candidates_df = pole_candidates_df[candidate_existing_front_cols + candidate_remaining_cols]

selection_front_cols = [
    "image_id",
    "file_name",
    "image_path",
    "selection_status",
    "selection_mode",
    "n_raw_candidates",
    "n_kept_candidates",
    "prompt",
    "det_idx",
    "score",
    "x1",
    "y1",
    "x2",
    "y2",
    "box_w",
    "box_h",
    "box_area",
    "pole_cx",
    "pole_cy",
    "x_center_dist_norm",
    "width_frac",
    "height_frac",
    "area_frac",
    "aspect_ratio",
    "has_mask",
    "shaft_penalty",
    "final_score",
    "fallback_triggered",
    "overlay_label_text",
    "overlay_image_path",
]

if not pole_selection_df.empty:
    selection_existing_front_cols = [c for c in selection_front_cols if c in pole_selection_df.columns]
    selection_remaining_cols = [c for c in pole_selection_df.columns if c not in selection_existing_front_cols]
    pole_selection_df = pole_selection_df[selection_existing_front_cols + selection_remaining_cols]

# -----------------------------------------------------------------------------
# 7. Final checks
# -----------------------------------------------------------------------------
selected_count = int(
    (pole_selection_df["selection_status"] == "selected").sum()
) if "selection_status" in pole_selection_df.columns else 0

no_reliable_count = int(
    (pole_selection_df["selection_status"] == "no_reliable_pole_found").sum()
) if "selection_status" in pole_selection_df.columns else 0

if len(pole_selection_df) != len(run_images_df):
    raise RuntimeError(
        "pole_selection_df does not contain exactly one row per input image.\n"
        f"run_images_df rows     : {len(run_images_df)}\n"
        f"pole_selection_df rows : {len(pole_selection_df)}"
    )

missing_overlay_files = [
    p for p in pole_selection_df["overlay_image_path"].dropna().tolist()
    if not os.path.exists(p)
]

if len(missing_overlay_files) > 0:
    raise RuntimeError(
        "Some pole overlay images were expected but were not found on disk.\n"
        f"Missing files: {missing_overlay_files[:10]}"
    )

# -----------------------------------------------------------------------------
# 8. Persist only production outputs
# -----------------------------------------------------------------------------
if "save_state" in globals():
    save_state(
        df_names=[
            name for name in [
                "pole_candidates_df",
                "pole_selection_df",
            ]
            if isinstance(globals().get(name), pd.DataFrame)
        ],
        config_extra={
            "POLE_PROMPT_TEXT": pole_prompt_texts,
            "POLE_TEXT_THRESHOLD": POLE_TEXT_THRESHOLD,
            "POLE_MIN_SCORE": POLE_MIN_SCORE,
            "POLE_MIN_AREA_FRAC": POLE_MIN_AREA_FRAC,
            "POLE_MIN_HEIGHT_FRAC": POLE_MIN_HEIGHT_FRAC,
            "POLE_MIN_ASPECT": POLE_MIN_ASPECT,
            "POLE_MAX_WIDTH_FRAC": POLE_MAX_WIDTH_FRAC,
            "POLE_MAX_BOX_W_PX": POLE_MAX_BOX_W_PX,
            "SHAFT_WIDTH_FRAC_THRESHOLD": SHAFT_WIDTH_FRAC_THRESHOLD,
            "SHAFT_PENALTY_FACTOR": SHAFT_PENALTY_FACTOR,
            "W_X_CENTER": W_X_CENTER,
            "W_HEIGHT": W_HEIGHT,
            "W_AREA": W_AREA,
            "W_CONF": W_CONF,
            "W_EDGE": W_EDGE,
            "POLE_SELECTION_OVERLAY_DIR": POLE_SELECTION_OVERLAY_DIR,
            "POLE_MASK_ALPHA": POLE_MASK_ALPHA,
        },
        nb_globals=globals(),
    )
else:
    print(
        "Note: save_state not available in this Databricks notebook; "
        "outputs remain in globals only."
    )

# -----------------------------------------------------------------------------
# 9. Final summary
# -----------------------------------------------------------------------------
print("\nCELL 13 production completed.")
print(f"  run_images_df rows          : {len(run_images_df)}")
print(f"  pole_candidates_df rows     : {len(pole_candidates_df)}")
print(f"  pole_selection_df rows      : {len(pole_selection_df)}")
print(f"  selected poles              : {selected_count}")
print(f"  no_reliable_pole_found rows : {no_reliable_count}")
print(f"  pole_mask_lookup entries    : {len(pole_mask_lookup)}")
print(f"  overlay folder              : {POLE_SELECTION_OVERLAY_DIR}")

print("\nPersisted outputs:")
print("  - pole_candidates_df")
print("  - pole_selection_df")
print("  - pole_mask_lookup")
