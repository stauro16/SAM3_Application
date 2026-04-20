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
print("SAM3_REPO_ROOT  :", SAM3_REPO_ROOT)





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
SILVER_POLE_SELECTION = os.path.join(SILVER_ROOT, "pole_selection")
SILVER_POLE_SELECTION_OVERLAYS = os.path.join(SILVER_POLE_SELECTION,"overlays")

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
    SILVER_POLE_SELECTION,
    SILVER_POLE_SELECTION_OVERLAYS,
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
required_cols = [
    "image_path",
    "relative_image_path",
    "file_name",
    "stem",
    "ext",
]
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
# 3. Sort into stable processing order
# -----------------------------------------------------------------------------
# EXPLANATION:
# This makes production runs easier to compare across reruns.
# -----------------------------------------------------------------------------
run_images_df = run_images_df.sort_values(
    ["relative_image_path", "file_name", "image_path"]
).reset_index(drop=True)

# -----------------------------------------------------------------------------
# 4. Create stable unique image_id if missing
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
# 5. Reorder key columns for readability
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
# 6. Final checks
# -----------------------------------------------------------------------------
# EXPLANATION:
# Enforce the expected production output shape before later inference cell runs.
# -----------------------------------------------------------------------------
if run_images_df["image_id"].duplicated().any():
    dup_ids = run_images_df.loc[run_images_df["image_id"].duplicated(), "image_id"].tolist()

    raise RuntimeError(
        f"run_images_df contains duplicate image_id values.\n"
        f"  Duplicates: {dup_ids[:10]}"
    )

# -----------------------------------------------------------------------------
# 7. Print summary
# -----------------------------------------------------------------------------
# EXPLANATION:
# This provides a quick overview before later pipeline inference cells.
# -----------------------------------------------------------------------------
print("Production image manifest preparation complete.\n")
print(f"  run_images_df rows : {len(run_images_df)}")
print(f"  IMAGE_ID_PREFIX    : {IMAGE_ID_PREFIX}")





