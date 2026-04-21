15-16 Current 
==================
# =============================================================================
# CELL 15A — DRAW CROP BOX ON ONE IMAGE (DEBUG)
# =============================================================================
# EXPLANATION:
# This cell is the Databricks single-image debug version of the Colab crop-box
# step that follows pole selection.
#
# WHAT THIS CELL DOES:
#   1) reads pole_selection_df from CELL 14C
#   2) selects exactly ONE row using a debug index
#   3) requires that the chosen row has a selected pole
#   4) builds an expanded crop / ROI box around that selected pole
#   5) draws:
#        - selected pole box = red solid
#        - expanded crop box = cyan dashed
#   6) creates pole_roi_debug_df for the next downstream crop/save step
#
# IMPORTANT:
# - this cell does NOT crop and save images yet
# - this cell only computes and visualizes the crop box geometry
# - this is the single-image debug bridge between CELL 14C and the later
#   all-images ROI extraction step
# - POLE_ROI_DEBUG_ROW_INDEX indexes into pole_selection_df, which has one
#   row per image across ALL of run_images_df — not just the 8-row debug
#   subset. DEBUG_ROW_INDEX (value 4) is NOT a safe default here. The
#   default is 0. Override POLE_ROI_DEBUG_ROW_INDEX before running if you
#   want a different row.
#
# INPUT:
# - pole_selection_df from CELL 14C
#
# OUTPUT:
# - pole_roi_debug_df : exactly one row containing the selected pole + crop box
# =============================================================================

# -----------------------------------------------------------------------------
# 0. SAFETY CHECKS
# -----------------------------------------------------------------------------
required_globals = [
    "pole_selection_df",
]

missing_globals = [name for name in required_globals if name not in globals()]
if missing_globals:
    raise NameError(
        "CELL 15A cannot run because some required variables are missing.\n"
        "Please run CELL 14C first.\n"
        f"Missing globals: {missing_globals}"
    )

if not isinstance(pole_selection_df, pd.DataFrame):
    raise TypeError("pole_selection_df exists but is not a pandas DataFrame.")

if pole_selection_df.empty:
    raise ValueError("pole_selection_df is empty. Please check CELL 14C.")

# -----------------------------------------------------------------------------
# 1. DEBUG ROW + CROP BOX CONFIG
# -----------------------------------------------------------------------------
# EXPLANATION:
# Use the same general expansion strategy as your earlier Colab crop-box step.
# This is kept explicit here so you can tune it easily before moving to 15B.
#
# IMPORTANT:
# POLE_ROI_DEBUG_ROW_INDEX indexes into pole_selection_df, which spans all
# run_images_df rows — not the 8-row debug subset. The default is 0.
# To inspect a different image, set POLE_ROI_DEBUG_ROW_INDEX before running
# this cell, e.g.:
#   POLE_ROI_DEBUG_ROW_INDEX = 3
# -----------------------------------------------------------------------------
POLE_ROI_DEBUG_ROW_INDEX = int(globals().get("POLE_ROI_DEBUG_ROW_INDEX", 0))

EXPANDED_BOX_WIDTH_FACTOR_FROM_POLE_HEIGHT = float(
    globals().get("EXPANDED_BOX_WIDTH_FACTOR_FROM_POLE_HEIGHT", 0.90)
)
MIN_EXPANDED_BOX_WIDTH = float(
    globals().get("MIN_EXPANDED_BOX_WIDTH", 600)
)

TOP_EXTRA_FACTOR_FROM_POLE_HEIGHT = float(
    globals().get("TOP_EXTRA_FACTOR_FROM_POLE_HEIGHT", 0.10)
)
BOTTOM_EXTRA_FACTOR_FROM_POLE_HEIGHT = float(
    globals().get("BOTTOM_EXTRA_FACTOR_FROM_POLE_HEIGHT", 0.20)
)

MIN_TOP_EXTRA_PIXELS = float(
    globals().get("MIN_TOP_EXTRA_PIXELS", 40)
)
MIN_BOTTOM_EXTRA_PIXELS = float(
    globals().get("MIN_BOTTOM_EXTRA_PIXELS", 10)
)

print("Single-image crop-box config:")
print(f"  POLE_ROI_DEBUG_ROW_INDEX                    : {POLE_ROI_DEBUG_ROW_INDEX}")
print(f"  pole_selection_df rows                      : {len(pole_selection_df)}")
print(f"  EXPANDED_BOX_WIDTH_FACTOR_FROM_POLE_HEIGHT : {EXPANDED_BOX_WIDTH_FACTOR_FROM_POLE_HEIGHT}")
print(f"  MIN_EXPANDED_BOX_WIDTH                     : {MIN_EXPANDED_BOX_WIDTH}")
print(f"  TOP_EXTRA_FACTOR_FROM_POLE_HEIGHT          : {TOP_EXTRA_FACTOR_FROM_POLE_HEIGHT}")
print(f"  BOTTOM_EXTRA_FACTOR_FROM_POLE_HEIGHT       : {BOTTOM_EXTRA_FACTOR_FROM_POLE_HEIGHT}")
print(f"  MIN_TOP_EXTRA_PIXELS                       : {MIN_TOP_EXTRA_PIXELS}")
print(f"  MIN_BOTTOM_EXTRA_PIXELS                    : {MIN_BOTTOM_EXTRA_PIXELS}")

# -----------------------------------------------------------------------------
# 2. HELPER: SAFE DISPLAY
# -----------------------------------------------------------------------------
def _safe_display(obj):
    """
    Display a pandas object in a notebook if possible; otherwise print it.

    Args:
        obj:
            Object to display.

    Returns:
        None
    """
    try:
        display(obj)
    except Exception:
        print("WARNING: display() unavailable, falling back to print().")
        print(obj)

# -----------------------------------------------------------------------------
# 3. HELPER: CANDIDATE KEY
# -----------------------------------------------------------------------------
def _candidate_key(image_id, prompt, det_idx):
    """
    Build the same mask-lookup key used in CELL 14C.

    Args:
        image_id:
            Stable image identifier.

        prompt:
            Pole prompt used for the selected detection.

        det_idx:
            Detection index within that prompt run.

    Returns:
        tuple:
            (image_id, prompt, det_idx) as a stable lookup key.
    """
    return (str(image_id), str(prompt), int(det_idx))

# -----------------------------------------------------------------------------
# 4. HELPER: BUILD EXPANDED BOX FROM SELECTED POLE
# -----------------------------------------------------------------------------
def _build_expanded_box_from_pole(
    pole_x1,
    pole_y1,
    pole_x2,
    pole_y2,
    image_w,
    image_h,
    width_factor_from_pole_height,
    min_box_width,
    top_extra_factor_from_pole_height,
    bottom_extra_factor_from_pole_height,
    min_top_extra_pixels,
    min_bottom_extra_pixels,
):
    """
    Build a larger ROI box around the selected pole.

    Args:
        pole_x1, pole_y1, pole_x2, pole_y2:
            Selected pole box coordinates.

        image_w, image_h:
            Full image dimensions.

        width_factor_from_pole_height:
            ROI width multiplier based on pole height.

        min_box_width:
            Minimum allowed ROI width in pixels.

        top_extra_factor_from_pole_height:
            Extra height above the pole top.

        bottom_extra_factor_from_pole_height:
            Extra height below the pole bottom.

        min_top_extra_pixels, min_bottom_extra_pixels:
            Minimum top / bottom expansion in pixels.

    Returns:
        dict:
            Expanded ROI geometry.
    """
    # Compute basic pole geometry.
    pole_w = max(0.0, pole_x2 - pole_x1)
    pole_h = max(0.0, pole_y2 - pole_y1)
    pole_cx = pole_x1 + pole_w / 2.0
    pole_cy = pole_y1 + pole_h / 2.0

    # Set crop width mainly from pole height, with a hard minimum.
    expanded_w = max(
        float(min_box_width),
        float(round(pole_h * width_factor_from_pole_height)),
    )

    # Expand vertically using separate top / bottom rules.
    top_extra = max(
        float(min_top_extra_pixels),
        float(round(pole_h * top_extra_factor_from_pole_height)),
    )

    bottom_extra = max(
        float(min_bottom_extra_pixels),
        float(round(pole_h * bottom_extra_factor_from_pole_height)),
    )

    half_w = expanded_w / 2.0

    ex1 = pole_cx - half_w
    ex2 = pole_cx + half_w
    ey1 = pole_y1 - top_extra
    ey2 = pole_y2 + bottom_extra

    # -------------------------------------------------------------------------
    # Clamp to image bounds
    # -------------------------------------------------------------------------
    # If the box spills left, shift it right first.
    if ex1 < 0:
        ex2 -= ex1
        ex1 = 0.0

    # If the box spills right, shift it left.
    if ex2 > image_w:
        shift = ex2 - image_w
        ex1 -= shift
        ex2 = float(image_w)

    # Clamp top / bottom directly.
    if ey1 < 0:
        ey1 = 0.0

    if ey2 > image_h:
        ey2 = float(image_h)

    # Final hard clamp.
    ex1 = max(0.0, ex1)
    ex2 = min(float(image_w), ex2)
    ey1 = max(0.0, ey1)
    ey2 = min(float(image_h), ey2)

    final_w = max(0.0, ex2 - ex1)
    final_h = max(0.0, ey2 - ey1)

    return {
        "expanded_x1": float(ex1),
        "expanded_y1": float(ey1),
        "expanded_x2": float(ex2),
        "expanded_y2": float(ey2),
        "expanded_w": float(final_w),
        "expanded_h": float(final_h),
        "pole_w": float(pole_w),
        "pole_h": float(pole_h),
        "pole_cx": float(pole_cx),
        "pole_cy": float(pole_cy),
    }

# -----------------------------------------------------------------------------
# 5. HELPER: DRAW SELECTED POLE + EXPANDED CROP BOX
# -----------------------------------------------------------------------------
def draw_selected_pole_and_crop_box(ax, image_rgb, row, title):
    """
    Draw the selected pole box and the expanded crop box.

    VISUAL STYLE:
    - selected pole box = red solid
    - expanded crop box = cyan dashed

    Args:
        ax:
            Matplotlib axis.

        image_rgb:
            RGB numpy image.

        row:
            Row-like object containing selected pole + crop box coordinates.

        title:
            Plot title.

    Returns:
        None
    """
    ax.imshow(image_rgb)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    # -------------------------------------------------------------------------
    # Red selected pole box
    # -------------------------------------------------------------------------
    pole_x1 = float(row["x1"])
    pole_y1 = float(row["y1"])
    pole_x2 = float(row["x2"])
    pole_y2 = float(row["y2"])

    pole_w = pole_x2 - pole_x1
    pole_h = pole_y2 - pole_y1

    selected_rect = patches.Rectangle(
        (pole_x1, pole_y1),
        pole_w,
        pole_h,
        linewidth=3.0,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(selected_rect)

    # -------------------------------------------------------------------------
    # Cyan dashed expanded crop box
    # -------------------------------------------------------------------------
    ex1 = float(row["expanded_x1"])
    ey1 = float(row["expanded_y1"])
    ew = float(row["expanded_w"])
    eh = float(row["expanded_h"])

    expanded_rect = patches.Rectangle(
        (ex1, ey1),
        ew,
        eh,
        linewidth=2.4,
        edgecolor="cyan",
        linestyle="--",
        facecolor="none",
        alpha=0.95,
    )
    ax.add_patch(expanded_rect)

    # -------------------------------------------------------------------------
    # Red label for selected pole
    # -------------------------------------------------------------------------
    prompt_text = None
    if "prompt" in row and pd.notna(row["prompt"]) and str(row["prompt"]).strip():
        prompt_text = str(row["prompt"]).strip()
    elif "POLE_PROMPT" in globals():
        prompt_text = str(POLE_PROMPT).strip()
    else:
        prompt_text = "timber power pole"

    pole_label = f"POLE | {prompt_text}"
    if "score" in row and pd.notna(row["score"]):
        pole_label += f" | score={float(row['score']):.3f}"
    if "final_score" in row and pd.notna(row["final_score"]):
        pole_label += f" | final={float(row['final_score']):.3f}"

    ax.text(
        pole_x1,
        max(5, pole_y1 - 6),
        pole_label,
        color="white",
        fontsize=8.5,
        bbox=dict(facecolor="red", alpha=0.95, pad=1.5, edgecolor="none"),
    )

    # -------------------------------------------------------------------------
    # Cyan label for expanded crop box
    # -------------------------------------------------------------------------
    crop_label = f"CROP BOX | {int(round(ew))}x{int(round(eh))}"

    ax.text(
        ex1,
        min(ey1 + 18, float(row["image_h"]) - 10),
        crop_label,
        color="black",
        fontsize=8.0,
        bbox=dict(facecolor="cyan", alpha=0.90, pad=1.5, edgecolor="none"),
    )

# -----------------------------------------------------------------------------
# 6. SELECT EXACTLY ONE ROW FROM pole_selection_df
# -----------------------------------------------------------------------------
if POLE_ROI_DEBUG_ROW_INDEX >= len(pole_selection_df):
    raise IndexError(
        f"POLE_ROI_DEBUG_ROW_INDEX={POLE_ROI_DEBUG_ROW_INDEX} is out of range for "
        f"pole_selection_df with {len(pole_selection_df)} rows.\n"
        "pole_selection_df spans all run_images_df rows, not just the debug subset."
    )

selected_row = pole_selection_df.reset_index(drop=True).iloc[POLE_ROI_DEBUG_ROW_INDEX]

image_id = selected_row.get("image_id", None)
file_name = selected_row.get("file_name", None)
image_path = selected_row.get("image_path", None)
selection_status = selected_row.get("selection_status", "unknown")

if pd.isna(file_name) or not isinstance(file_name, str) or len(file_name.strip()) == 0:
    if isinstance(image_path, str):
        file_name = os.path.basename(image_path)

if selection_status != "selected":
    raise ValueError(
        "The chosen row does not contain a selected pole.\n"
        f"POLE_ROI_DEBUG_ROW_INDEX={POLE_ROI_DEBUG_ROW_INDEX}\n"
        f"selection_status={selection_status}\n"
        "Choose a row from pole_selection_df where selection_status == 'selected'.\n"
        "Note: pole_selection_df spans all run_images_df rows — check which rows\n"
        "have selection_status == 'selected' before setting POLE_ROI_DEBUG_ROW_INDEX."
    )

if not isinstance(image_path, str) or len(image_path.strip()) == 0:
    raise ValueError(f"Invalid image_path for selected row: {image_path}")

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

# -----------------------------------------------------------------------------
# 7. LOAD THE IMAGE AND BUILD THE DEBUG ROI ROW
# -----------------------------------------------------------------------------
with Image.open(image_path) as img:
    image = img.convert("RGB")
    image.load()

image_w, image_h = image.size
image_rgb = np.array(image)

pole_x1 = float(selected_row["x1"])
pole_y1 = float(selected_row["y1"])
pole_x2 = float(selected_row["x2"])
pole_y2 = float(selected_row["y2"])

expanded_box = _build_expanded_box_from_pole(
    pole_x1=pole_x1,
    pole_y1=pole_y1,
    pole_x2=pole_x2,
    pole_y2=pole_y2,
    image_w=image_w,
    image_h=image_h,
    width_factor_from_pole_height=EXPANDED_BOX_WIDTH_FACTOR_FROM_POLE_HEIGHT,
    min_box_width=MIN_EXPANDED_BOX_WIDTH,
    top_extra_factor_from_pole_height=TOP_EXTRA_FACTOR_FROM_POLE_HEIGHT,
    bottom_extra_factor_from_pole_height=BOTTOM_EXTRA_FACTOR_FROM_POLE_HEIGHT,
    min_top_extra_pixels=MIN_TOP_EXTRA_PIXELS,
    min_bottom_extra_pixels=MIN_BOTTOM_EXTRA_PIXELS,
)

# -------------------------------------------------------------------------
# Carry forward selected-pole identity fields from CELL 14C so later ROI /
# mask-overlap steps can still map this ROI row back to the selected pole.
# -------------------------------------------------------------------------
prompt = selected_row.get(
    "prompt",
    globals().get("POLE_PROMPT", "timber power pole")
)
det_idx = selected_row.get("det_idx", np.nan)
score = selected_row.get("score", np.nan)
has_mask = bool(selected_row.get("has_mask", False))
selection_mode = selected_row.get("selection_mode", None)
final_score = selected_row.get("final_score", np.nan)

mask_lookup_key = None
mask_lookup_hit = False

if (
    "pole_mask_lookup" in globals()
    and isinstance(globals().get("pole_mask_lookup"), dict)
    and pd.notna(det_idx)
):
    mask_lookup_key = _candidate_key(image_id, prompt, int(det_idx))
    mask_lookup_hit = mask_lookup_key in pole_mask_lookup

pole_roi_debug_df = pd.DataFrame([
    {
        "image_id": image_id,
        "file_name": file_name,
        "image_path": image_path,
        "debug_row_index": int(POLE_ROI_DEBUG_ROW_INDEX),

        # ---------------------------------------------------------------------
        # Selected-pole carry-forward fields from CELL 14C
        # ---------------------------------------------------------------------
        "selection_status": selection_status,
        "selection_mode": selection_mode,
        "roi_status": "ok",
        "prompt": prompt,
        "det_idx": int(det_idx) if pd.notna(det_idx) else np.nan,
        "score": score,
        "final_score": final_score,
        "has_mask": has_mask,
        "mask_lookup_hit": bool(mask_lookup_hit),

        # ---------------------------------------------------------------------
        # Selected pole geometry
        # ---------------------------------------------------------------------
        "x1": pole_x1,
        "y1": pole_y1,
        "x2": pole_x2,
        "y2": pole_y2,
        "pole_w": expanded_box["pole_w"],
        "pole_h": expanded_box["pole_h"],
        "pole_cx": expanded_box["pole_cx"],
        "pole_cy": expanded_box["pole_cy"],

        # ---------------------------------------------------------------------
        # Full image geometry
        # ---------------------------------------------------------------------
        "image_w": int(image_w),
        "image_h": int(image_h),

        # ---------------------------------------------------------------------
        # Expanded ROI geometry
        # ---------------------------------------------------------------------
        "expanded_x1": expanded_box["expanded_x1"],
        "expanded_y1": expanded_box["expanded_y1"],
        "expanded_x2": expanded_box["expanded_x2"],
        "expanded_y2": expanded_box["expanded_y2"],
        "expanded_w": expanded_box["expanded_w"],
        "expanded_h": expanded_box["expanded_h"],
    }
])

# -----------------------------------------------------------------------------
# 8. OUTPUT TABLE
# -----------------------------------------------------------------------------
print("\n" + "=" * 100)
print("ONE-IMAGE POLE ROI DEBUG TABLE")
print("=" * 100)

_safe_display(
    pole_roi_debug_df[
        [
            "image_id",
            "file_name",
            "debug_row_index",
            "selection_status",
            "selection_mode",
            "roi_status",
            "prompt",
            "det_idx",
            "score",
            "final_score",
            "has_mask",
            "mask_lookup_hit",
            "x1",
            "y1",
            "x2",
            "y2",
            "pole_cx",
            "pole_cy",
            "expanded_x1",
            "expanded_y1",
            "expanded_x2",
            "expanded_y2",
            "expanded_w",
            "expanded_h",
        ]
    ].reset_index(drop=True)
)

# -----------------------------------------------------------------------------
# 9. DRAW THE SINGLE IMAGE — SELECTED POLE + CROP BOX
# -----------------------------------------------------------------------------
print("\n" + "=" * 100)
print("ONE-IMAGE CROP-BOX DEBUG VIEW")
print("=" * 100)

fig, ax = plt.subplots(1, 1, figsize=(11, 8))

draw_selected_pole_and_crop_box(
    ax=ax,
    image_rgb=image_rgb,
    row=pole_roi_debug_df.iloc[0],
    title=f"{file_name}\nCELL 15A — DRAW CROP BOX ON ONE IMAGE",
)

plt.tight_layout()
plt.show()
plt.close()

# -----------------------------------------------------------------------------
# 10. SAVE STATE
# -----------------------------------------------------------------------------
if "save_state" in globals():
    save_state(
        df_names=[
            name for name in [
                "pole_selection_df",
                "pole_roi_debug_df",
            ]
            if isinstance(globals().get(name), pd.DataFrame)
        ],
        config_extra={
            "POLE_ROI_DEBUG_ROW_INDEX": POLE_ROI_DEBUG_ROW_INDEX,
            "EXPANDED_BOX_WIDTH_FACTOR_FROM_POLE_HEIGHT": EXPANDED_BOX_WIDTH_FACTOR_FROM_POLE_HEIGHT,
            "MIN_EXPANDED_BOX_WIDTH": MIN_EXPANDED_BOX_WIDTH,
            "TOP_EXTRA_FACTOR_FROM_POLE_HEIGHT": TOP_EXTRA_FACTOR_FROM_POLE_HEIGHT,
            "BOTTOM_EXTRA_FACTOR_FROM_POLE_HEIGHT": BOTTOM_EXTRA_FACTOR_FROM_POLE_HEIGHT,
            "MIN_TOP_EXTRA_PIXELS": MIN_TOP_EXTRA_PIXELS,
            "MIN_BOTTOM_EXTRA_PIXELS": MIN_BOTTOM_EXTRA_PIXELS,
        },
        nb_globals=globals(),
    )
else:
    print(
        "Note: save_state not available in this Databricks notebook; "
        "outputs remain in globals only."
    )

print("\nCELL 15A completed.")
print("Saved outputs:")
print("  - pole_roi_debug_df")




# =============================================================================
# CELL 15B — PRODUCTION POLE-TOP FIXED CANVAS ROI + SHIFT + PAD + SAVE TO SILVER
# =============================================================================
# EXPLANATION:
# Build one fixed-size pole-top ROI for every selected pole and save to Silver.
#
# STRATEGY:
# - anchor from the selected pole top
# - keep the final ROI size fixed for every image
# - shift the requested box inside the image first when possible
# - pad only if needed
#
# RESULT:
# - every saved ROI image has exactly the same size
# - the pole-top hardware zone is framed more consistently
# - the gallery shows debug-style labels for both the pole box and crop box
# - draws:
#        - selected pole box = red solid
#        - requested ROI box = cyan dashed
#
# DATABRICKS DIFFERENCES FROM COLAB CELL 8:
# - SILVER_POLE_ROIS comes from CELL 10 globals (not a hardcoded path)
# - save_state is optional — else-branch prints a note
# - matplotlib.patches is already imported at module level in CELL 3A
# - all config constants use globals().get() with matching defaults
#
# INPUT:
# - pole_selection_df from CELL 14C
#
# OUTPUT:
# - pole_rois_df : one row per selected pole with full crop geometry + file paths
# =============================================================================

# -----------------------------------------------------------------------------
# 0. SAFETY CHECKS
# -----------------------------------------------------------------------------
required_globals = [
    "pole_selection_df",
    "SILVER_POLE_ROIS",
]

missing_globals = [name for name in required_globals if name not in globals()]
if missing_globals:
    raise NameError(
        "CELL 15B cannot run because some required variables are missing.\n"
        "Please run CELL 14C and CELL 10 first.\n"
        f"Missing globals: {missing_globals}"
    )

if not isinstance(pole_selection_df, pd.DataFrame):
    raise TypeError("pole_selection_df exists but is not a pandas DataFrame.")

if pole_selection_df.empty:
    raise ValueError("pole_selection_df is empty. Please check CELL 14C.")

# -----------------------------------------------------------------------------
# 1. KEEP ONLY SELECTED POLES
# -----------------------------------------------------------------------------
# Keep only rows where a reliable pole was selected.
if "selection_status" in pole_selection_df.columns:
    selected_poles_df = pole_selection_df[
        pole_selection_df["selection_status"] == "selected"
    ].copy()
else:
    selected_poles_df = pole_selection_df[
        pole_selection_df[["x1", "y1", "x2", "y2"]].notna().all(axis=1)
    ].copy()

if selected_poles_df.empty:
    raise ValueError(
        "No selected poles were found in pole_selection_df.\n"
        "Please check CELL 14C production output."
    )

# -----------------------------------------------------------------------------
# 2. FIXED POLE-TOP ROI CONFIG
# -----------------------------------------------------------------------------
# EXPLANATION:
# Use one fixed final canvas size for every image.
# All constants use globals().get() so they can be overridden before running.
# -----------------------------------------------------------------------------
FIXED_ROI_WIDTH = int(globals().get("FIXED_ROI_WIDTH", 2600))
FIXED_ROI_HEIGHT = int(globals().get("FIXED_ROI_HEIGHT", 3500))

# Start the requested ROI this many pixels above the selected pole top.
POLE_TOP_BUFFER_ABOVE = int(globals().get("POLE_TOP_BUFFER_ABOVE", 350))

# Use a black background if padding is needed.
PAD_RGB = globals().get("PAD_RGB", (0, 0, 0))

# Show only a small gallery of selected poles.
POLE_ROI_GALLERY_COUNT = min(
    int(globals().get("POLE_ROI_GALLERY_COUNT", 6)),
    len(selected_poles_df)
)

print("Fixed pole-top ROI config:")
print(f"  FIXED_ROI_WIDTH        : {FIXED_ROI_WIDTH}")
print(f"  FIXED_ROI_HEIGHT       : {FIXED_ROI_HEIGHT}")
print(f"  POLE_TOP_BUFFER_ABOVE  : {POLE_TOP_BUFFER_ABOVE}")
print(f"  PAD_RGB                : {PAD_RGB}")
print(f"  POLE_ROI_GALLERY_COUNT : {POLE_ROI_GALLERY_COUNT}")
print(f"  selected_poles_df rows : {len(selected_poles_df)}")
print(f"  SILVER_POLE_ROIS       : {SILVER_POLE_ROIS}")

# -----------------------------------------------------------------------------
# 3. HELPER: SAFE DISPLAY
# -----------------------------------------------------------------------------
def _safe_display(obj):
    """
    Display a pandas object in a notebook if possible; otherwise print it.

    Args:
        obj:
            Object to display.

    Returns:
        None
    """
    try:
        display(obj)
    except Exception:
        print("WARNING: display() unavailable, falling back to print().")
        print(obj)

# -----------------------------------------------------------------------------
# 4. HELPER: SAFE COLUMN SUBSET
# -----------------------------------------------------------------------------
def _existing_cols(df, cols):
    """
    Return only the requested columns that actually exist in the DataFrame.

    Args:
        df:
            Input pandas DataFrame.

        cols:
            Requested column names.

    Returns:
        List[str]:
            Existing column names only.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return []

    return [c for c in cols if c in df.columns]

# -----------------------------------------------------------------------------
# 5. HELPER: CANDIDATE KEY
# -----------------------------------------------------------------------------
def _candidate_key(image_id, prompt, det_idx):
    """
    Build the same mask-lookup key used in CELL 14C / 15A.

    Args:
        image_id:
            Stable image identifier.

        prompt:
            Pole prompt used for the selected detection.

        det_idx:
            Detection index within that prompt run.

    Returns:
        tuple:
            (image_id, prompt, det_idx) as a stable lookup key.
    """
    return (str(image_id), str(prompt), int(det_idx))

# -----------------------------------------------------------------------------
# 6. HELPER: SAFE FILE-STEM
# -----------------------------------------------------------------------------
def make_safe_stem(text):
    """
    Convert a string into a filesystem-safe stem.

    Args:
        text:
            Input string.

    Returns:
        str:
            Filesystem-safe stem.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        text = "image"

    text = str(text).strip()
    if len(text) == 0:
        text = "image"

    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text if len(text) > 0 else "image"

# -----------------------------------------------------------------------------
# 7. HELPER: SHIFT A FIXED BOX INSIDE THE IMAGE WHEN POSSIBLE
# -----------------------------------------------------------------------------
def shift_box_inside_image(x1, y1, box_w, box_h, image_w, image_h):
    """
    Shift a fixed-size box inside the image when possible, while keeping the
    box size unchanged.

    Args:
        x1, y1:
            Requested top-left corner.

        box_w, box_h:
            Fixed box dimensions.

        image_w, image_h:
            Image dimensions.

    Returns:
        Dict[str, int]:
            Shifted fixed-size box coordinates.
    """
    x1 = int(round(x1))
    y1 = int(round(y1))
    box_w = int(box_w)
    box_h = int(box_h)

    x2 = x1 + box_w
    y2 = y1 + box_h

    # Shift horizontally if the full box can fit inside the image.
    if image_w >= box_w:
        if x1 < 0:
            x2 += (-x1)
            x1 = 0
        if x2 > image_w:
            x1 -= (x2 - image_w)
            x2 = image_w

    # Shift vertically if the full box can fit inside the image.
    if image_h >= box_h:
        if y1 < 0:
            y2 += (-y1)
            y1 = 0
        if y2 > image_h:
            y1 -= (y2 - image_h)
            y2 = image_h

    # Recompute the far corner from the final fixed-size origin.
    x2 = x1 + box_w
    y2 = y1 + box_h

    return {
        "req_x1": int(x1),
        "req_y1": int(y1),
        "req_x2": int(x2),
        "req_y2": int(y2),
        "req_w": int(box_w),
        "req_h": int(box_h),
    }

# -----------------------------------------------------------------------------
# 8. HELPER: BUILD THE FIXED POLE-TOP ROI REQUEST
# -----------------------------------------------------------------------------
def build_pole_top_roi_request(row):
    """
    Build a fixed-size pole-top ROI request from the selected pole row.

    Args:
        row:
            Selected pole row from pole_selection_df.

    Returns:
        Dict[str, Any]:
            Requested fixed-size ROI geometry after shift-to-fit.
    """
    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])

    image_w = int(row["image_w"])
    image_h = int(row["image_h"])

    pole_w = max(x2 - x1, 1.0)
    pole_h = max(y2 - y1, 1.0)

    # Use the provided pole center when available.
    if "pole_cx" in row.index and pd.notna(row["pole_cx"]):
        pole_cx = float(row["pole_cx"])
    else:
        pole_cx = (x1 + x2) / 2.0

    # Anchor the requested ROI from the pole top.
    req_x1 = pole_cx - (FIXED_ROI_WIDTH / 2.0)
    req_y1 = y1 - POLE_TOP_BUFFER_ABOVE

    # Shift the fixed-size ROI inside the image first when possible.
    shifted = shift_box_inside_image(
        x1=req_x1,
        y1=req_y1,
        box_w=FIXED_ROI_WIDTH,
        box_h=FIXED_ROI_HEIGHT,
        image_w=image_w,
        image_h=image_h,
    )

    return {
        "pole_w": float(pole_w),
        "pole_h": float(pole_h),
        "pole_cx_used": float(pole_cx),
        "req_x1": int(shifted["req_x1"]),
        "req_y1": int(shifted["req_y1"]),
        "req_x2": int(shifted["req_x2"]),
        "req_y2": int(shifted["req_y2"]),
        "req_w": int(shifted["req_w"]),
        "req_h": int(shifted["req_h"]),
    }

# -----------------------------------------------------------------------------
# 9. HELPER: RENDER A FIXED-SIZE CANVAS FROM THE REQUESTED ROI
# -----------------------------------------------------------------------------
def render_fixed_canvas_roi(image_pil, roi_request):
    """
    Render the final fixed-size ROI canvas by cropping the overlapping source
    region and pasting it onto a fixed-size canvas.

    Args:
        image_pil:
            Source PIL RGB image.

        roi_request:
            Dict from build_pole_top_roi_request.

    Returns:
        Dict[str, Any]:
            Final crop details and the fixed-size PIL canvas.
    """
    image_w, image_h = image_pil.size

    req_x1 = int(roi_request["req_x1"])
    req_y1 = int(roi_request["req_y1"])
    req_x2 = int(roi_request["req_x2"])
    req_y2 = int(roi_request["req_y2"])

    # Compute the overlapping source region.
    src_x1 = max(0, req_x1)
    src_y1 = max(0, req_y1)
    src_x2 = min(image_w, req_x2)
    src_y2 = min(image_h, req_y2)

    overlap_w = max(0, src_x2 - src_x1)
    overlap_h = max(0, src_y2 - src_y1)

    # Compute where the source crop should be pasted on the fixed canvas.
    dst_x1 = max(0, src_x1 - req_x1)
    dst_y1 = max(0, src_y1 - req_y1)

    # Create the fixed-size final canvas.
    roi_canvas = Image.new("RGB", (FIXED_ROI_WIDTH, FIXED_ROI_HEIGHT), PAD_RGB)

    # Paste the overlapping image crop onto the fixed-size canvas.
    if overlap_w > 0 and overlap_h > 0:
        src_crop = image_pil.crop((src_x1, src_y1, src_x2, src_y2))
        roi_canvas.paste(src_crop, (dst_x1, dst_y1))

    # Compute padding sizes on each side.
    pad_left   = int(max(0, -req_x1))
    pad_top    = int(max(0, -req_y1))
    pad_right  = int(max(0, req_x2 - image_w))
    pad_bottom = int(max(0, req_y2 - image_h))

    return {
        "roi_canvas": roi_canvas,
        "src_x1":     int(src_x1),
        "src_y1":     int(src_y1),
        "src_x2":     int(src_x2),
        "src_y2":     int(src_y2),
        "src_w":      int(overlap_w),
        "src_h":      int(overlap_h),
        "dst_x1":     int(dst_x1),
        "dst_y1":     int(dst_y1),
        "pad_left":   int(pad_left),
        "pad_top":    int(pad_top),
        "pad_right":  int(pad_right),
        "pad_bottom": int(pad_bottom),
        "was_padded": bool(
            (pad_left > 0) or (pad_top > 0) or
            (pad_right > 0) or (pad_bottom > 0)
        ),
    }

# -----------------------------------------------------------------------------
# 10. HELPER: BUILD DEBUG-STYLE LABELS
# -----------------------------------------------------------------------------
def build_pole_overlay_label(row):
    """
    Build the red pole-box label shown on the source-image gallery.

    Args:
        row:
            Row from pole_rois_df.

    Returns:
        str:
            Label string.
    """
    label_bits = ["POLE"]

    prompt_text = None
    if "prompt" in row.index and pd.notna(row["prompt"]) and str(row["prompt"]).strip():
        prompt_text = str(row["prompt"]).strip()
    elif "POLE_PROMPT" in globals():
        prompt_text = str(POLE_PROMPT).strip()

    if prompt_text:
        label_bits.append(prompt_text)

    if "score" in row.index and pd.notna(row["score"]):
        label_bits.append(f"score={float(row['score']):.3f}")

    if "final_score" in row.index and pd.notna(row["final_score"]):
        label_bits.append(f"final={float(row['final_score']):.3f}")

    return " | ".join(label_bits)


def build_crop_overlay_label(row):
    """
    Build the cyan crop-box label shown on the source-image gallery.

    Args:
        row:
            Row from pole_rois_df.

    Returns:
        str:
            Label string.
    """
    label_bits = ["CROP BOX"]

    if "req_w" in row.index and "req_h" in row.index:
        if pd.notna(row["req_w"]) and pd.notna(row["req_h"]):
            label_bits.append(f"{int(row['req_w'])}x{int(row['req_h'])}")

    if "was_padded" in row.index and bool(row["was_padded"]):
        label_bits.append("padded")

    return " | ".join(label_bits)

# -----------------------------------------------------------------------------
# 11. HELPER: DRAW SOURCE IMAGE WITH POLE + ROI BOXES
# -----------------------------------------------------------------------------
def draw_source_and_roi_gallery(pole_rois_df, gallery_count=6):
    """
    Show a small gallery with the selected pole box and the requested ROI box.
    Left panel: source image with red pole box and cyan crop box overlaid.
    Right panel: saved fixed-size ROI crop.

    Args:
        pole_rois_df:
            ROI manifest DataFrame.

        gallery_count:
            Maximum number of rows to display.

    Returns:
        None
    """
    gallery_df = pole_rois_df.head(gallery_count).copy()

    if gallery_df.empty:
        print("No ROI gallery rows to display.")
        return

    fig, axes = plt.subplots(len(gallery_df), 2, figsize=(16, 6 * len(gallery_df)))

    # Normalise axes shape for the single-row case.
    if len(gallery_df) == 1:
        axes = np.array([axes])

    for row_idx, (_, r) in enumerate(gallery_df.iterrows()):
        ax_left  = axes[row_idx, 0]
        ax_right = axes[row_idx, 1]

        # Reload the source image for display only.
        with Image.open(r["image_path"]) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            img.thumbnail((1600, 1600))
            disp_w, disp_h = img.size
            scale_x = disp_w / orig_w if orig_w > 0 else 1.0
            scale_y = disp_h / orig_h if orig_h > 0 else 1.0
            image_rgb = np.array(img)

        ax_left.imshow(image_rgb)
        ax_left.set_title(
            f"{r['file_name']}\nselected pole + requested ROI", fontsize=11
        )
        ax_left.axis("off")

        # Red selected pole box (scaled to display size).
        pole_x1 = float(r["x1"]) * scale_x
        pole_y1 = float(r["y1"]) * scale_y
        pole_w  = (float(r["x2"]) - float(r["x1"])) * scale_x
        pole_h  = (float(r["y2"]) - float(r["y1"])) * scale_y

        pole_rect = patches.Rectangle(
            (pole_x1, pole_y1),
            pole_w,
            pole_h,
            linewidth=3.0,
            edgecolor="red",
            facecolor="none",
        )
        ax_left.add_patch(pole_rect)

        pole_label = build_pole_overlay_label(r)
        ax_left.text(
            max(pole_x1, 8),
            max(pole_y1 - 6, 8),
            pole_label,
            fontsize=8.5,
            color="white",
            bbox=dict(facecolor="red", alpha=0.85, edgecolor="none", pad=2.0),
        )

        # Cyan dashed requested ROI box (scaled to display size).
        crop_x1 = float(r["req_x1"]) * scale_x
        crop_y1 = float(r["req_y1"]) * scale_y
        crop_w  = (float(r["req_x2"]) - float(r["req_x1"])) * scale_x
        crop_h  = (float(r["req_y2"]) - float(r["req_y1"])) * scale_y

        roi_rect = patches.Rectangle(
            (crop_x1, crop_y1),
            crop_w,
            crop_h,
            linewidth=2.5,
            edgecolor="cyan",
            facecolor="none",
            linestyle="--",
        )
        ax_left.add_patch(roi_rect)

        crop_label = build_crop_overlay_label(r)
        ax_left.text(
            max(crop_x1, 8),
            max(crop_y1 - 6, 8),
            crop_label,
            fontsize=8.5,
            color="black",
            bbox=dict(facecolor="cyan", alpha=0.85, edgecolor="none", pad=2.0),
        )

        # Right panel: saved fixed-size ROI.
        with Image.open(r["roi_image_path"]) as crop_img:
            crop_img = crop_img.convert("RGB")
            crop_rgb = np.array(crop_img)

        crop_title = (
            f"{r['roi_file_name']}\n"
            f"ROI size={int(r['roi_w'])}x{int(r['roi_h'])}"
        )
        if "was_padded" in r and bool(r["was_padded"]):
            crop_title += " | padded"

        ax_right.imshow(crop_rgb)
        ax_right.set_title(crop_title, fontsize=11)
        ax_right.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()

# -----------------------------------------------------------------------------
# 12. RESET THE SILVER ROI FOLDER
# -----------------------------------------------------------------------------
# Clear the old ROI folder so there are no stale crops from a previous run.
if os.path.isdir(SILVER_POLE_ROIS):
    shutil.rmtree(SILVER_POLE_ROIS)

os.makedirs(SILVER_POLE_ROIS, exist_ok=True)

# -----------------------------------------------------------------------------
# 13. BUILD FIXED-SIZE ROI CANVASES AND SAVE TO SILVER
# -----------------------------------------------------------------------------
roi_rows = []

print(f"\nCreating fixed pole-top ROI crops for {len(selected_poles_df)} selected image(s)...")

for idx, row in selected_poles_df.iterrows():
    image_id   = row["image_id"]   if "image_id"   in row.index and pd.notna(row["image_id"])   else None
    file_name  = row["file_name"]  if "file_name"  in row.index and pd.notna(row["file_name"])  else None
    image_path = row["image_path"] if "image_path" in row.index and pd.notna(row["image_path"]) else None

    if pd.isna(image_path) or not isinstance(image_path, str) or len(image_path.strip()) == 0:
        raise ValueError("A selected pole row is missing a valid image_path.")

    if (
        pd.isna(file_name)
        or not isinstance(file_name, str)
        or len(file_name.strip()) == 0
    ):
        file_name = os.path.basename(image_path)

    # -------------------------------------------------------------------------
    # Carry forward selected-pole identity fields from CELL 14C.
    # These keep pole_rois_df linked back to the selected pole and its mask.
    # -------------------------------------------------------------------------
    prompt = row["prompt"] if "prompt" in row.index and pd.notna(row["prompt"]) else None
    det_idx = int(row["det_idx"]) if "det_idx" in row.index and pd.notna(row["det_idx"]) else np.nan
    selection_mode = row["selection_mode"] if "selection_mode" in row.index and pd.notna(row["selection_mode"]) else None
    has_mask = bool(row["has_mask"]) if "has_mask" in row.index and pd.notna(row["has_mask"]) else False
    final_score = float(row["final_score"]) if "final_score" in row.index and pd.notna(row["final_score"]) else np.nan

    mask_lookup_hit = False
    if (
        "pole_mask_lookup" in globals()
        and isinstance(globals().get("pole_mask_lookup"), dict)
        and prompt is not None
        and pd.notna(det_idx)
    ):
        mask_lookup_hit = _candidate_key(image_id, prompt, int(det_idx)) in pole_mask_lookup

    # Build the fixed pole-top ROI request.
    roi_request = build_pole_top_roi_request(row)

    # Build a stable saved crop filename.
    roi_stem       = make_safe_stem(image_id if image_id is not None else os.path.splitext(file_name)[0])
    roi_file_name  = f"{roi_stem}__pole_roi.png"
    roi_image_path = os.path.join(SILVER_POLE_ROIS, roi_file_name)

    # Open the source image and render the fixed-size ROI canvas.
    with Image.open(image_path) as img:
        img = img.convert("RGB")

        roi_render = render_fixed_canvas_roi(
            image_pil=img,
            roi_request=roi_request,
        )

        # Save the fixed-size ROI canvas to Silver.
        roi_render["roi_canvas"].save(roi_image_path, format="PNG")

    roi_rows.append({
        "image_id":         image_id,
        "file_name":        file_name,
        "image_path":       image_path,

        # ---------------------------------------------------------------------
        # Carry-forward selected-pole identity fields
        # ---------------------------------------------------------------------
        "selection_status": "selected",
        "selection_mode":   selection_mode,
        "prompt":           prompt,
        "det_idx":          det_idx,
        "score":            float(row["score"]) if "score" in row.index and pd.notna(row["score"]) else np.nan,
        "final_score":      final_score,
        "has_mask":         has_mask,
        "mask_lookup_hit":  bool(mask_lookup_hit),

        # ---------------------------------------------------------------------
        # Full image geometry
        # ---------------------------------------------------------------------
        "image_w":          int(row["image_w"]),
        "image_h":          int(row["image_h"]),

        # ---------------------------------------------------------------------
        # Selected pole geometry
        # ---------------------------------------------------------------------
        "x1":               float(row["x1"]),
        "y1":               float(row["y1"]),
        "x2":               float(row["x2"]),
        "y2":               float(row["y2"]),
        "pole_cx":          float(row["pole_cx"]) if "pole_cx" in row.index and pd.notna(row["pole_cx"]) else np.nan,
        "pole_cy":          float(row["pole_cy"]) if "pole_cy" in row.index and pd.notna(row["pole_cy"]) else np.nan,
        "pole_w":           float(roi_request["pole_w"]),
        "pole_h":           float(roi_request["pole_h"]),
        "pole_cx_used":     float(roi_request["pole_cx_used"]),

        # ---------------------------------------------------------------------
        # Requested fixed ROI geometry
        # ---------------------------------------------------------------------
        "req_x1":           int(roi_request["req_x1"]),
        "req_y1":           int(roi_request["req_y1"]),
        "req_x2":           int(roi_request["req_x2"]),
        "req_y2":           int(roi_request["req_y2"]),
        "req_w":            int(roi_request["req_w"]),
        "req_h":            int(roi_request["req_h"]),

        # ---------------------------------------------------------------------
        # Actual source overlap / paste geometry
        # ---------------------------------------------------------------------
        "src_x1":           int(roi_render["src_x1"]),
        "src_y1":           int(roi_render["src_y1"]),
        "src_x2":           int(roi_render["src_x2"]),
        "src_y2":           int(roi_render["src_y2"]),
        "src_w":            int(roi_render["src_w"]),
        "src_h":            int(roi_render["src_h"]),
        "dst_x1":           int(roi_render["dst_x1"]),
        "dst_y1":           int(roi_render["dst_y1"]),
        "pad_left":         int(roi_render["pad_left"]),
        "pad_top":          int(roi_render["pad_top"]),
        "pad_right":        int(roi_render["pad_right"]),
        "pad_bottom":       int(roi_render["pad_bottom"]),
        "was_padded":       bool(roi_render["was_padded"]),

        # ---------------------------------------------------------------------
        # Saved fixed canvas ROI output
        # ---------------------------------------------------------------------
        "roi_w":            int(FIXED_ROI_WIDTH),
        "roi_h":            int(FIXED_ROI_HEIGHT),
        "roi_file_name":    roi_file_name,
        "roi_image_path":   roi_image_path,
        "source_layer":     "silver",
        "source_folder":    SILVER_POLE_ROIS,
    })

    if (len(roi_rows) % 20 == 0) or (len(roi_rows) == 1) or (len(roi_rows) == len(selected_poles_df)):
        print(f"  [{len(roi_rows)}/{len(selected_poles_df)}] {roi_file_name}")

    del roi_render

gc.collect()

# -----------------------------------------------------------------------------
# 14. BUILD THE ROI MANIFEST
# -----------------------------------------------------------------------------
pole_rois_df = pd.DataFrame(roi_rows)

if pole_rois_df.empty:
    raise ValueError("pole_rois_df ended up empty. Please check the crop/save loop.")

# -----------------------------------------------------------------------------
# 15. PRINT SUMMARY
# -----------------------------------------------------------------------------
print("\n" + "=" * 100)
print("FIXED POLE-TOP ROI SUMMARY")
print("=" * 100)
print(f"Selected poles used      : {len(selected_poles_df)}")
print(f"Saved ROI crops          : {len(pole_rois_df)}")
print(f"Silver ROI folder        : {SILVER_POLE_ROIS}")
print(f"Fixed ROI size           : {FIXED_ROI_WIDTH}x{FIXED_ROI_HEIGHT}")
print(
    f"Padded ROI count         : "
    f"{int(pole_rois_df['was_padded'].sum()) if 'was_padded' in pole_rois_df.columns else 0}"
)

# -----------------------------------------------------------------------------
# 16. DISPLAY THE ROI TABLE
# -----------------------------------------------------------------------------
roi_cols = _existing_cols(
    pole_rois_df,
    [
        "image_id",
        "file_name",
        "prompt",
        "det_idx",
        "score",
        "final_score",
        "has_mask",
        "mask_lookup_hit",
        "selection_mode",
        "pole_w",
        "pole_h",
        "req_x1",
        "req_y1",
        "req_x2",
        "req_y2",
        "roi_w",
        "roi_h",
        "pad_left",
        "pad_top",
        "pad_right",
        "pad_bottom",
        "was_padded",
        "roi_file_name",
        "roi_image_path",
    ]
)

_safe_display(
    pole_rois_df[roi_cols]
    .sort_values(["file_name"], ascending=[True])
    .reset_index(drop=True)
)

# -----------------------------------------------------------------------------
# 17. SHOW A SMALL ROI GALLERY
# -----------------------------------------------------------------------------
print("\n" + "=" * 100)
print("FIXED POLE-TOP ROI GALLERY")
print("=" * 100)

draw_source_and_roi_gallery(
    pole_rois_df=pole_rois_df,
    gallery_count=POLE_ROI_GALLERY_COUNT,
)

# -----------------------------------------------------------------------------
# 18. SAVE OUTPUTS
# -----------------------------------------------------------------------------
if "save_state" in globals():
    save_state(
        df_names=[
            name for name in [
                "pole_selection_df",
                "pole_rois_df",
            ]
            if isinstance(globals().get(name), pd.DataFrame)
        ],
        config_extra={
            "FIXED_ROI_WIDTH":        FIXED_ROI_WIDTH,
            "FIXED_ROI_HEIGHT":       FIXED_ROI_HEIGHT,
            "POLE_TOP_BUFFER_ABOVE":  POLE_TOP_BUFFER_ABOVE,
            "PAD_RGB":                list(PAD_RGB),
            "POLE_ROI_GALLERY_COUNT": POLE_ROI_GALLERY_COUNT,
        },
        nb_globals=globals(),
    )
else:
    print(
        "Note: save_state not available in this Databricks notebook; "
        "outputs remain in globals only."
    )

print("\nCELL 15B completed.")
print("Saved outputs:")
print("  - pole_rois_df")




# =============================================================================
# CELL 16A — CROSSARM DETECTION ON ONE POLE ROI CROP (DEBUG)
#             + CONTAINMENT SUPPRESSION
#             + MAIN-CLUSTER FALSE-POSITIVE FILTER
#             + POLE MASK OVERLAP FILTER
#             + CROSSARM STRUCTURE FILTER
#             + CROSSARM LEVEL DEDUPE FILTER
#             + OPTIONAL PCA CHECK ON SUSPICIOUS REMAINING BOXES
#             + COLOURED CROSSARM MASKS + YELLOW BOXES
#             + POLE MASK RED OVERLAY ON FINAL DISPLAY
#             + SAVED STAGE SNAPSHOTS + FINAL 2×4 DEBUG GRID
#             + Xarm_1 / Xarm_2 / ... LABELS
# =============================================================================

# -----------------------------------------------------------------------------
# 0. SAFETY CHECKS
# -----------------------------------------------------------------------------
required_globals = [
    "pole_rois_df",
    "model",
    "processor",
    "DEVICE",
    "GLOBAL_TEXT_SCORE_THRESHOLD",
]

missing_globals = [name for name in required_globals if name not in globals()]
if missing_globals:
    raise NameError(
        "CELL 16A requires objects from earlier cells.\n"
        "Please run CELL 15B and the model setup cells first.\n"
        f"Missing globals: {missing_globals}"
    )

if not isinstance(pole_rois_df, pd.DataFrame):
    raise TypeError("pole_rois_df exists but is not a pandas DataFrame.")

if pole_rois_df.empty:
    raise ValueError("pole_rois_df is empty. Please check CELL 15B.")

if "roi_image_path" not in pole_rois_df.columns:
    raise ValueError(
        "pole_rois_df does not contain roi_image_path.\n"
        "Please check CELL 15B."
    )

POLE_MASK_LOOKUP_AVAILABLE = (
    "pole_mask_lookup" in globals()
    and isinstance(globals().get("pole_mask_lookup"), dict)
    and len(globals().get("pole_mask_lookup")) > 0
)

if not POLE_MASK_LOOKUP_AVAILABLE:
    print(
        "NOTE: pole_mask_lookup is not available or is empty.\n"
        "The pole mask overlap filter will be skipped.\n"
        "Run CELL 14C first to enable pole mask filtering."
    )

# -----------------------------------------------------------------------------
# 1. PREPARE SAFE ROI INPUT TABLE
# -----------------------------------------------------------------------------
roi_input_df = pole_rois_df.copy()

roi_input_df = roi_input_df[
    roi_input_df["roi_image_path"].notna()
].copy()

if "selection_status" in roi_input_df.columns:
    roi_input_df = roi_input_df[
        roi_input_df["selection_status"] == "selected"
    ].copy()

roi_input_df = roi_input_df.reset_index(drop=True)

if roi_input_df.empty:
    raise ValueError(
        "No usable ROI rows were found in pole_rois_df.\n"
        "Please check CELL 15B output."
    )

# -----------------------------------------------------------------------------
# 2. CONFIG
# -----------------------------------------------------------------------------
#CROSSARM_ROI_DEBUG_ROW_INDEX = int(globals().get("CROSSARM_ROI_DEBUG_ROW_INDEX", 0))

CROSSARM_ROI_DEBUG_ROW_INDEX = 0

PROMPT_TEXT = "utility pole crossarm"

# keep threshold at 0.3 throughout
TEXT_THRESHOLD = 0.30
RUN_DEVICE = DEVICE

# Optional diagnostic to check whether SAM3 itself can render masks
RUN_PLOT_RESULTS_DIAGNOSTIC = False
_plot_results_shown = False

# Containment suppression settings
CONTAINMENT_THRESHOLD = 0.80
MIN_AREA_RATIO = 1.20
MIN_SCORE_ADVANTAGE = 0.0

# Main-cluster filtering settings
CENTER_DIST_FACTOR = 2.75

# Pole mask overlap filter settings
POLE_MASK_FILTER_ENABLED = True
POLE_OVERLAP_MIN_FRACTION = 0.02

# Fuse-arm / small-arm suppression settings
CROSSARM_STRUCTURE_FILTER_ENABLED = True
CROSSARM_MIN_ASPECT_RATIO = 1.0
POLE_ATTACH_MARGIN_PX = 120
MIN_RELATIVE_WIDTH_TO_MAX = 0.35

# Crossarm level dedupe filter settings
CROSSARM_LEVEL_FILTER_ENABLED = True
CROSSARM_LEVEL_BAND_FACTOR = 0.60
MAX_BOX_H_TO_MEDIAN_RATIO = 1.80
KEEP_PER_LEVEL = 3

# Optional PCA filter settings
CROSSARM_PCA_FILTER_ENABLED = True
PCA_SUSPICIOUS_ASPECT_MAX = 2.20
PCA_SUSPICIOUS_HEIGHT_TO_MEDIAN_MIN = 1.15
PCA_SUSPICIOUS_REL_WIDTH_MAX = 0.85
PCA_MIN_MASK_PIXELS = 80
PCA_MIN_PC1_RATIO = 0.85
PCA_MIN_ANISOTROPY = 4.00

# Display settings
CROSSARM_MASK_ALPHA = 0.40
POLE_MASK_ALPHA = 0.30
LABEL_BG = "#1E90FF"

# Stage-grid settings
SHOW_STAGE_GRID = True
GRID_FIGSIZE = (20, 10)

print("Single-prompt SAM3 crossarm debug run on pole ROI crop:\n")
print(f"  CROSSARM_ROI_DEBUG_ROW_INDEX        : {CROSSARM_ROI_DEBUG_ROW_INDEX}")
print(f"  roi_input_df rows                   : {len(roi_input_df)}")
print(f"  prompt                              : {PROMPT_TEXT}")
print(f"  TEXT_THRESHOLD                      : {TEXT_THRESHOLD}")
print(f"  RUN_DEVICE                          : {RUN_DEVICE}")
print(f"  CONTAINMENT_THRESHOLD               : {CONTAINMENT_THRESHOLD}")
print(f"  MIN_AREA_RATIO                      : {MIN_AREA_RATIO}")
print(f"  MIN_SCORE_ADVANTAGE                 : {MIN_SCORE_ADVANTAGE}")
print(f"  CENTER_DIST_FACTOR                  : {CENTER_DIST_FACTOR}")
print(f"  POLE_MASK_FILTER_ENABLED            : {POLE_MASK_FILTER_ENABLED}")
print(f"  POLE_OVERLAP_MIN_FRACTION           : {POLE_OVERLAP_MIN_FRACTION}")
print(f"  CROSSARM_STRUCTURE_FILTER_ENABLED   : {CROSSARM_STRUCTURE_FILTER_ENABLED}")
print(f"  CROSSARM_MIN_ASPECT_RATIO           : {CROSSARM_MIN_ASPECT_RATIO}")
print(f"  POLE_ATTACH_MARGIN_PX               : {POLE_ATTACH_MARGIN_PX}")
print(f"  MIN_RELATIVE_WIDTH_TO_MAX           : {MIN_RELATIVE_WIDTH_TO_MAX}")
print(f"  CROSSARM_LEVEL_FILTER_ENABLED       : {CROSSARM_LEVEL_FILTER_ENABLED}")
print(f"  CROSSARM_LEVEL_BAND_FACTOR          : {CROSSARM_LEVEL_BAND_FACTOR}")
print(f"  MAX_BOX_H_TO_MEDIAN_RATIO           : {MAX_BOX_H_TO_MEDIAN_RATIO}")
print(f"  KEEP_PER_LEVEL                      : {KEEP_PER_LEVEL}")
print(f"  CROSSARM_PCA_FILTER_ENABLED         : {CROSSARM_PCA_FILTER_ENABLED}")
print(f"  PCA_SUSPICIOUS_ASPECT_MAX           : {PCA_SUSPICIOUS_ASPECT_MAX}")
print(f"  PCA_SUSPICIOUS_HEIGHT_TO_MEDIAN_MIN : {PCA_SUSPICIOUS_HEIGHT_TO_MEDIAN_MIN}")
print(f"  PCA_SUSPICIOUS_REL_WIDTH_MAX        : {PCA_SUSPICIOUS_REL_WIDTH_MAX}")
print(f"  PCA_MIN_MASK_PIXELS                 : {PCA_MIN_MASK_PIXELS}")
print(f"  PCA_MIN_PC1_RATIO                   : {PCA_MIN_PC1_RATIO}")
print(f"  PCA_MIN_ANISOTROPY                  : {PCA_MIN_ANISOTROPY}")
print(f"  POLE_MASK_LOOKUP_AVAILABLE          : {POLE_MASK_LOOKUP_AVAILABLE}")
print(f"  SHOW_STAGE_GRID                     : {SHOW_STAGE_GRID}")

# -----------------------------------------------------------------------------
# 3. HELPER: SAFE DISPLAY
# -----------------------------------------------------------------------------
def _safe_display(obj):
    try:
        display(obj)
    except Exception:
        print(obj)

# -----------------------------------------------------------------------------
# 4. SELECT ONE ROW FROM roi_input_df
# -----------------------------------------------------------------------------
if CROSSARM_ROI_DEBUG_ROW_INDEX >= len(roi_input_df):
    raise IndexError(
        f"CROSSARM_ROI_DEBUG_ROW_INDEX={CROSSARM_ROI_DEBUG_ROW_INDEX} is out of range "
        f"for roi_input_df with {len(roi_input_df)} rows.\n"
        "roi_input_df contains only usable selected pole ROI crops."
    )

row = roi_input_df.iloc[CROSSARM_ROI_DEBUG_ROW_INDEX]

image_id = row.get("image_id", None)
if pd.isna(image_id):
    image_id = None

file_name = row.get("file_name", None)
if pd.isna(file_name) or not isinstance(file_name, str) or len(file_name.strip()) == 0:
    file_name = str(image_id) if image_id is not None else "unknown"

roi_image_path = row.get("roi_image_path", None)
roi_file_name = row.get("roi_file_name", None)

if pd.isna(roi_image_path) or not isinstance(roi_image_path, str) or len(roi_image_path.strip()) == 0:
    raise ValueError(
        f"roi_image_path is missing for CROSSARM_ROI_DEBUG_ROW_INDEX={CROSSARM_ROI_DEBUG_ROW_INDEX}.\n"
        "Please check CELL 15B completed correctly."
    )

if not os.path.exists(roi_image_path):
    raise FileNotFoundError(
        f"ROI crop file not found:\n{roi_image_path}\n"
        "Please check CELL 15B saved files to SILVER_POLE_ROIS correctly."
    )

display_title = (
    roi_file_name
    if isinstance(roi_file_name, str) and len(roi_file_name.strip()) > 0
    else os.path.basename(roi_image_path)
)

roi_file_name = (
    roi_file_name
    if isinstance(roi_file_name, str) and len(roi_file_name.strip()) > 0
    else display_title
)

pole_prompt = row.get("prompt", None)
if pd.isna(pole_prompt):
    pole_prompt = None

_pole_det_idx_raw = row.get("det_idx", None)
pole_det_idx = int(_pole_det_idx_raw) if (
    _pole_det_idx_raw is not None and pd.notna(_pole_det_idx_raw)
) else None

_src_x1 = row.get("src_x1", None)
_src_y1 = row.get("src_y1", None)
_src_x2 = row.get("src_x2", None)
_src_y2 = row.get("src_y2", None)
_dst_x1 = row.get("dst_x1", None)
_dst_y1 = row.get("dst_y1", None)

pole_roi_geometry_available = all(
    v is not None and pd.notna(v)
    for v in [_src_x1, _src_y1, _src_x2, _src_y2, _dst_x1, _dst_y1]
)

if pole_roi_geometry_available:
    pole_src_x1 = int(_src_x1)
    pole_src_y1 = int(_src_y1)
    pole_src_x2 = int(_src_x2)
    pole_src_y2 = int(_src_y2)
    pole_dst_x1 = int(_dst_x1)
    pole_dst_y1 = int(_dst_y1)

print(f"\n  image_id             : {image_id}")
print(f"  file_name            : {file_name}")
print(f"  roi_image_path       : {roi_image_path}")
print(f"  display_title        : {display_title}")
print(f"  pole_prompt          : {pole_prompt}")
print(f"  pole_det_idx         : {pole_det_idx}")
print(f"  pole_roi_geometry_ok : {pole_roi_geometry_available}")

# -----------------------------------------------------------------------------
# 5. LOAD ROI CROP
# -----------------------------------------------------------------------------
with Image.open(roi_image_path) as img:
    original_mode = img.mode
    if original_mode != "RGB":
        print(f"WARNING: ROI image mode is '{original_mode}', not 'RGB'. Converting.")
        image = img.convert("RGB")
    else:
        image = img.copy()
    image.load()

roi_w, roi_h = image.size

print(f"  roi_size             : {roi_w} x {roi_h}")
print(f"  original_mode        : {original_mode}")

# -----------------------------------------------------------------------------
# 6. HELPER: TO NUMPY SAFE
# -----------------------------------------------------------------------------
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# -----------------------------------------------------------------------------
# 7. HELPER: INFER / NORMALIZE SAM3 OUTPUTS
# -----------------------------------------------------------------------------
def infer_num_detections(raw_boxes, raw_scores, raw_masks):
    boxes_arr = to_numpy(raw_boxes) if raw_boxes is not None else None
    scores_arr = to_numpy(raw_scores) if raw_scores is not None else None

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

        masks_arr = to_numpy(raw_masks)
        if masks_arr.ndim == 2:
            return 1
        if masks_arr.ndim >= 3:
            return int(masks_arr.shape[0])

    return 0


def normalize_boxes(boxes, num_detections):
    if num_detections <= 0:
        return np.zeros((0, 4), dtype=np.float32)

    if boxes is None:
        return np.zeros((num_detections, 4), dtype=np.float32)

    arr = to_numpy(boxes).astype(np.float32)

    if arr.ndim == 1 and arr.shape[0] == 4:
        arr = arr.reshape(1, 4)

    if arr.ndim != 2 or arr.shape[1] != 4:
        return np.zeros((num_detections, 4), dtype=np.float32)

    if arr.shape[0] < num_detections:
        pad = np.zeros((num_detections - arr.shape[0], 4), dtype=np.float32)
        arr = np.vstack([arr, pad])

    return arr[:num_detections]


def normalize_scores(scores, num_detections):
    if num_detections <= 0:
        return np.zeros((0,), dtype=np.float32)

    if scores is None:
        return np.zeros((num_detections,), dtype=np.float32)

    arr = to_numpy(scores).astype(np.float32).reshape(-1)

    if arr.size < num_detections:
        pad = np.zeros((num_detections - arr.size,), dtype=np.float32)
        arr = np.concatenate([arr, pad])

    return arr[:num_detections]


def normalize_masks(raw_masks, num_detections, image_h, image_w):
    """
    Normalize raw stateful-path masks into a list of 2D boolean masks.
    Uses the same robust style as your pole code.
    """
    if num_detections <= 0:
        return []

    if raw_masks is None:
        return [None] * num_detections

    if isinstance(raw_masks, (list, tuple)):
        mask_items = list(raw_masks)
    else:
        arr = to_numpy(raw_masks)

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

        m = to_numpy(mask_items[det_idx])

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

# -----------------------------------------------------------------------------
# 8. HELPER: CONTAINMENT-BASED SUPPRESSION
# -----------------------------------------------------------------------------
def box_area_xyxy(box_xyxy):
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def suppress_contained_shorter_detections(
    detections_df,
    containment_threshold=0.80,
    min_area_ratio=1.20,
    min_score_advantage=0.0,
):
    if detections_df.empty:
        return detections_df.copy(), detections_df.iloc[0:0].copy()

    df = detections_df.copy().reset_index(drop=True)
    df["box_area"] = df.apply(
        lambda r: box_area_xyxy([r["x1"], r["y1"], r["x2"], r["y2"]]),
        axis=1,
    )

    keep_mask = np.ones(len(df), dtype=bool)
    removal_reason = [None] * len(df)

    for j in range(len(df)):
        if not keep_mask[j]:
            continue

        area_j = float(df.loc[j, "box_area"])
        score_j = float(df.loc[j, "score"])
        box_j = [df.loc[j, "x1"], df.loc[j, "y1"], df.loc[j, "x2"], df.loc[j, "y2"]]

        if area_j <= 0:
            keep_mask[j] = False
            removal_reason[j] = "invalid_box_area"
            continue

        for i in range(len(df)):
            if i == j:
                continue

            area_i = float(df.loc[i, "box_area"])
            score_i = float(df.loc[i, "score"])
            box_i = [df.loc[i, "x1"], df.loc[i, "y1"], df.loc[i, "x2"], df.loc[i, "y2"]]

            if area_i <= 0:
                continue

            inter = intersection_area_xyxy(box_i, box_j)
            containment_of_j_inside_i = inter / area_j if area_j > 0 else 0.0
            area_ratio = area_i / area_j if area_j > 0 else 0.0
            score_advantage = score_i - score_j

            if (
                containment_of_j_inside_i >= containment_threshold
                and area_ratio >= min_area_ratio
                and score_advantage >= min_score_advantage
            ):
                keep_mask[j] = False
                removal_reason[j] = (
                    f"contained_in_orig_{int(df.loc[i, 'orig_det_idx'])}"
                )
                break

    kept_df = df[keep_mask].copy().reset_index(drop=True)
    removed_df = df[~keep_mask].copy().reset_index(drop=True)

    if len(removed_df) > 0:
        removed_df["removal_reason"] = [
            removal_reason[idx] for idx in np.where(~keep_mask)[0]
        ]

    return kept_df, removed_df

# -----------------------------------------------------------------------------
# 9. HELPER: MAIN-CLUSTER FILTERING
# -----------------------------------------------------------------------------
def compute_centers_and_scale(detections_df):
    if detections_df.empty:
        df = detections_df.copy()
        df["cx"] = []
        df["cy"] = []
        df["diag"] = []
        return df, 0.0

    df = detections_df.copy().reset_index(drop=True)
    df["cx"] = (df["x1"] + df["x2"]) / 2.0
    df["cy"] = (df["y1"] + df["y2"]) / 2.0
    df["w"] = (df["x2"] - df["x1"]).clip(lower=0.0)
    df["h"] = (df["y2"] - df["y1"]).clip(lower=0.0)
    df["diag"] = np.sqrt(df["w"] ** 2 + df["h"] ** 2)

    positive_diags = df.loc[df["diag"] > 0, "diag"]
    median_diag = float(positive_diags.median()) if len(positive_diags) > 0 else 0.0

    return df, median_diag


def connected_components_from_center_distance(df, center_dist_factor=2.75):
    n = len(df)

    if n == 0:
        return [], 0.0

    if n == 1:
        return [[0]], float(center_dist_factor * float(df["diag"].iloc[0]))

    positive_diags = df.loc[df["diag"] > 0, "diag"]
    median_diag = float(positive_diags.median()) if len(positive_diags) > 0 else 0.0
    center_dist_threshold = float(center_dist_factor * median_diag)

    adjacency = {i: [] for i in range(n)}
    visited = [False] * n
    components = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = math.hypot(
                float(df.loc[i, "cx"] - df.loc[j, "cx"]),
                float(df.loc[i, "cy"] - df.loc[j, "cy"]),
            )
            if dist <= center_dist_threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        comp = []
        visited[start] = True

        while stack:
            node = stack.pop()
            comp.append(node)
            for nbr in adjacency[node]:
                if not visited[nbr]:
                    visited[nbr] = True
                    stack.append(nbr)

        components.append(sorted(comp))

    return components, center_dist_threshold


def keep_main_detection_cluster(detections_df, center_dist_factor=2.75):
    if detections_df.empty:
        return detections_df.copy(), detections_df.iloc[0:0].copy(), 0.0

    if len(detections_df) == 1:
        df1, median_diag = compute_centers_and_scale(detections_df)
        return (
            df1.reset_index(drop=True),
            df1.iloc[0:0].copy(),
            center_dist_factor * median_diag,
        )

    df, _ = compute_centers_and_scale(detections_df)
    components, cluster_threshold = connected_components_from_center_distance(
        df, center_dist_factor=center_dist_factor,
    )

    best_component = None
    best_key = None

    for comp in components:
        comp_df = df.iloc[comp]
        comp_size = len(comp)
        comp_score_sum = float(comp_df["score"].sum())
        key = (comp_size, comp_score_sum)

        if best_key is None or key > best_key:
            best_key = key
            best_component = comp

    keep_idx = set(best_component)
    kept_df = df.iloc[sorted(keep_idx)].copy().reset_index(drop=True)
    removed_df = df.drop(index=sorted(keep_idx)).copy().reset_index(drop=True)

    if len(removed_df) > 0:
        removed_df["removal_reason"] = "outside_main_cluster"

    return kept_df, removed_df, cluster_threshold

# -----------------------------------------------------------------------------
# 10. HELPER: PROJECT POLE MASK INTO ROI COORDINATES
# -----------------------------------------------------------------------------
def project_pole_mask_to_roi(
    pole_mask,
    src_x1, src_y1, src_x2, src_y2,
    dst_x1, dst_y1,
    roi_w, roi_h,
):
    roi_mask = np.zeros((roi_h, roi_w), dtype=bool)

    if pole_mask is None:
        return roi_mask

    arr = pole_mask

    if isinstance(arr, (list, tuple)):
        if len(arr) == 0:
            return roi_mask
        arr = arr[0]

    arr = to_numpy(arr)

    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 3:
        arr = arr[0]

    if arr.ndim != 2:
        return roi_mask

    arr = arr.astype(bool)
    src_h, src_w = arr.shape

    src_x1 = int(src_x1)
    src_y1 = int(src_y1)
    src_x2 = int(src_x2)
    src_y2 = int(src_y2)
    dst_x1 = int(dst_x1)
    dst_y1 = int(dst_y1)

    clip_src_x1 = max(0, min(src_w, src_x1))
    clip_src_y1 = max(0, min(src_h, src_y1))
    clip_src_x2 = max(0, min(src_w, src_x2))
    clip_src_y2 = max(0, min(src_h, src_y2))

    if clip_src_x2 <= clip_src_x1 or clip_src_y2 <= clip_src_y1:
        return roi_mask

    src_crop = arr[clip_src_y1:clip_src_y2, clip_src_x1:clip_src_x2]

    paste_x1 = dst_x1 + (clip_src_x1 - src_x1)
    paste_y1 = dst_y1 + (clip_src_y1 - src_y1)
    paste_x2 = paste_x1 + (clip_src_x2 - clip_src_x1)
    paste_y2 = paste_y1 + (clip_src_y2 - clip_src_y1)

    dst_clip_x1 = max(0, min(roi_w, paste_x1))
    dst_clip_y1 = max(0, min(roi_h, paste_y1))
    dst_clip_x2 = max(0, min(roi_w, paste_x2))
    dst_clip_y2 = max(0, min(roi_h, paste_y2))

    if dst_clip_x2 <= dst_clip_x1 or dst_clip_y2 <= dst_clip_y1:
        return roi_mask

    src_off_x1 = dst_clip_x1 - paste_x1
    src_off_y1 = dst_clip_y1 - paste_y1
    src_off_x2 = src_off_x1 + (dst_clip_x2 - dst_clip_x1)
    src_off_y2 = src_off_y1 + (dst_clip_y2 - dst_clip_y1)

    roi_mask[dst_clip_y1:dst_clip_y2, dst_clip_x1:dst_clip_x2] = (
        src_crop[src_off_y1:src_off_y2, src_off_x1:src_off_x2]
    )

    return roi_mask


def compute_box_overlap_with_mask(box_xyxy, binary_mask):
    if binary_mask is None:
        return 0.0

    mask = to_numpy(binary_mask)

    if mask.ndim != 2:
        return 0.0

    mask = mask.astype(bool)
    h, w = mask.shape

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    box_area = box_area_xyxy([x1, y1, x2, y2])

    if box_area <= 0:
        return 0.0

    ix1 = int(np.floor(max(0.0, min(w, x1))))
    iy1 = int(np.floor(max(0.0, min(h, y1))))
    ix2 = int(np.ceil(max(0.0, min(w, x2))))
    iy2 = int(np.ceil(max(0.0, min(h, y2))))

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    overlap_pixels = int(mask[iy1:iy2, ix1:ix2].sum())
    return float(overlap_pixels / box_area)


def crop_mask_to_box(mask_bool, box_xyxy):
    if mask_bool is None:
        return np.zeros((0, 0), dtype=bool)

    mask = to_numpy(mask_bool)
    if mask.ndim != 2:
        return np.zeros((0, 0), dtype=bool)

    mask = mask.astype(bool)
    h, w = mask.shape

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]

    ix1 = int(np.floor(max(0.0, min(w, x1))))
    iy1 = int(np.floor(max(0.0, min(h, y1))))
    ix2 = int(np.ceil(max(0.0, min(w, x2))))
    iy2 = int(np.ceil(max(0.0, min(h, y2))))

    if ix2 <= ix1 or iy2 <= iy1:
        return np.zeros((0, 0), dtype=bool)

    return mask[iy1:iy2, ix1:ix2]


def compute_binary_mask_pca_stats(mask_bool):
    out = {
        "valid": False,
        "num_pixels": 0,
        "pc1_ratio": np.nan,
        "pc2_ratio": np.nan,
        "anisotropy": np.nan,
        "perp_std": np.nan,
    }

    if mask_bool is None:
        return out

    mask = to_numpy(mask_bool)
    if mask.ndim != 2:
        return out

    mask = mask.astype(bool)
    ys, xs = np.where(mask)

    num_pixels = int(len(xs))
    out["num_pixels"] = num_pixels

    if num_pixels < 3:
        return out

    coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])

    if coords.shape[0] < 3:
        return out

    coords = coords - coords.mean(axis=0, keepdims=True)

    try:
        cov = np.cov(coords, rowvar=False)
        eigvals, _ = np.linalg.eigh(cov)
    except Exception:
        return out

    eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]

    if len(eigvals) < 2:
        return out

    total_var = float(eigvals.sum())
    if total_var <= 0:
        return out

    pc1_ratio = float(eigvals[0] / total_var)
    pc2_ratio = float(eigvals[1] / total_var)
    anisotropy = float(eigvals[0] / max(eigvals[1], 1e-8))
    perp_std = float(np.sqrt(max(eigvals[1], 0.0)))

    out.update({
        "valid": True,
        "pc1_ratio": pc1_ratio,
        "pc2_ratio": pc2_ratio,
        "anisotropy": anisotropy,
        "perp_std": perp_std,
    })
    return out

# -----------------------------------------------------------------------------
# 11. HELPER: DRAW ONE STAGE TO AN AXIS
# -----------------------------------------------------------------------------
def draw_pole_label(ax, pole_mask):
    if pole_mask is None or not np.any(pole_mask):
        return
    pole_ys, pole_xs = np.where(pole_mask)
    if len(pole_xs) > 0 and len(pole_ys) > 0:
        pole_label_x = float(pole_xs.min())
        pole_label_y = float(max(8, pole_ys.min() - 6))
        ax.text(
            pole_label_x,
            pole_label_y,
            "POLE",
            color="white",
            fontsize=8,
            bbox=dict(facecolor="red", alpha=0.90, pad=0.3, edgecolor="none"),
        )


def plot_stage_on_ax(
    ax,
    image,
    detections_df,
    title,
    projected_pole_mask=None,
    crossarm_mask_lookup=None,
    crossarm_mask_alpha=0.40,
    pole_mask_alpha=0.30,
    label_bg="#1E90FF",
    final_style=False,
):
    ax.imshow(image)

    # First: coloured crossarm masks
    if detections_df is not None and len(detections_df) > 0:
        try:
            cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(detections_df), 1))
        except Exception:
            cmap = plt.cm.get_cmap("tab10", max(len(detections_df), 1))

        for plot_idx, (_, det_row) in enumerate(detections_df.iterrows()):
            orig_idx = int(det_row["orig_det_idx"])
            mask_i = crossarm_mask_lookup.get(orig_idx, None) if crossarm_mask_lookup is not None else None

            if isinstance(mask_i, np.ndarray) and mask_i.ndim == 2 and mask_i.sum() > 0:
                color_rgba = cmap(plot_idx)

                overlay = np.zeros((mask_i.shape[0], mask_i.shape[1], 4), dtype=np.float32)
                overlay[..., 0] = color_rgba[0]
                overlay[..., 1] = color_rgba[1]
                overlay[..., 2] = color_rgba[2]
                overlay[..., 3] = mask_i.astype(np.float32) * crossarm_mask_alpha
                ax.imshow(overlay)

    # Second: red pole mask overlay
    if projected_pole_mask is not None and np.any(projected_pole_mask):
        pole_overlay = np.zeros(
            (projected_pole_mask.shape[0], projected_pole_mask.shape[1], 4),
            dtype=np.float32
        )
        pole_overlay[..., 0] = 1.0
        pole_overlay[..., 1] = 0.0
        pole_overlay[..., 2] = 0.0
        pole_overlay[..., 3] = projected_pole_mask.astype(np.float32) * pole_mask_alpha
        ax.imshow(pole_overlay)
        draw_pole_label(ax, projected_pole_mask)

    # Third: yellow boxes + blue labels
    if detections_df is not None and len(detections_df) > 0:
        for _, det_row in detections_df.iterrows():
            x1, y1, x2, y2 = [
                float(det_row["x1"]),
                float(det_row["y1"]),
                float(det_row["x2"]),
                float(det_row["y2"]),
            ]
            score_i = float(det_row["score"])

            rect = patches.Rectangle(
                (x1, y1),
                max(0.0, x2 - x1),
                max(0.0, y2 - y1),
                linewidth=2.0,
                edgecolor="yellow",
                facecolor="none",
            )
            ax.add_patch(rect)

            if final_style and "xarm_label" in detections_df.columns and pd.notna(det_row.get("xarm_label", np.nan)):
                label_text = f"{det_row['xarm_label']}: {score_i:.2f}"
            else:
                label_text = f"idx {int(det_row['orig_det_idx'])}: {score_i:.2f}"

            ax.text(
                x1,
                max(5, y1 - 5),
                label_text,
                color="white",
                fontsize=7,
                bbox=dict(facecolor=label_bg, alpha=0.95, pad=0.4, edgecolor="none"),
            )

    ax.set_title(f"{title}\nN = {0 if detections_df is None else len(detections_df)}", fontsize=10)
    ax.axis("off")

# -----------------------------------------------------------------------------
# 12. PROJECT POLE MASK INTO ROI SPACE (IF AVAILABLE)
# -----------------------------------------------------------------------------
projected_pole_mask = np.zeros((roi_h, roi_w), dtype=bool)
projected_pole_mask_available = False
pole_mask_filter_applied = False

if (
    POLE_MASK_LOOKUP_AVAILABLE
    and pole_roi_geometry_available
    and image_id is not None
    and pole_prompt is not None
    and pole_det_idx is not None
):
    pole_key = (image_id, pole_prompt, pole_det_idx)
    original_pole_mask = pole_mask_lookup.get(pole_key, None)

    if original_pole_mask is not None:
        projected_pole_mask = project_pole_mask_to_roi(
            pole_mask=original_pole_mask,
            src_x1=pole_src_x1,
            src_y1=pole_src_y1,
            src_x2=pole_src_x2,
            src_y2=pole_src_y2,
            dst_x1=pole_dst_x1,
            dst_y1=pole_dst_y1,
            roi_w=roi_w,
            roi_h=roi_h,
        )

        if projected_pole_mask.any():
            projected_pole_mask_available = True

print(f"\nProjected pole mask available   : {projected_pole_mask_available}")

# -----------------------------------------------------------------------------
# 13. RUN SAM3 INFERENCE — STATEFUL PROCESSOR PATH
# -----------------------------------------------------------------------------
if hasattr(processor, "device"):
    processor.device = RUN_DEVICE

if hasattr(processor, "set_confidence_threshold"):
    processor.set_confidence_threshold(TEXT_THRESHOLD)

state = {}
state = processor.set_image(image, state=state)

reset_result = processor.reset_all_prompts(state)
if reset_result is not None:
    state = reset_result

state = processor.set_text_prompt(PROMPT_TEXT, state)

if RUN_PLOT_RESULTS_DIAGNOSTIC and not _plot_results_shown:
    try:
        print("\nSAM3 plot_results diagnostic:")
        print(f"  prompt_text : {PROMPT_TEXT}")
        plot_results(image.copy(), state)
        _plot_results_shown = True
    except Exception as e:
        print(f"plot_results diagnostic failed: {e}")

raw_boxes = state.get("boxes", None)
raw_scores = state.get("scores", None)
raw_masks = state.get("masks", None)

# -----------------------------------------------------------------------------
# 14. NORMALISE OUTPUTS + BUILD CROSSARM MASK LOOKUP
# -----------------------------------------------------------------------------
num_detections = infer_num_detections(raw_boxes, raw_scores, raw_masks)
boxes = normalize_boxes(raw_boxes, num_detections)
scores = normalize_scores(raw_scores, num_detections)
masks_2d = normalize_masks(raw_masks, num_detections, roi_h, roi_w)

crossarm_mask_lookup = {}
for det_idx in range(num_detections):
    mask_i = masks_2d[det_idx] if det_idx < len(masks_2d) else None
    has_mask = isinstance(mask_i, np.ndarray) and mask_i.ndim == 2 and mask_i.sum() > 0
    if has_mask:
        crossarm_mask_lookup[int(det_idx)] = mask_i

HAS_ANY_VALID_MASKS = len(crossarm_mask_lookup) > 0

print(f"\nDetection summary:")
print(f"  raw num detections           : {num_detections}")
print(f"  detections with valid masks  : {len(crossarm_mask_lookup)}")

if not HAS_ANY_VALID_MASKS and num_detections > 0:
    print(
        "NOTE: No usable crossarm masks were returned in a per-detection 2D form.\n"
        "Boxes and labels will still be shown, and pole-mask filtering can still run."
    )

# -----------------------------------------------------------------------------
# 15. BUILD RAW DETECTIONS TABLE + CONTAINMENT SUPPRESSION
# -----------------------------------------------------------------------------
if num_detections == 0:
    print("  No detections were returned for this prompt.")
    raw_detections_df = pd.DataFrame(
        columns=["orig_det_idx", "score", "x1", "y1", "x2", "y2", "has_mask"]
    )
    kept_after_containment_df = raw_detections_df.copy()
    removed_by_containment_df = raw_detections_df.copy()
else:
    raw_detections_df = pd.DataFrame({
        "orig_det_idx": np.arange(num_detections, dtype=int),
        "score": scores.astype(float),
        "x1": boxes[:, 0].astype(float),
        "y1": boxes[:, 1].astype(float),
        "x2": boxes[:, 2].astype(float),
        "y2": boxes[:, 3].astype(float),
        "has_mask": [int(i) in crossarm_mask_lookup for i in range(num_detections)],
    })

    raw_detections_df = raw_detections_df.sort_values(
        "score", ascending=False
    ).reset_index(drop=True)

    print("\nRaw detections table:")
    _safe_display(raw_detections_df)

    kept_after_containment_df, removed_by_containment_df = suppress_contained_shorter_detections(
        detections_df=raw_detections_df,
        containment_threshold=CONTAINMENT_THRESHOLD,
        min_area_ratio=MIN_AREA_RATIO,
        min_score_advantage=MIN_SCORE_ADVANTAGE,
    )

    print(f"\nKept after containment suppression : {len(kept_after_containment_df)}")
    print(f"Removed by containment suppression : {len(removed_by_containment_df)}")

    _safe_display(kept_after_containment_df)

    if len(removed_by_containment_df) > 0:
        print("\nRemoved by containment suppression:")
        _safe_display(removed_by_containment_df)

# Save stage snapshot
stage_raw_df = raw_detections_df.copy()
stage_containment_df = kept_after_containment_df.copy()

# -----------------------------------------------------------------------------
# 16. MAIN-CLUSTER FILTERING
# -----------------------------------------------------------------------------
if len(kept_after_containment_df) == 0:
    kept_after_cluster_df = kept_after_containment_df.copy()
    removed_by_cluster_df = kept_after_containment_df.copy()
    cluster_threshold_used = 0.0
else:
    kept_after_cluster_df, removed_by_cluster_df, cluster_threshold_used = keep_main_detection_cluster(
        detections_df=kept_after_containment_df,
        center_dist_factor=CENTER_DIST_FACTOR,
    )

print(f"\nCluster distance threshold used : {cluster_threshold_used:.2f}")
print(f"Kept after cluster filter       : {len(kept_after_cluster_df)}")
print(f"Removed as isolated false-pos   : {len(removed_by_cluster_df)}")

_safe_display(kept_after_cluster_df)

if len(removed_by_cluster_df) > 0:
    print("\nRemoved by main-cluster filter:")
    _safe_display(removed_by_cluster_df)

# Save stage snapshot
stage_cluster_df = kept_after_cluster_df.copy()

# -----------------------------------------------------------------------------
# 17. POLE MASK OVERLAP FILTER
# -----------------------------------------------------------------------------
if len(kept_after_cluster_df) == 0:
    final_kept_detections_df = kept_after_cluster_df.copy()
    removed_by_pole_mask_df = kept_after_cluster_df.copy()
elif (
    POLE_MASK_FILTER_ENABLED
    and projected_pole_mask_available
):
    pole_mask_filter_applied = True

    overlap_fracs = []
    keep_flags = []

    for _, det_row in kept_after_cluster_df.iterrows():
        box_i = [det_row["x1"], det_row["y1"], det_row["x2"], det_row["y2"]]
        frac = compute_box_overlap_with_mask(box_i, projected_pole_mask)
        overlap_fracs.append(float(frac))
        keep_flags.append(bool(frac >= POLE_OVERLAP_MIN_FRACTION))

    tmp_df = kept_after_cluster_df.copy().reset_index(drop=True)
    tmp_df["pole_overlap_fraction"] = overlap_fracs

    keep_flags_arr = np.array(keep_flags, dtype=bool)

    final_kept_detections_df = tmp_df[keep_flags_arr].copy().reset_index(drop=True)
    removed_by_pole_mask_df = tmp_df[~keep_flags_arr].copy().reset_index(drop=True)

    if len(removed_by_pole_mask_df) > 0:
        removed_by_pole_mask_df["removal_reason"] = (
            "pole_overlap_lt_"
            + f"{POLE_OVERLAP_MIN_FRACTION:.3f}"
        )
else:
    final_kept_detections_df = kept_after_cluster_df.copy()
    removed_by_pole_mask_df = kept_after_cluster_df.iloc[0:0].copy()
    final_kept_detections_df["pole_overlap_fraction"] = np.nan

print(f"\nPole mask filter applied        : {pole_mask_filter_applied}")
print(f"Final kept detections           : {len(final_kept_detections_df)}")
print(f"Removed by pole overlap filter  : {len(removed_by_pole_mask_df)}")

_safe_display(final_kept_detections_df)

if len(removed_by_pole_mask_df) > 0:
    print("\nRemoved by pole overlap filter:")
    _safe_display(removed_by_pole_mask_df)

# Save stage snapshot
stage_pole_overlap_df = final_kept_detections_df.copy()

# -----------------------------------------------------------------------------
# 18. CROSSARM STRUCTURE FILTER
# -----------------------------------------------------------------------------
removed_by_structure_df = final_kept_detections_df.iloc[0:0].copy()
structure_filter_applied = False

if (
    CROSSARM_STRUCTURE_FILTER_ENABLED
    and len(final_kept_detections_df) > 0
    and projected_pole_mask_available
):
    pole_cols = np.where(projected_pole_mask.any(axis=0))[0]

    if len(pole_cols) > 0:
        structure_filter_applied = True

        pole_x1 = int(pole_cols.min())
        pole_x2 = int(pole_cols.max())

        attach_x1 = max(0, pole_x1 - int(POLE_ATTACH_MARGIN_PX))
        attach_x2 = min(roi_w - 1, pole_x2 + int(POLE_ATTACH_MARGIN_PX))

        tmp_df = final_kept_detections_df.copy().reset_index(drop=True)

        tmp_df["box_w"] = (tmp_df["x2"] - tmp_df["x1"]).clip(lower=0.0)
        tmp_df["box_h"] = (tmp_df["y2"] - tmp_df["y1"]).clip(lower=1.0)
        tmp_df["aspect_ratio"] = tmp_df["box_w"] / tmp_df["box_h"]

        tmp_df["keep_aspect_ratio"] = (
            tmp_df["aspect_ratio"] >= CROSSARM_MIN_ASPECT_RATIO
        )

        tmp_df["touches_attach_corridor"] = (
            (tmp_df["x2"] >= attach_x1) &
            (tmp_df["x1"] <= attach_x2)
        )

        max_box_w = float(tmp_df["box_w"].max()) if len(tmp_df) > 0 else 0.0
        if max_box_w > 0:
            tmp_df["relative_width_to_max"] = tmp_df["box_w"] / max_box_w
        else:
            tmp_df["relative_width_to_max"] = 0.0

        tmp_df["keep_relative_width"] = (
            tmp_df["relative_width_to_max"] >= MIN_RELATIVE_WIDTH_TO_MAX
        )

        structure_keep_mask = (
            #tmp_df["keep_aspect_ratio"] &
            tmp_df["touches_attach_corridor"] 
            #tmp_df["keep_relative_width"]
        )

        removed_by_structure_df = tmp_df[~structure_keep_mask].copy().reset_index(drop=True)
        final_kept_detections_df = tmp_df[structure_keep_mask].copy().reset_index(drop=True)

        if len(removed_by_structure_df) > 0:
            removed_by_structure_df["removal_reason"] = "failed_crossarm_structure_filter"

        print(f"\nCrossarm structure filter applied : {structure_filter_applied}")
        print(f"Pole attachment corridor x-range  : [{attach_x1}, {attach_x2}]")
        print(f"CROSSARM_MIN_ASPECT_RATIO        : {CROSSARM_MIN_ASPECT_RATIO}")
        print(f"MIN_RELATIVE_WIDTH_TO_MAX        : {MIN_RELATIVE_WIDTH_TO_MAX}")
        print(f"Removed by structure filter      : {len(removed_by_structure_df)}")
        print(f"Final kept after structure filter: {len(final_kept_detections_df)}")

        if len(removed_by_structure_df) > 0:
            print("\nRemoved by crossarm structure filter:")
            _safe_display(
                removed_by_structure_df[
                    [
                        c for c in [
                            "orig_det_idx",
                            "score",
                            "pole_overlap_fraction",
                            "box_w",
                            "box_h",
                            "aspect_ratio",
                            "relative_width_to_max",
                            "touches_attach_corridor",
                            "keep_aspect_ratio",
                            "keep_relative_width",
                            "x1", "y1", "x2", "y2",
                            "removal_reason",
                        ]
                        if c in removed_by_structure_df.columns
                    ]
                ]
            )

        if len(final_kept_detections_df) > 0:
            print("\nKept after crossarm structure filter:")
            _safe_display(
                final_kept_detections_df[
                    [
                        c for c in [
                            "orig_det_idx",
                            "score",
                            "pole_overlap_fraction",
                            "box_w",
                            "box_h",
                            "aspect_ratio",
                            "relative_width_to_max",
                            "touches_attach_corridor",
                            "keep_aspect_ratio",
                            "keep_relative_width",
                            "x1", "y1", "x2", "y2",
                        ]
                        if c in final_kept_detections_df.columns
                    ]
                ]
            )
    else:
        print("\nCrossarm structure filter skipped — projected pole mask has no x-span.")
else:
    print("\nCrossarm structure filter skipped.")

# Save stage snapshot
stage_structure_df = final_kept_detections_df.copy()

# -----------------------------------------------------------------------------
# 19. CROSSARM LEVEL DEDUPE FILTER
# -----------------------------------------------------------------------------
removed_by_level_df = final_kept_detections_df.iloc[0:0].copy()
level_filter_applied = False
level_median_box_h = 0.0
level_band_threshold = 0.0

if CROSSARM_LEVEL_FILTER_ENABLED and len(final_kept_detections_df) > 0:
    level_filter_applied = True

    tmp_df = final_kept_detections_df.copy().reset_index(drop=True)

    if "cy" not in tmp_df.columns:
        tmp_df["cy"] = (tmp_df["y1"] + tmp_df["y2"]) / 2.0

    if "cx" not in tmp_df.columns:
        tmp_df["cx"] = (tmp_df["x1"] + tmp_df["x2"]) / 2.0

    if "box_w" not in tmp_df.columns:
        tmp_df["box_w"] = (tmp_df["x2"] - tmp_df["x1"]).clip(lower=0.0)

    if "box_h" not in tmp_df.columns:
        tmp_df["box_h"] = (tmp_df["y2"] - tmp_df["y1"]).clip(lower=1.0)

    if "aspect_ratio" not in tmp_df.columns:
        tmp_df["aspect_ratio"] = tmp_df["box_w"] / tmp_df["box_h"]

    if "relative_width_to_max" not in tmp_df.columns:
        max_box_w = float(tmp_df["box_w"].max()) if len(tmp_df) > 0 else 0.0
        if max_box_w > 0:
            tmp_df["relative_width_to_max"] = tmp_df["box_w"] / max_box_w
        else:
            tmp_df["relative_width_to_max"] = 0.0

    level_median_box_h = float(tmp_df["box_h"].median()) if len(tmp_df) > 0 else 0.0
    level_band_threshold = max(40.0, CROSSARM_LEVEL_BAND_FACTOR * level_median_box_h)

    tmp_df = tmp_df.sort_values("cy").reset_index(drop=True)

    band_ids = []
    current_band = 0
    running_band_center = None

    for cy in tmp_df["cy"].tolist():
        if running_band_center is None:
            band_ids.append(current_band)
            running_band_center = float(cy)
        else:
            if abs(float(cy) - running_band_center) <= level_band_threshold:
                band_ids.append(current_band)
                running_band_center = (running_band_center + float(cy)) / 2.0
            else:
                current_band += 1
                band_ids.append(current_band)
                running_band_center = float(cy)

    tmp_df["level_band_id"] = band_ids

    if level_median_box_h > 0:
        tmp_df["keep_not_tall"] = (
            tmp_df["box_h"] <= (MAX_BOX_H_TO_MEDIAN_RATIO * level_median_box_h)
        )
    else:
        tmp_df["keep_not_tall"] = True

    tmp_df["level_rank_score"] = (
        tmp_df["score"]
        + 0.35 * tmp_df["relative_width_to_max"]
    )

    tmp_df["level_rank"] = tmp_df.groupby("level_band_id")["level_rank_score"].rank(
        method="first",
        ascending=False,
    )

    level_keep_mask = (
        tmp_df["keep_not_tall"]
        & (tmp_df["level_rank"] <= KEEP_PER_LEVEL)
    )

    removed_by_level_df = tmp_df[~level_keep_mask].copy().reset_index(drop=True)
    final_kept_detections_df = tmp_df[level_keep_mask].copy().reset_index(drop=True)

    if len(removed_by_level_df) > 0:
        removed_by_level_df["removal_reason"] = "failed_crossarm_level_dedupe"

    print(f"\nCrossarm level filter applied    : {level_filter_applied}")
    print(f"median_box_h                     : {level_median_box_h:.2f}")
    print(f"level_band_threshold             : {level_band_threshold:.2f}")
    print(f"MAX_BOX_H_TO_MEDIAN_RATIO        : {MAX_BOX_H_TO_MEDIAN_RATIO}")
    print(f"KEEP_PER_LEVEL                   : {KEEP_PER_LEVEL}")
    print(f"Removed by level dedupe          : {len(removed_by_level_df)}")
    print(f"Final kept after level dedupe    : {len(final_kept_detections_df)}")

    if len(removed_by_level_df) > 0:
        print("\nRemoved by crossarm level dedupe:")
        _safe_display(
            removed_by_level_df[
                [
                    c for c in [
                        "orig_det_idx",
                        "score",
                        "cy",
                        "box_w",
                        "box_h",
                        "aspect_ratio",
                        "relative_width_to_max",
                        "level_band_id",
                        "level_rank_score",
                        "level_rank",
                        "keep_not_tall",
                        "x1", "y1", "x2", "y2",
                        "removal_reason",
                    ]
                    if c in removed_by_level_df.columns
                ]
            ]
        )

    if len(final_kept_detections_df) > 0:
        print("\nKept after crossarm level dedupe:")
        _safe_display(
            final_kept_detections_df[
                [
                    c for c in [
                        "orig_det_idx",
                        "score",
                        "cy",
                        "box_w",
                        "box_h",
                        "aspect_ratio",
                        "relative_width_to_max",
                        "level_band_id",
                        "level_rank_score",
                        "level_rank",
                        "x1", "y1", "x2", "y2",
                    ]
                    if c in final_kept_detections_df.columns
                ]
            ]
        )
else:
    print("\nCrossarm level dedupe filter skipped.")

# Save stage snapshot
stage_level_df = final_kept_detections_df.copy()

# -----------------------------------------------------------------------------
# 20. OPTIONAL PCA FILTER ON SUSPICIOUS REMAINING BOXES
# -----------------------------------------------------------------------------
removed_by_pca_df = final_kept_detections_df.iloc[0:0].copy()
pca_filter_applied = False
pca_num_suspicious_checked = 0

if CROSSARM_PCA_FILTER_ENABLED and len(final_kept_detections_df) > 0:
    pca_filter_applied = True

    tmp_df = final_kept_detections_df.copy().reset_index(drop=True)

    if "box_w" not in tmp_df.columns:
        tmp_df["box_w"] = (tmp_df["x2"] - tmp_df["x1"]).clip(lower=0.0)

    if "box_h" not in tmp_df.columns:
        tmp_df["box_h"] = (tmp_df["y2"] - tmp_df["y1"]).clip(lower=1.0)

    if "aspect_ratio" not in tmp_df.columns:
        tmp_df["aspect_ratio"] = tmp_df["box_w"] / tmp_df["box_h"]

    if "relative_width_to_max" not in tmp_df.columns:
        max_box_w = float(tmp_df["box_w"].max()) if len(tmp_df) > 0 else 0.0
        if max_box_w > 0:
            tmp_df["relative_width_to_max"] = tmp_df["box_w"] / max_box_w
        else:
            tmp_df["relative_width_to_max"] = 0.0

    pca_median_box_h = float(tmp_df["box_h"].median()) if len(tmp_df) > 0 else 0.0

    if pca_median_box_h > 0:
        suspicious_height_mask = (
            tmp_df["box_h"] >= (PCA_SUSPICIOUS_HEIGHT_TO_MEDIAN_MIN * pca_median_box_h)
        )
    else:
        suspicious_height_mask = pd.Series([False] * len(tmp_df), index=tmp_df.index)

    tmp_df["pca_suspicious"] = (
        (tmp_df["aspect_ratio"] <= PCA_SUSPICIOUS_ASPECT_MAX)
        | suspicious_height_mask
        | (tmp_df["relative_width_to_max"] <= PCA_SUSPICIOUS_REL_WIDTH_MAX)
    )

    tmp_df["pca_valid"] = False
    tmp_df["pca_num_pixels"] = np.nan
    tmp_df["pca_pc1_ratio"] = np.nan
    tmp_df["pca_pc2_ratio"] = np.nan
    tmp_df["pca_anisotropy"] = np.nan
    tmp_df["pca_perp_std"] = np.nan
    tmp_df["pca_check_skipped"] = False
    tmp_df["pca_keep_pc1"] = np.nan
    tmp_df["pca_keep_anisotropy"] = np.nan
    tmp_df["pca_keep"] = True

    pca_keep_mask = np.ones(len(tmp_df), dtype=bool)

    for idx, det_row in tmp_df.iterrows():
        if not bool(det_row["pca_suspicious"]):
            continue

        pca_num_suspicious_checked += 1

        orig_idx = int(det_row["orig_det_idx"])
        mask_i = crossarm_mask_lookup.get(orig_idx, None)

        if mask_i is None:
            tmp_df.loc[idx, "pca_check_skipped"] = True
            continue

        box_i = [det_row["x1"], det_row["y1"], det_row["x2"], det_row["y2"]]
        cropped_mask = crop_mask_to_box(mask_i, box_i)
        pca_stats = compute_binary_mask_pca_stats(cropped_mask)

        tmp_df.loc[idx, "pca_valid"] = bool(pca_stats["valid"])
        tmp_df.loc[idx, "pca_num_pixels"] = float(pca_stats["num_pixels"])
        tmp_df.loc[idx, "pca_pc1_ratio"] = pca_stats["pc1_ratio"]
        tmp_df.loc[idx, "pca_pc2_ratio"] = pca_stats["pc2_ratio"]
        tmp_df.loc[idx, "pca_anisotropy"] = pca_stats["anisotropy"]
        tmp_df.loc[idx, "pca_perp_std"] = pca_stats["perp_std"]

        if (not bool(pca_stats["valid"])) or (int(pca_stats["num_pixels"]) < PCA_MIN_MASK_PIXELS):
            tmp_df.loc[idx, "pca_check_skipped"] = True
            continue

        keep_pc1 = bool(pca_stats["pc1_ratio"] >= PCA_MIN_PC1_RATIO)
        keep_aniso = bool(pca_stats["anisotropy"] >= PCA_MIN_ANISOTROPY)
        keep_pca = bool(keep_pc1 and keep_aniso)

        tmp_df.loc[idx, "pca_keep_pc1"] = keep_pc1
        tmp_df.loc[idx, "pca_keep_anisotropy"] = keep_aniso
        tmp_df.loc[idx, "pca_keep"] = keep_pca

        if not keep_pca:
            pca_keep_mask[idx] = False

    removed_by_pca_df = tmp_df[~pca_keep_mask].copy().reset_index(drop=True)
    final_kept_detections_df = tmp_df[pca_keep_mask].copy().reset_index(drop=True)

    if len(removed_by_pca_df) > 0:
        removed_by_pca_df["removal_reason"] = "failed_optional_pca_filter"

    print(f"\nOptional PCA filter applied      : {pca_filter_applied}")
    print(f"PCA suspicious checked           : {pca_num_suspicious_checked}")
    print(f"PCA_MIN_MASK_PIXELS             : {PCA_MIN_MASK_PIXELS}")
    print(f"PCA_MIN_PC1_RATIO               : {PCA_MIN_PC1_RATIO}")
    print(f"PCA_MIN_ANISOTROPY              : {PCA_MIN_ANISOTROPY}")
    print(f"Removed by optional PCA         : {len(removed_by_pca_df)}")
    print(f"Final kept after optional PCA   : {len(final_kept_detections_df)}")

    if len(removed_by_pca_df) > 0:
        print("\nRemoved by optional PCA filter:")
        _safe_display(
            removed_by_pca_df[
                [
                    c for c in [
                        "orig_det_idx",
                        "score",
                        "aspect_ratio",
                        "box_w",
                        "box_h",
                        "relative_width_to_max",
                        "pca_suspicious",
                        "pca_valid",
                        "pca_num_pixels",
                        "pca_pc1_ratio",
                        "pca_anisotropy",
                        "pca_keep_pc1",
                        "pca_keep_anisotropy",
                        "pca_keep",
                        "x1", "y1", "x2", "y2",
                        "removal_reason",
                    ]
                    if c in removed_by_pca_df.columns
                ]
            ]
        )

    if len(final_kept_detections_df) > 0:
        print("\nKept after optional PCA filter:")
        _safe_display(
            final_kept_detections_df[
                [
                    c for c in [
                        "orig_det_idx",
                        "score",
                        "aspect_ratio",
                        "box_w",
                        "box_h",
                        "relative_width_to_max",
                        "pca_suspicious",
                        "pca_valid",
                        "pca_num_pixels",
                        "pca_pc1_ratio",
                        "pca_anisotropy",
                        "pca_keep",
                        "x1", "y1", "x2", "y2",
                    ]
                    if c in final_kept_detections_df.columns
                ]
            ]
        )
else:
    print("\nOptional PCA filter skipped.")

# Save stage snapshot
stage_pca_df = final_kept_detections_df.copy()

# -----------------------------------------------------------------------------
# 21. ASSIGN Xarm LABELS
# -----------------------------------------------------------------------------
if len(final_kept_detections_df) > 0:
    final_kept_detections_df = final_kept_detections_df.copy()

    if "cx" not in final_kept_detections_df.columns:
        final_kept_detections_df["cx"] = (
            final_kept_detections_df["x1"] + final_kept_detections_df["x2"]
        ) / 2.0

    if "cy" not in final_kept_detections_df.columns:
        final_kept_detections_df["cy"] = (
            final_kept_detections_df["y1"] + final_kept_detections_df["y2"]
        ) / 2.0

    final_kept_detections_df = final_kept_detections_df.sort_values(
        ["cy", "cx"], ascending=[True, True]
    ).reset_index(drop=True)

    final_kept_detections_df["xarm_label"] = [
        f"Xarm_{i+1}" for i in range(len(final_kept_detections_df))
    ]

    print("\nFinal kept detections with Xarm labels:")
    _safe_display(
        final_kept_detections_df[
            [
                c for c in [
                    "xarm_label",
                    "orig_det_idx",
                    "score",
                    "x1", "y1", "x2", "y2",
                    "cx", "cy",
                    "pole_overlap_fraction",
                    "box_w", "box_h", "aspect_ratio",
                    "relative_width_to_max",
                    "level_band_id",
                    "pca_suspicious",
                    "pca_pc1_ratio",
                    "pca_anisotropy",
                ]
                if c in final_kept_detections_df.columns
            ]
        ]
    )

# Save final stage snapshot
stage_final_df = final_kept_detections_df.copy()

# -----------------------------------------------------------------------------
# 22. FINAL GRID OF STAGES
# -----------------------------------------------------------------------------
if SHOW_STAGE_GRID:
    stage_dfs = [
        ("1. Raw", stage_raw_df),
        ("2. Containment", stage_containment_df),
        ("3. Main cluster", stage_cluster_df),
        ("4. Pole overlap", stage_pole_overlap_df),
        ("5. Structure", stage_structure_df),
        ("6. Level dedupe", stage_level_df),
        ("7. PCA", stage_pca_df),
        ("8. Final", stage_final_df),
    ]

    fig, axes = plt.subplots(2, 4, figsize=GRID_FIGSIZE)
    axes = axes.flatten()

    for ax, (stage_name, stage_df) in zip(axes, stage_dfs):
        plot_stage_on_ax(
            ax=ax,
            image=image,
            detections_df=stage_df,
            title=stage_name,
            projected_pole_mask=projected_pole_mask if projected_pole_mask_available else None,
            crossarm_mask_lookup=crossarm_mask_lookup,
            crossarm_mask_alpha=CROSSARM_MASK_ALPHA,
            pole_mask_alpha=POLE_MASK_ALPHA,
            label_bg=LABEL_BG,
            final_style=(stage_name == "8. Final"),
        )

    plt.tight_layout()
    plt.show()
    plt.close()

# -----------------------------------------------------------------------------
# 23. VISUALIZE FINAL KEPT DETECTIONS
# -----------------------------------------------------------------------------
plt.figure(figsize=(12, 9))
ax = plt.gca()

# Base ROI image
ax.imshow(image)

num_final_kept = len(final_kept_detections_df)

# First: coloured crossarm masks
if num_final_kept > 0:
    try:
        cmap = plt.colormaps.get_cmap("tab10").resampled(max(num_final_kept, 1))
    except Exception:
        cmap = plt.cm.get_cmap("tab10", max(num_final_kept, 1))

    for plot_idx, det_row in final_kept_detections_df.iterrows():
        orig_idx = int(det_row["orig_det_idx"])
        mask_i = crossarm_mask_lookup.get(orig_idx, None)

        if isinstance(mask_i, np.ndarray) and mask_i.ndim == 2 and mask_i.sum() > 0:
            color_rgba = cmap(plot_idx)

            overlay = np.zeros((mask_i.shape[0], mask_i.shape[1], 4), dtype=np.float32)
            overlay[..., 0] = color_rgba[0]
            overlay[..., 1] = color_rgba[1]
            overlay[..., 2] = color_rgba[2]
            overlay[..., 3] = mask_i.astype(np.float32) * CROSSARM_MASK_ALPHA
            ax.imshow(overlay)

# Second: red pole mask overlay + POLE label
if projected_pole_mask_available:
    pole_overlay = np.zeros(
        (projected_pole_mask.shape[0], projected_pole_mask.shape[1], 4),
        dtype=np.float32
    )
    pole_overlay[..., 0] = 1.0
    pole_overlay[..., 1] = 0.0
    pole_overlay[..., 2] = 0.0
    pole_overlay[..., 3] = projected_pole_mask.astype(np.float32) * POLE_MASK_ALPHA
    ax.imshow(pole_overlay)

    draw_pole_label(ax, projected_pole_mask)

# Third: yellow boxes + blue labels (no pole ov)
if num_final_kept > 0:
    for _, det_row in final_kept_detections_df.iterrows():
        x1, y1, x2, y2 = [float(det_row["x1"]), float(det_row["y1"]), float(det_row["x2"]), float(det_row["y2"])]
        score_i = float(det_row["score"])
        xarm_label = det_row["xarm_label"]

        rect = patches.Rectangle(
            (x1, y1),
            max(0.0, x2 - x1),
            max(0.0, y2 - y1),
            linewidth=2.0,
            edgecolor="yellow",
            facecolor="none",
        )
        ax.add_patch(rect)

        label_text = f"{xarm_label}: {score_i:.2f}"

        ax.text(
            x1,
            max(5, y1 - 5),
            label_text,
            color="white",
            fontsize=7,
            bbox=dict(facecolor=LABEL_BG, alpha=0.95, pad=0.4, edgecolor="none"),
        )

ax.set_title(
    f"{display_title}\n"
    f"source: {file_name} | Prompt: {PROMPT_TEXT} | "
    f"Raw: {num_detections} | Final kept: {num_final_kept} | "
    f"Valid crossarm masks: {len(crossarm_mask_lookup)}",
    fontsize=11,
    pad=12,
)
ax.axis("off")
plt.show()
plt.close()

# -----------------------------------------------------------------------------
# 24. BUILD ONE-ROW DEBUG RESULTS TABLE
# -----------------------------------------------------------------------------
crossarm_roi_debug_results_df = pd.DataFrame([{
    "crossarm_roi_debug_row_index": CROSSARM_ROI_DEBUG_ROW_INDEX,
    "image_id": image_id,
    "file_name": file_name,
    "roi_file_name": roi_file_name,
    "display_title": display_title,
    "roi_image_path": roi_image_path,
    "original_mode": original_mode,
    "roi_w": roi_w,
    "roi_h": roi_h,
    "pole_prompt": pole_prompt,
    "pole_det_idx": pole_det_idx,
    "pole_roi_geometry_available": bool(pole_roi_geometry_available),
    "projected_pole_mask_available": bool(projected_pole_mask_available),
    "pole_mask_filter_applied": bool(pole_mask_filter_applied),
    "pole_overlap_min_fraction": float(POLE_OVERLAP_MIN_FRACTION),
    "structure_filter_applied": bool(structure_filter_applied),
    "removed_by_structure_filter": int(len(removed_by_structure_df)),
    "pole_attach_margin_px": int(POLE_ATTACH_MARGIN_PX),
    "crossarm_min_aspect_ratio": float(CROSSARM_MIN_ASPECT_RATIO),
    "min_relative_width_to_max": float(MIN_RELATIVE_WIDTH_TO_MAX),
    "level_filter_applied": bool(level_filter_applied),
    "removed_by_level_filter": int(len(removed_by_level_df)),
    "crossarm_level_band_factor": float(CROSSARM_LEVEL_BAND_FACTOR),
    "max_box_h_to_median_ratio": float(MAX_BOX_H_TO_MEDIAN_RATIO),
    "keep_per_level": int(KEEP_PER_LEVEL),
    "pca_filter_applied": bool(pca_filter_applied),
    "removed_by_pca_filter": int(len(removed_by_pca_df)),
    "pca_num_suspicious_checked": int(pca_num_suspicious_checked),
    "pca_min_mask_pixels": int(PCA_MIN_MASK_PIXELS),
    "pca_min_pc1_ratio": float(PCA_MIN_PC1_RATIO),
    "pca_min_anisotropy": float(PCA_MIN_ANISOTROPY),
    "pca_suspicious_aspect_max": float(PCA_SUSPICIOUS_ASPECT_MAX),
    "pca_suspicious_height_to_median_min": float(PCA_SUSPICIOUS_HEIGHT_TO_MEDIAN_MIN),
    "pca_suspicious_rel_width_max": float(PCA_SUSPICIOUS_REL_WIDTH_MAX),
    "raw_num_detections": int(num_detections),
    "valid_crossarm_masks": int(len(crossarm_mask_lookup)),
    "has_any_valid_masks": bool(HAS_ANY_VALID_MASKS),
    "kept_after_containment": int(len(kept_after_containment_df)),
    "removed_by_containment": int(len(removed_by_containment_df)),
    "kept_after_cluster": int(len(kept_after_cluster_df)),
    "removed_by_cluster": int(len(removed_by_cluster_df)),
    "removed_by_pole_mask": int(len(removed_by_pole_mask_df)),
    "removed_by_structure": int(len(removed_by_structure_df)),
    "removed_by_level": int(len(removed_by_level_df)),
    "removed_by_pca": int(len(removed_by_pca_df)),
    "final_kept": int(len(final_kept_detections_df)),
    "cluster_threshold_used": float(cluster_threshold_used),
    "level_median_box_h": float(level_median_box_h),
    "level_band_threshold": float(level_band_threshold),
    "xarm_labels": ", ".join(
        final_kept_detections_df["xarm_label"].tolist()
    ) if len(final_kept_detections_df) > 0 else "",
    "run_status": "ok",
    "error_message": "",
}])

# -----------------------------------------------------------------------------
# 25. CLEANUP
# -----------------------------------------------------------------------------
del image

# -----------------------------------------------------------------------------
# 26. FINAL CONFIRMATION
# -----------------------------------------------------------------------------
print("\nCELL 16A completed successfully.")
print("Helper functions exposed as globals for CELL 16B:")
print("  - normalize_masks")
print("  - normalize_boxes")
print("  - normalize_scores")
print("  - suppress_contained_shorter_detections")
print("  - keep_main_detection_cluster")
print("  - project_pole_mask_to_roi")
print("  - compute_box_overlap_with_mask")
print("  - crop_mask_to_box")
print("  - compute_binary_mask_pca_stats")
print("  - plot_stage_on_ax")
print("Saved outputs:")
print("  - crossarm_roi_debug_results_df")
print("  - crossarm_mask_lookup")
print("  - stage_raw_df")
print("  - stage_containment_df")
print("  - stage_cluster_df")
print("  - stage_pole_overlap_df")
print("  - stage_structure_df")
print("  - stage_level_df")
print("  - stage_pca_df")
print("  - stage_final_df")
