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