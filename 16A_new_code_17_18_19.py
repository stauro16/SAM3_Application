# =============================================================================
# Fixed_1
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

CROSSARM_ROI_DEBUG_ROW_INDEX = 2
#PROMPT_TEXT = "utility pole crossarm"
PROMPT_TEXT = "utility pole crossarm"

# keep threshold at 0.3 throughout
TEXT_THRESHOLD = 0.3
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
# 17. POLE MASK OVERLAP FILTER + TOP-OF-POLE RESCUE + POLE-DOMINATED REJECT
# -----------------------------------------------------------------------------
# EXPLANATION:
# This step keeps real crossarms that are attached to the pole, while removing
# detections that are likely the pole itself.
#
# COMBINED RULE:
# Keep a detection if:
#   1) its pole overlap is inside the valid band:
#          POLE_OVERLAP_MIN_FRACTION <= overlap <= POLE_OVERLAP_MAX_FRACTION
#
#      OR
#
#   2) it passes the top-of-pole rescue:
#          touches pole corridor
#          touches top-of-pole band
#          is wide enough relative to other candidates
#
#   AND
#
#   3) it is NOT pole-dominated:
#          overlap <= POLE_OVERLAP_MAX_FRACTION
#
# WHY:
# - The lower bound removes detections far away from the pole.
# - The top rescue keeps genuine top-mounted crossarms with low measured overlap.
# - The upper bound removes boxes that are mostly the pole itself.
# -----------------------------------------------------------------------------

removed_by_pole_mask_df = kept_after_cluster_df.iloc[0:0].copy()

# Local rescue-band settings for this debug cell.
TOP_BAND_ABOVE = 80
TOP_BAND_BELOW = 250

# Restore the old working upper-bound rule.
# This rejects boxes that are mostly pole, e.g. pole trunk detected as crossarm.
POLE_OVERLAP_MAX_FRACTION = float(
    globals().get("POLE_OVERLAP_MAX_FRACTION", 0.30)
)

if len(kept_after_cluster_df) == 0:
    final_kept_detections_df = kept_after_cluster_df.copy()
    removed_by_pole_mask_df = kept_after_cluster_df.copy()

elif (
    POLE_MASK_FILTER_ENABLED
    and projected_pole_mask_available
):
    pole_mask_filter_applied = True

    tmp_df = kept_after_cluster_df.copy().reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 1. Standard pole-overlap fraction
    # -------------------------------------------------------------------------
    overlap_fracs = []

    for _, det_row in tmp_df.iterrows():
        box_i = [det_row["x1"], det_row["y1"], det_row["x2"], det_row["y2"]]
        frac = compute_box_overlap_with_mask(box_i, projected_pole_mask)
        overlap_fracs.append(float(frac))

    tmp_df["pole_overlap_fraction"] = overlap_fracs

    # -------------------------------------------------------------------------
    # 2. Pole-overlap band
    # -------------------------------------------------------------------------
    # Lower bound:
    #   Must touch the pole at least a little.
    #
    # Upper bound:
    #   Must NOT be mostly pole.
    # -------------------------------------------------------------------------
    tmp_df["pole_overlap_touches_min"] = (
        tmp_df["pole_overlap_fraction"] >= POLE_OVERLAP_MIN_FRACTION
    )

    tmp_df["pole_overlap_under_max"] = (
        tmp_df["pole_overlap_fraction"] <= POLE_OVERLAP_MAX_FRACTION
    )

    tmp_df["pole_dominated_reject"] = (
        tmp_df["pole_overlap_fraction"] > POLE_OVERLAP_MAX_FRACTION
    )

    tmp_df["kept_by_pole_overlap"] = (
        tmp_df["pole_overlap_touches_min"]
        & tmp_df["pole_overlap_under_max"]
    )

    # -------------------------------------------------------------------------
    # 3. Default debug columns
    # -------------------------------------------------------------------------
    tmp_df["touches_attach_corridor"] = False
    tmp_df["touches_top_band"] = False
    tmp_df["top_attach_rescue_candidate"] = False
    tmp_df["rescued_by_top_attach"] = False

    tmp_df["box_w"] = (tmp_df["x2"] - tmp_df["x1"]).clip(lower=0.0)

    max_box_w = float(tmp_df["box_w"].max()) if len(tmp_df) > 0 else 0.0
    if max_box_w > 0:
        tmp_df["relative_width_to_max"] = tmp_df["box_w"] / max_box_w
    else:
        tmp_df["relative_width_to_max"] = 0.0

    # -------------------------------------------------------------------------
    # 4. Pole corridor + top-of-pole rescue
    # -------------------------------------------------------------------------
    pole_cols = np.where(projected_pole_mask.any(axis=0))[0]
    pole_rows = np.where(projected_pole_mask.any(axis=1))[0]

    if len(pole_cols) > 0 and len(pole_rows) > 0:
        pole_x1 = int(pole_cols.min())
        pole_x2 = int(pole_cols.max())
        pole_top_y = int(pole_rows.min())

        attach_x1 = max(0, pole_x1 - int(POLE_ATTACH_MARGIN_PX))
        attach_x2 = min(roi_w - 1, pole_x2 + int(POLE_ATTACH_MARGIN_PX))

        top_band_y1 = max(0, pole_top_y - int(TOP_BAND_ABOVE))
        top_band_y2 = min(roi_h - 1, pole_top_y + int(TOP_BAND_BELOW))

        tmp_df["touches_attach_corridor"] = (
            (tmp_df["x2"] >= attach_x1)
            & (tmp_df["x1"] <= attach_x2)
        )

        tmp_df["touches_top_band"] = (
            (tmp_df["y2"] >= top_band_y1)
            & (tmp_df["y1"] <= top_band_y2)
        )

        tmp_df["top_attach_rescue_candidate"] = (
            tmp_df["touches_attach_corridor"]
            & tmp_df["touches_top_band"]
            & (tmp_df["relative_width_to_max"] >= MIN_RELATIVE_WIDTH_TO_MAX)
        )

        # Actual rescue only applies to boxes that:
        # - were not already kept by normal overlap band
        # - are not pole-dominated
        tmp_df["rescued_by_top_attach"] = (
            (~tmp_df["kept_by_pole_overlap"])
            & tmp_df["top_attach_rescue_candidate"]
            & (~tmp_df["pole_dominated_reject"])
        )

        print("\nPole-overlap rescue geometry:")
        print(f"  pole_x_range                   : [{pole_x1}, {pole_x2}]")
        print(f"  attach_corridor_x_range        : [{attach_x1}, {attach_x2}]")
        print(f"  pole_top_y                     : {pole_top_y}")
        print(f"  top_band_y_range               : [{top_band_y1}, {top_band_y2}]")
        print(f"  TOP_BAND_ABOVE                 : {TOP_BAND_ABOVE}")
        print(f"  TOP_BAND_BELOW                 : {TOP_BAND_BELOW}")

    # -------------------------------------------------------------------------
    # 5. Final combined keep rule
    # -------------------------------------------------------------------------
    # Important:
    # Even rescued detections must satisfy the upper-bound guard.
    # This prevents pole-shaped / pole-dominated detections from being rescued.
    # -------------------------------------------------------------------------
    keep_flags_arr = (
        (
            tmp_df["kept_by_pole_overlap"]
            | tmp_df["rescued_by_top_attach"]
        )
        & (~tmp_df["pole_dominated_reject"])
    ).to_numpy(dtype=bool)

    final_kept_detections_df = tmp_df[keep_flags_arr].copy().reset_index(drop=True)
    removed_by_pole_mask_df = tmp_df[~keep_flags_arr].copy().reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 6. Add removal reason for debugging
    # -------------------------------------------------------------------------
    if len(removed_by_pole_mask_df) > 0:
        removal_reasons = []

        for _, det_row in removed_by_pole_mask_df.iterrows():
            if bool(det_row.get("pole_dominated_reject", False)):
                removal_reasons.append(
                    f"pole_overlap_gt_max_{POLE_OVERLAP_MAX_FRACTION:.3f}"
                )
            elif not bool(det_row.get("pole_overlap_touches_min", False)):
                removal_reasons.append(
                    "failed_low_overlap_and_not_top_rescued"
                )
            else:
                removal_reasons.append(
                    "failed_pole_overlap_band_or_top_attach_rescue"
                )

        removed_by_pole_mask_df["removal_reason"] = removal_reasons

else:
    # -------------------------------------------------------------------------
    # No projected pole mask available:
    # Keep all detections, but create consistent debug columns.
    # -------------------------------------------------------------------------
    final_kept_detections_df = kept_after_cluster_df.copy()
    removed_by_pole_mask_df = kept_after_cluster_df.iloc[0:0].copy()

    final_kept_detections_df["pole_overlap_fraction"] = np.nan
    final_kept_detections_df["pole_overlap_touches_min"] = False
    final_kept_detections_df["pole_overlap_under_max"] = False
    final_kept_detections_df["pole_dominated_reject"] = False
    final_kept_detections_df["kept_by_pole_overlap"] = False

    final_kept_detections_df["touches_attach_corridor"] = False
    final_kept_detections_df["touches_top_band"] = False
    final_kept_detections_df["top_attach_rescue_candidate"] = False
    final_kept_detections_df["rescued_by_top_attach"] = False

    final_kept_detections_df["box_w"] = (
        final_kept_detections_df["x2"] - final_kept_detections_df["x1"]
    ).clip(lower=0.0)

    max_box_w = (
        float(final_kept_detections_df["box_w"].max())
        if len(final_kept_detections_df) > 0
        else 0.0
    )

    if max_box_w > 0:
        final_kept_detections_df["relative_width_to_max"] = (
            final_kept_detections_df["box_w"] / max_box_w
        )
    else:
        final_kept_detections_df["relative_width_to_max"] = 0.0

# -----------------------------------------------------------------------------
# 7. PRINT DEBUG SUMMARY
# -----------------------------------------------------------------------------
print(f"\nPole mask filter applied        : {pole_mask_filter_applied}")
print(f"POLE_OVERLAP_MIN_FRACTION       : {POLE_OVERLAP_MIN_FRACTION}")
print(f"POLE_OVERLAP_MAX_FRACTION       : {POLE_OVERLAP_MAX_FRACTION}")
print(f"Final kept detections           : {len(final_kept_detections_df)}")
print(f"Removed by pole overlap filter  : {len(removed_by_pole_mask_df)}")

if len(final_kept_detections_df) > 0:
    print("\nKept after pole overlap band + top-attach rescue:")
    _safe_display(
        final_kept_detections_df[
            [
                c for c in [
                    "orig_det_idx",
                    "score",
                    "pole_overlap_fraction",
                    "pole_overlap_touches_min",
                    "pole_overlap_under_max",
                    "pole_dominated_reject",
                    "kept_by_pole_overlap",
                    "touches_attach_corridor",
                    "touches_top_band",
                    "top_attach_rescue_candidate",
                    "rescued_by_top_attach",
                    "box_w",
                    "relative_width_to_max",
                    "x1", "y1", "x2", "y2",
                ]
                if c in final_kept_detections_df.columns
            ]
        ]
    )

if len(removed_by_pole_mask_df) > 0:
    print("\nRemoved by pole overlap band / rescue filter:")
    _safe_display(
        removed_by_pole_mask_df[
            [
                c for c in [
                    "orig_det_idx",
                    "score",
                    "pole_overlap_fraction",
                    "pole_overlap_touches_min",
                    "pole_overlap_under_max",
                    "pole_dominated_reject",
                    "kept_by_pole_overlap",
                    "touches_attach_corridor",
                    "touches_top_band",
                    "top_attach_rescue_candidate",
                    "rescued_by_top_attach",
                    "box_w",
                    "relative_width_to_max",
                    "x1", "y1", "x2", "y2",
                    "removal_reason",
                ]
                if c in removed_by_pole_mask_df.columns
            ]
        ]
    )

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
# 18A. CROSSARM CONTINUITY MERGE — ANGLE + LINE-THROUGH-POLE GROUPING
# -----------------------------------------------------------------------------
# EXPLANATION:
# Some physical crossarms are detected as two separate pieces when they cross
# behind or through the pole. This step groups those split pieces before level
# dedupe so they are treated as one physical crossarm.
#
# WHY THIS STEP EXISTS:
# The current filters understand:
#   - pole overlap
#   - pole corridor contact
#   - width/height checks
#   - level bands
#
# But they do not yet understand:
#   "these two boxes are on the same diagonal line and belong to one arm."
#
# WHAT THIS STEP DOES:
#   1) estimates each detection's dominant direction from its full ROI mask
#   2) checks if two detections have similar direction
#   3) checks if their implied lines pass near the pole
#   4) checks if they lie on a similar continuous line
#   5) checks stricter split-across-pole / far-from-pole evidence
#   6) groups same-arm pieces into one representative detection
#
# IMPORTANT:
# - This does not mutate the original crossarm_mask_lookup in place.
# - It creates crossarm_mask_lookup_original and crossarm_grouped_mask_lookup.
# - It then points crossarm_mask_lookup to the grouped lookup for downstream
#   Step 20 / final display compatibility.
# -----------------------------------------------------------------------------

removed_by_continuity_merge_df = final_kept_detections_df.iloc[0:0].copy()
continuity_pair_debug_df = pd.DataFrame()
continuity_merge_applied = False

# -----------------------------------------------------------------------------
# 18A.1 CONFIG
# -----------------------------------------------------------------------------
CONTINUITY_MERGE_ENABLED = True

# Angle similarity between two arm pieces.
CONTINUITY_ANGLE_DIFF_MAX_DEG = 15.0

# Distance from each inferred arm line to the pole centre.
CONTINUITY_POLE_LINE_DIST_MAX_PX = 90.0

# Distance between the two inferred arm centre-lines.
CONTINUITY_PAIR_LINE_DIST_MAX_PX = 80.0

# Scale compatibility checks.
CONTINUITY_MIN_WIDTH_RATIO = 0.35
CONTINUITY_MIN_HEIGHT_RATIO = 0.30

# More permissive than the later PCA reject filter.
# This is a grouping step, not a false-positive rejection step.
CONTINUITY_MIN_MASK_PIXELS = 80
CONTINUITY_MERGE_MIN_PC1_RATIO = 0.70
CONTINUITY_MERGE_MIN_ANISOTROPY = 2.50

# Split-across-pole evidence.
CONTINUITY_SIDE_MARGIN_PX = 120

# -----------------------------------------------------------------------------
# 18A.2 HELPERS
# -----------------------------------------------------------------------------
def _angle_diff_deg_180(a_deg, b_deg):
    """
    Smallest angle difference under 180-degree symmetry.
    """
    if pd.isna(a_deg) or pd.isna(b_deg):
        return np.nan

    diff = abs(float(a_deg) - float(b_deg)) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff

    return float(diff)


def _mean_angle_deg_180(angle_a_deg, angle_b_deg):
    """
    Mean angle under 180-degree symmetry.

    EXPLANATION:
    A line at 5 degrees and a line at 175 degrees are almost the same line.
    A normal arithmetic mean would incorrectly return 90 degrees. The double-
    angle method avoids that problem.
    """
    theta_a = math.radians(float(angle_a_deg) * 2.0)
    theta_b = math.radians(float(angle_b_deg) * 2.0)

    mean_x = math.cos(theta_a) + math.cos(theta_b)
    mean_y = math.sin(theta_a) + math.sin(theta_b)

    if abs(mean_x) < 1e-8 and abs(mean_y) < 1e-8:
        return float(angle_a_deg) % 180.0

    mean_angle = (math.degrees(math.atan2(mean_y, mean_x)) / 2.0) % 180.0
    return float(mean_angle)


def _compute_mask_principal_angle_global(mask_bool):
    """
    Compute dominant mask direction using full ROI/global mask coordinates.

    EXPLANATION:
    This avoids ambiguity from using box-local crop coordinates. Since the mask
    already lives in ROI coordinates, PCA is computed directly over the true
    mask pixel positions.
    """
    out = {
        "valid": False,
        "num_pixels": 0,
        "angle_deg": np.nan,
        "pc1_ratio": np.nan,
        "anisotropy": np.nan,
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

    if num_pixels < CONTINUITY_MIN_MASK_PIXELS:
        return out

    coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    coords = coords - coords.mean(axis=0, keepdims=True)

    try:
        cov = np.cov(coords, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
    except Exception:
        return out

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if len(eigvals) < 2:
        return out

    eigvals = np.maximum(eigvals, 0.0)
    total_var = float(eigvals.sum())

    if total_var <= 0:
        return out

    pc1 = float(eigvals[0])
    pc2 = float(eigvals[1])
    pc1_ratio = pc1 / total_var
    anisotropy = pc1 / max(pc2, 1e-8)

    principal_vec = eigvecs[:, 0]
    angle_deg = float(
        np.degrees(np.arctan2(principal_vec[1], principal_vec[0])) % 180.0
    )

    out.update({
        "valid": True,
        "angle_deg": angle_deg,
        "pc1_ratio": pc1_ratio,
        "anisotropy": anisotropy,
    })

    return out


def _point_to_line_distance(px, py, x0, y0, angle_deg):
    """
    Perpendicular distance from a point to an infinite line.
    """
    theta = math.radians(float(angle_deg))
    ux = math.cos(theta)
    uy = math.sin(theta)

    dx = float(px) - float(x0)
    dy = float(py) - float(y0)

    return abs((-uy * dx) + (ux * dy))


def _project_onto_line(px, py, x0, y0, angle_deg):
    """
    Scalar projection of a point onto a line.
    """
    theta = math.radians(float(angle_deg))
    ux = math.cos(theta)
    uy = math.sin(theta)

    dx = float(px) - float(x0)
    dy = float(py) - float(y0)

    return float((ux * dx) + (uy * dy))


def _find_parent(parent, x):
    """
    Union-find parent lookup with path compression.
    """
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union_parent(parent, a, b):
    """
    Union two union-find components.
    """
    root_a = _find_parent(parent, a)
    root_b = _find_parent(parent, b)

    if root_a != root_b:
        parent[root_b] = root_a


# -----------------------------------------------------------------------------
# 18A.3 APPLY CONTINUITY MERGE
# -----------------------------------------------------------------------------
if (
    CONTINUITY_MERGE_ENABLED
    and len(final_kept_detections_df) >= 2
    and projected_pole_mask_available
):
    continuity_merge_applied = True

    tmp_df = final_kept_detections_df.copy().reset_index(drop=True)

    # Preserve original masks before creating grouped masks.
    crossarm_mask_lookup_original = dict(crossarm_mask_lookup)
    crossarm_grouped_mask_lookup = dict(crossarm_mask_lookup_original)

    # -------------------------------------------------------------------------
    # 1. Ensure geometry columns exist
    # -------------------------------------------------------------------------
    if "cx" not in tmp_df.columns:
        tmp_df["cx"] = (tmp_df["x1"] + tmp_df["x2"]) / 2.0

    if "cy" not in tmp_df.columns:
        tmp_df["cy"] = (tmp_df["y1"] + tmp_df["y2"]) / 2.0

    if "box_w" not in tmp_df.columns:
        tmp_df["box_w"] = (tmp_df["x2"] - tmp_df["x1"]).clip(lower=0.0)

    if "box_h" not in tmp_df.columns:
        tmp_df["box_h"] = (tmp_df["y2"] - tmp_df["y1"]).clip(lower=1.0)

    if "aspect_ratio" not in tmp_df.columns:
        tmp_df["aspect_ratio"] = tmp_df["box_w"] / tmp_df["box_h"]

    # -------------------------------------------------------------------------
    # 2. Pole geometry from projected pole mask
    # -------------------------------------------------------------------------
    pole_ys, pole_xs = np.where(projected_pole_mask)

    pole_x1 = int(pole_xs.min())
    pole_x2 = int(pole_xs.max())
    pole_y1 = int(pole_ys.min())
    pole_y2 = int(pole_ys.max())

    pole_cx = float((pole_x1 + pole_x2) / 2.0)
    pole_cy = float((pole_y1 + pole_y2) / 2.0)
    pole_width_px = float(max(1, pole_x2 - pole_x1))

    # -------------------------------------------------------------------------
    # 3. Per-detection global-mask direction features
    # -------------------------------------------------------------------------
    tmp_df["continuity_mask_valid"] = False
    tmp_df["continuity_num_pixels"] = 0
    tmp_df["continuity_angle_deg"] = np.nan
    tmp_df["continuity_pc1_ratio"] = np.nan
    tmp_df["continuity_anisotropy"] = np.nan
    tmp_df["continuity_pole_line_dist"] = np.nan
    tmp_df["continuity_side_from_pole"] = np.nan
    tmp_df["continuity_can_merge"] = False

    for idx, det_row in tmp_df.iterrows():
        orig_idx = int(det_row["orig_det_idx"])
        mask_i = crossarm_mask_lookup_original.get(orig_idx, None)

        angle_stats = _compute_mask_principal_angle_global(mask_i)

        tmp_df.loc[idx, "continuity_mask_valid"] = bool(angle_stats["valid"])
        tmp_df.loc[idx, "continuity_num_pixels"] = int(angle_stats["num_pixels"])
        tmp_df.loc[idx, "continuity_angle_deg"] = angle_stats["angle_deg"]
        tmp_df.loc[idx, "continuity_pc1_ratio"] = angle_stats["pc1_ratio"]
        tmp_df.loc[idx, "continuity_anisotropy"] = angle_stats["anisotropy"]
        tmp_df.loc[idx, "continuity_side_from_pole"] = float(det_row["cx"]) - pole_cx

        if bool(angle_stats["valid"]):
            line_dist_to_pole = _point_to_line_distance(
                px=pole_cx,
                py=pole_cy,
                x0=float(det_row["cx"]),
                y0=float(det_row["cy"]),
                angle_deg=float(angle_stats["angle_deg"]),
            )

            tmp_df.loc[idx, "continuity_pole_line_dist"] = line_dist_to_pole

            tmp_df.loc[idx, "continuity_can_merge"] = bool(
                (line_dist_to_pole <= CONTINUITY_POLE_LINE_DIST_MAX_PX)
                and (float(angle_stats["pc1_ratio"]) >= CONTINUITY_MERGE_MIN_PC1_RATIO)
                and (float(angle_stats["anisotropy"]) >= CONTINUITY_MERGE_MIN_ANISOTROPY)
            )

    # -------------------------------------------------------------------------
    # 4. Pairwise continuity test + union-find grouping
    # -------------------------------------------------------------------------
    n = len(tmp_df)
    parent = list(range(n))
    pair_debug_rows = []

    for i in range(n):
        row_i = tmp_df.iloc[i]

        if not bool(row_i["continuity_can_merge"]):
            continue

        for j in range(i + 1, n):
            row_j = tmp_df.iloc[j]

            if not bool(row_j["continuity_can_merge"]):
                continue

            angle_i = float(row_i["continuity_angle_deg"])
            angle_j = float(row_j["continuity_angle_deg"])

            angle_diff = _angle_diff_deg_180(angle_i, angle_j)
            mean_angle = _mean_angle_deg_180(angle_i, angle_j)

            width_ratio = (
                min(float(row_i["box_w"]), float(row_j["box_w"])) /
                max(float(row_i["box_w"]), float(row_j["box_w"]), 1.0)
            )

            height_ratio = (
                min(float(row_i["box_h"]), float(row_j["box_h"])) /
                max(float(row_i["box_h"]), float(row_j["box_h"]), 1.0)
            )

            pair_line_dist_ij = _point_to_line_distance(
                px=float(row_j["cx"]),
                py=float(row_j["cy"]),
                x0=float(row_i["cx"]),
                y0=float(row_i["cy"]),
                angle_deg=angle_i,
            )

            pair_line_dist_ji = _point_to_line_distance(
                px=float(row_i["cx"]),
                py=float(row_i["cy"]),
                x0=float(row_j["cx"]),
                y0=float(row_j["cy"]),
                angle_deg=angle_j,
            )

            proj_i = _project_onto_line(
                px=float(row_i["cx"]),
                py=float(row_i["cy"]),
                x0=pole_cx,
                y0=pole_cy,
                angle_deg=mean_angle,
            )

            proj_j = _project_onto_line(
                px=float(row_j["cx"]),
                py=float(row_j["cy"]),
                x0=pole_cx,
                y0=pole_cy,
                angle_deg=mean_angle,
            )

            along_gap = abs(proj_i - proj_j)
            along_gap_max = max(
                2.0 * pole_width_px,
                1.5 * max(float(row_i["box_w"]), float(row_j["box_w"]), 1.0),
            )

            # -----------------------------------------------------------------
            # Stricter side-of-pole evidence
            # -----------------------------------------------------------------
            side_i = float(row_i["continuity_side_from_pole"])
            side_j = float(row_j["continuity_side_from_pole"])

            opposite_sides_strict = bool(
                (side_i * side_j < 0.0)
                and (abs(side_i) >= 0.25 * pole_width_px)
                and (abs(side_j) >= 0.25 * pole_width_px)
            )

            both_far_from_pole = bool(
                (abs(side_i) >= 0.5 * pole_width_px)
                and (abs(side_j) >= 0.5 * pole_width_px)
            )

            passes_side_check = bool(
                opposite_sides_strict
                or both_far_from_pole
            )

            same_arm_pair = bool(
                (angle_diff <= CONTINUITY_ANGLE_DIFF_MAX_DEG)
                and (pair_line_dist_ij <= CONTINUITY_PAIR_LINE_DIST_MAX_PX)
                and (pair_line_dist_ji <= CONTINUITY_PAIR_LINE_DIST_MAX_PX)
                and (float(row_i["continuity_pole_line_dist"]) <= CONTINUITY_POLE_LINE_DIST_MAX_PX)
                and (float(row_j["continuity_pole_line_dist"]) <= CONTINUITY_POLE_LINE_DIST_MAX_PX)
                and (width_ratio >= CONTINUITY_MIN_WIDTH_RATIO)
                and (height_ratio >= CONTINUITY_MIN_HEIGHT_RATIO)
                and (along_gap <= along_gap_max)
                and passes_side_check
            )

            pair_debug_rows.append({
                "orig_det_idx_i": int(row_i["orig_det_idx"]),
                "orig_det_idx_j": int(row_j["orig_det_idx"]),
                "angle_i_deg": angle_i,
                "angle_j_deg": angle_j,
                "mean_angle_deg": mean_angle,
                "angle_diff_deg": angle_diff,
                "pair_line_dist_ij": pair_line_dist_ij,
                "pair_line_dist_ji": pair_line_dist_ji,
                "pole_line_dist_i": float(row_i["continuity_pole_line_dist"]),
                "pole_line_dist_j": float(row_j["continuity_pole_line_dist"]),
                "width_ratio": width_ratio,
                "height_ratio": height_ratio,
                "along_gap": along_gap,
                "along_gap_max": along_gap_max,
                "side_i": side_i,
                "side_j": side_j,
                "opposite_sides_strict": opposite_sides_strict,
                "both_far_from_pole": both_far_from_pole,
                "passes_side_check": passes_side_check,
                "same_arm_pair": same_arm_pair,
            })

            if same_arm_pair:
                _union_parent(parent, i, j)

    continuity_pair_debug_df = pd.DataFrame(pair_debug_rows)

    # -------------------------------------------------------------------------
    # 5. Build grouped representative detections
    # -------------------------------------------------------------------------
    group_map = {}
    for idx in range(n):
        root = _find_parent(parent, idx)
        group_map.setdefault(root, []).append(idx)

    grouped_rows = []
    grouped_away_rows = []
    next_group_id = 1

    for _, member_indices in group_map.items():
        member_df = tmp_df.iloc[member_indices].copy().reset_index(drop=True)

        member_orig_ids = [int(v) for v in member_df["orig_det_idx"].tolist()]
        member_scores = [float(v) for v in member_df["score"].tolist()]

        rep_pos = int(member_df["score"].astype(float).idxmax())
        rep_row = member_df.iloc[rep_pos].copy()
        rep_orig_idx = int(rep_row["orig_det_idx"])

        union_x1 = float(member_df["x1"].min())
        union_y1 = float(member_df["y1"].min())
        union_x2 = float(member_df["x2"].max())
        union_y2 = float(member_df["y2"].max())

        rep_row["x1"] = union_x1
        rep_row["y1"] = union_y1
        rep_row["x2"] = union_x2
        rep_row["y2"] = union_y2
        rep_row["cx"] = (union_x1 + union_x2) / 2.0
        rep_row["cy"] = (union_y1 + union_y2) / 2.0
        rep_row["box_w"] = max(0.0, union_x2 - union_x1)
        rep_row["box_h"] = max(1.0, union_y2 - union_y1)
        rep_row["aspect_ratio"] = rep_row["box_w"] / rep_row["box_h"]

        rep_row["continuity_group_id"] = int(next_group_id)
        rep_row["continuity_group_size"] = int(len(member_df))
        rep_row["continuity_member_orig_det_idxs"] = ",".join(
            [str(v) for v in member_orig_ids]
        )
        rep_row["continuity_member_scores"] = ",".join(
            [f"{float(v):.3f}" for v in member_scores]
        )
        rep_row["continuity_merged"] = bool(len(member_df) > 1)

        union_mask = np.zeros((roi_h, roi_w), dtype=bool)
        any_union_mask = False

        for orig_idx_member in member_orig_ids:
            mask_member = crossarm_mask_lookup_original.get(int(orig_idx_member), None)

            if (
                isinstance(mask_member, np.ndarray)
                and mask_member.ndim == 2
                and mask_member.shape == (roi_h, roi_w)
            ):
                union_mask |= mask_member.astype(bool)
                any_union_mask = True

        if any_union_mask:
            crossarm_grouped_mask_lookup[rep_orig_idx] = union_mask

        grouped_rows.append(rep_row)

        if len(member_df) > 1:
            for _, row_member in member_df.iterrows():
                if int(row_member["orig_det_idx"]) == rep_orig_idx:
                    continue

                removed_row = row_member.copy()
                removed_row["continuity_group_id"] = int(next_group_id)
                removed_row["continuity_group_size"] = int(len(member_df))
                removed_row["continuity_representative_orig_det_idx"] = int(rep_orig_idx)
                removed_row["removal_reason"] = (
                    f"merged_into_continuity_group_{int(next_group_id)}"
                )
                grouped_away_rows.append(removed_row)

        next_group_id += 1

    if len(grouped_rows) > 0:
        final_kept_detections_df = pd.DataFrame(grouped_rows).reset_index(drop=True)
    else:
        final_kept_detections_df = tmp_df.copy().reset_index(drop=True)

    if len(grouped_away_rows) > 0:
        removed_by_continuity_merge_df = pd.DataFrame(grouped_away_rows).reset_index(drop=True)
    else:
        removed_by_continuity_merge_df = final_kept_detections_df.iloc[0:0].copy()

    # -------------------------------------------------------------------------
    # 6. Recompute geometry after grouping
    # -------------------------------------------------------------------------
    if len(final_kept_detections_df) > 0:
        final_kept_detections_df["cx"] = (
            final_kept_detections_df["x1"] + final_kept_detections_df["x2"]
        ) / 2.0

        final_kept_detections_df["cy"] = (
            final_kept_detections_df["y1"] + final_kept_detections_df["y2"]
        ) / 2.0

        final_kept_detections_df["box_w"] = (
            final_kept_detections_df["x2"] - final_kept_detections_df["x1"]
        ).clip(lower=0.0)

        final_kept_detections_df["box_h"] = (
            final_kept_detections_df["y2"] - final_kept_detections_df["y1"]
        ).clip(lower=1.0)

        final_kept_detections_df["aspect_ratio"] = (
            final_kept_detections_df["box_w"] / final_kept_detections_df["box_h"]
        )

        max_box_w_grouped = float(final_kept_detections_df["box_w"].max())
        if max_box_w_grouped > 0:
            final_kept_detections_df["relative_width_to_max"] = (
                final_kept_detections_df["box_w"] / max_box_w_grouped
            )
        else:
            final_kept_detections_df["relative_width_to_max"] = 0.0

    # -------------------------------------------------------------------------
    # 7. Clear stale PCA columns before downstream Step 20 recomputes them
    # -------------------------------------------------------------------------
    stale_pca_cols = [
        c for c in final_kept_detections_df.columns
        if str(c).startswith("pca_")
    ]

    if len(stale_pca_cols) > 0:
        final_kept_detections_df = final_kept_detections_df.drop(
            columns=stale_pca_cols,
            errors="ignore",
        )

    # Use grouped masks downstream without mutating the original lookup in place.
    crossarm_mask_lookup = crossarm_grouped_mask_lookup

    print(f"\nCrossarm continuity merge applied : {continuity_merge_applied}")
    print(f"Pole center used for merge        : ({pole_cx:.1f}, {pole_cy:.1f})")
    print(f"Pole x-range                      : [{pole_x1}, {pole_x2}]")
    print(f"Pole width px                     : {pole_width_px:.1f}")
    print(f"ANGLE_DIFF_MAX_DEG                : {CONTINUITY_ANGLE_DIFF_MAX_DEG}")
    print(f"POLE_LINE_DIST_MAX_PX             : {CONTINUITY_POLE_LINE_DIST_MAX_PX}")
    print(f"PAIR_LINE_DIST_MAX_PX             : {CONTINUITY_PAIR_LINE_DIST_MAX_PX}")
    print(f"MERGE_MIN_PC1_RATIO               : {CONTINUITY_MERGE_MIN_PC1_RATIO}")
    print(f"MERGE_MIN_ANISOTROPY              : {CONTINUITY_MERGE_MIN_ANISOTROPY}")
    print(f"After continuity merge            : {len(final_kept_detections_df)}")
    print(f"Merged away as same-arm pieces    : {len(removed_by_continuity_merge_df)}")

    if len(continuity_pair_debug_df) > 0:
        print("\nPairwise continuity diagnostics:")
        _safe_display(continuity_pair_debug_df)

    if len(removed_by_continuity_merge_df) > 0:
        print("\nMerged away by continuity grouping:")
        _safe_display(
            removed_by_continuity_merge_df[
                [
                    c for c in [
                        "orig_det_idx",
                        "score",
                        "continuity_group_id",
                        "continuity_representative_orig_det_idx",
                        "continuity_angle_deg",
                        "continuity_pole_line_dist",
                        "x1", "y1", "x2", "y2",
                        "removal_reason",
                    ]
                    if c in removed_by_continuity_merge_df.columns
                ]
            ]
        )

    if len(final_kept_detections_df) > 0:
        print("\nKept after continuity merge:")
        _safe_display(
            final_kept_detections_df[
                [
                    c for c in [
                        "orig_det_idx",
                        "score",
                        "continuity_group_id",
                        "continuity_group_size",
                        "continuity_member_orig_det_idxs",
                        "continuity_merged",
                        "continuity_angle_deg",
                        "continuity_pole_line_dist",
                        "continuity_pc1_ratio",
                        "continuity_anisotropy",
                        "x1", "y1", "x2", "y2",
                        "box_w", "box_h",
                        "aspect_ratio",
                        "relative_width_to_max",
                    ]
                    if c in final_kept_detections_df.columns
                ]
            ]
        )

else:
    print("\nCrossarm continuity merge skipped.")

# Save stage snapshot
stage_continuity_df = final_kept_detections_df.copy()


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

    # -------------------------------------------------------------------------
    # Height sanity check
    # -------------------------------------------------------------------------
    if level_median_box_h > 0:
        tmp_df["keep_not_tall"] = (
            tmp_df["box_h"] <= (MAX_BOX_H_TO_MEDIAN_RATIO * level_median_box_h)
        )
    else:
        tmp_df["keep_not_tall"] = True

    # IMPORTANT:
    # A continuity-merged diagonal crossarm can have a taller union box than the
    # individual detections. Do not remove it as "too tall" during level dedupe.
    if "continuity_merged" in tmp_df.columns:
        tmp_df.loc[tmp_df["continuity_merged"] == True, "keep_not_tall"] = True

    # -------------------------------------------------------------------------
    # Ranking within each level band
    # -------------------------------------------------------------------------
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
                        "continuity_group_id",
                        "continuity_group_size",
                        "continuity_merged",
                        "continuity_member_orig_det_idxs",
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
                        "continuity_group_id",
                        "continuity_group_size",
                        "continuity_merged",
                        "continuity_member_orig_det_idxs",
                        "level_band_id",
                        "level_rank_score",
                        "level_rank",
                        "keep_not_tall",
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