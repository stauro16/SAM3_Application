# =============================================================================
# CELL 14B — DETECT POLES ON 5 DEBUG IMAGES
#            (3-STAGE TROUBLESHOOTING VIEW + MASK OVERLAY CARRIED FORWARD)
# =============================================================================
# EXPLANATION:
# This cell runs SAM3 on up to FIVE debug images using pole-focused text prompts.
#
# WHAT THIS CELL DOES:
#   1) validates required objects
#   2) selects up to 5 debug images from debug_images_df
#   3) loads each image in RGB mode
#   4) sets each image once on the SAM3 processor
#   5) runs the configured pole prompts
#   6) shows ONE plot_results(...) diagnostic to confirm SAM3 can render masks
#   7) collects all raw detections
#   8) normalises masks and stores them in a separate lookup dictionary
#   9) computes simple pole ranking features PER IMAGE
#   10) prefilters weak / tiny / short / non-vertical / too-wide candidates
#   11) scores remaining candidates with shaft-penalty weighting
#   12) selects one best pole box per image
#   13) visualizes 3 troubleshooting stages across all debug images:
#       - RAW detections      -> yellow mask + yellow box
#       - KEPT candidates     -> cyan mask + cyan box
#       - SELECTED best pole  -> red mask + red box
#
# SAVED GLOBALS:
#   - POLE_5IMG_RAW_CANDIDATES_DF
#   - POLE_5IMG_FEATURE_DEBUG_DF
#   - POLE_5IMG_CANDIDATES_DF
#   - POLE_5IMG_SELECTION_DF
#   - POLE_5IMG_RESULTS
#   - POLE_5IMG_MASK_LOOKUP
#
# IMPORTANT:
# - this cell is intentionally MULTI-IMAGE DEBUG scoped
# - feature normalisation is computed PER IMAGE inside the loop
# - this cell does NOT replace 14A; it is the 5-image companion debug cell
# - if you later promote this logic to a production batch cell, keep the
#   per-image scoring boundaries intact
# - POLE_5IMG_RESULTS stores image_rgb arrays in memory for quick gallery
#   rendering; this is fine for 5 debug images only, not for large batch runs
# - Cell 9 already creates the processor with confidence_threshold=0.3
#   so this cell does NOT re-set the confidence threshold
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Safety checks
# -----------------------------------------------------------------------------
required_globals = [
    "debug_images_df",
    "model",
    "processor",
    "DEVICE",
    "POLE_PROMPTS",
    "GLOBAL_TEXT_SCORE_THRESHOLD",
]

missing_globals = [name for name in required_globals if name not in globals()]

if missing_globals:
    raise NameError(
        "CELL 14B requires objects from earlier cells.\n"
        "Please run the required setup / image-preparation cells first.\n"
        f"Missing globals: {missing_globals}"
    )

if not isinstance(debug_images_df, pd.DataFrame):
    raise TypeError("debug_images_df exists but is not a pandas DataFrame.")

if debug_images_df.empty:
    raise ValueError(
        "debug_images_df is empty.\n"
        "Please check the earlier image preparation cell."
    )

# -----------------------------------------------------------------------------
# 1. Config
# -----------------------------------------------------------------------------
POLE_DEBUG_IMAGE_COUNT = min(5, len(debug_images_df))
RUN_DEVICE = DEVICE
PROCESSOR_CONFIDENCE_THRESHOLD = 0.3

# Show plot_results only once across the whole 5-image run.
RUN_PLOT_RESULTS_DIAGNOSTIC = True
_plot_results_shown = False

# Simple prefilter / ranking rules for troubleshooting.
POLE_MIN_SCORE = float(globals().get("POLE_MIN_SCORE", 0.25))
POLE_MIN_AREA_FRAC = float(globals().get("POLE_MIN_AREA_FRAC", 0.005))    # 0.5% of image area
POLE_MIN_HEIGHT_FRAC = float(globals().get("POLE_MIN_HEIGHT_FRAC", 0.15)) # 15% of image height
POLE_MIN_ASPECT = float(globals().get("POLE_MIN_ASPECT", 1.80))           # h / w
POLE_MAX_WIDTH_FRAC = float(globals().get("POLE_MAX_WIDTH_FRAC", 0.08))   # 8% of image width
POLE_MAX_BOX_W_PX = float(globals().get("POLE_MAX_BOX_W_PX", 400))

W_X_CENTER = float(globals().get("W_X_CENTER", 0.45))
W_HEIGHT   = float(globals().get("W_HEIGHT", 0.30))
W_AREA     = float(globals().get("W_AREA", 0.10))
W_CONF     = float(globals().get("W_CONF", 0.10))
W_EDGE     = float(globals().get("W_EDGE", 0.05))

# Shaft-penalty settings.
SHAFT_WIDTH_FRAC_THRESHOLD = float(globals().get("SHAFT_WIDTH_FRAC_THRESHOLD", 0.12))
SHAFT_PENALTY_FACTOR       = float(globals().get("SHAFT_PENALTY_FACTOR", 0.40))

# -----------------------------------------------------------------------------
# 2. Small local helpers
# -----------------------------------------------------------------------------
def _to_numpy_safe(x):
    # IMPORTANT:
    # Do NOT special-case list/tuple here.
    # _normalize_masks_local handles raw_masks as list/tuple directly before
    # calling _to_numpy_safe, which avoids object-array problems for lists of
    # differently-shaped arrays.
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _candidate_key(image_id, prompt_text, det_idx):
    """
    Build a stable lookup key for one detection.
    """
    return (str(image_id), str(prompt_text), int(det_idx))

def _candidate_key_from_row(row):
    """
    Rebuild the same lookup key from a DataFrame row.
    """
    return _candidate_key(
        image_id=row["image_id"],
        prompt_text=row["prompt_text"],
        det_idx=row["det_idx"],
    )

def _infer_num_detections(raw_boxes, raw_scores, raw_masks):
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
    Normalize raw stateful-path masks into a list of 2D boolean masks.
    Handles the list/tuple-of-arrays case directly here before calling
    _to_numpy_safe, which avoids the object-array problem that would occur if
    np.asarray were called on differently-shaped arrays.
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

def _safe_display(obj):
    try:
        display(obj)
    except Exception:
        print("WARNING: display() unavailable, falling back to print().")
        print(obj)

def _clip_box_to_image(x1, y1, x2, y2, image_w, image_h):
    x1 = float(np.clip(x1, 0, max(image_w - 1, 0)))
    y1 = float(np.clip(y1, 0, max(image_h - 1, 0)))
    x2 = float(np.clip(x2, 0, max(image_w - 1, 0)))
    y2 = float(np.clip(y2, 0, max(image_h - 1, 0)))

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    return x1, y1, x2, y2

def _get_stage_style(stage="raw"):
    if stage == "raw":
        return {
            "line_color": "yellow",
            "mask_rgb": (1.0, 1.0, 0.0),
            "mask_alpha": 0.22,
            "linewidth": 2.0,
        }
    if stage == "kept":
        return {
            "line_color": "cyan",
            "mask_rgb": (0.0, 1.0, 1.0),
            "mask_alpha": 0.22,
            "linewidth": 2.5,
        }
    return {
        "line_color": "red",
        "mask_rgb": (1.0, 0.0, 0.0),
        "mask_alpha": 0.28,
        "linewidth": 3.5,
    }

def _draw_stage_boxes(ax, image_rgb, boxes_df, title, stage="raw", selected_only=False, mask_lookup=None):
    style = _get_stage_style(stage)
    color = style["line_color"]
    lw = style["linewidth"]
    mask_rgb = style["mask_rgb"]
    mask_alpha = style["mask_alpha"]

    ax.imshow(image_rgb)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    if boxes_df is None or len(boxes_df) == 0:
        return

    plot_df = boxes_df.copy()

    if selected_only:
        if "is_selected_pole" in plot_df.columns:
            plot_df = plot_df[plot_df["is_selected_pole"] == True].copy()
        else:
            plot_df = plot_df.iloc[0:0].copy()

    if plot_df.empty:
        return

    # -------------------------------------------------------------------------
    # Draw masks first so boxes and labels stay visible on top
    # -------------------------------------------------------------------------
    if isinstance(mask_lookup, dict):
        for _, r in plot_df.iterrows():
            key = _candidate_key_from_row(r)
            mask_2d = mask_lookup.get(key, None)

            if not isinstance(mask_2d, np.ndarray):
                continue

            if mask_2d.ndim != 2:
                continue

            if mask_2d.shape != image_rgb.shape[:2]:
                continue

            if mask_2d.sum() == 0:
                continue

            overlay = np.zeros((mask_2d.shape[0], mask_2d.shape[1], 4), dtype=np.float32)
            overlay[..., 0] = mask_rgb[0]
            overlay[..., 1] = mask_rgb[1]
            overlay[..., 2] = mask_rgb[2]
            overlay[..., 3] = mask_2d.astype(np.float32) * mask_alpha

            ax.imshow(overlay)

    # -------------------------------------------------------------------------
    # Draw boxes and labels second
    # -------------------------------------------------------------------------
    for _, r in plot_df.iterrows():
        x1, y1, x2, y2 = float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)

        rect = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=lw,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        label_bits = ["POLE"]

        prompt_text = None
        if "prompt_text" in r and pd.notna(r["prompt_text"]) and str(r["prompt_text"]).strip():
            prompt_text = str(r["prompt_text"]).strip()

        if prompt_text:
            label_bits.append(prompt_text)

        if "score" in r and pd.notna(r["score"]):
            label_bits.append(f"score={float(r['score']):.3f}")

        if stage == "selected" and "final_score" in r and pd.notna(r["final_score"]):
            label_bits.append(f"final={float(r['final_score']):.3f}")

        label = " | ".join(label_bits)

        ax.text(
            x1,
            max(y1 - 6, 8),
            label,
            fontsize=8.5,
            color="white",
            bbox=dict(
                facecolor=color,
                alpha=0.85,
                edgecolor="none",
                pad=2.0,
            ),
        )

def _existing_cols(df, cols):
    return [c for c in cols if c in df.columns]

def _show_multi_image_step_gallery(step_title, step_results, df_key, stage="raw", selected_only=False, mask_lookup=None):
    if len(step_results) == 0:
        print(f"No step results available for: {step_title}")
        return

    n = len(step_results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))

    if n == 1:
        axes = [axes]

    for ax, item in zip(axes, step_results):
        image_rgb = item["image_rgb"]
        file_name = item["file_name"]
        df = item.get(df_key, pd.DataFrame())

        panel_title = f"{file_name}\n{step_title}"
        if df_key == "kept_only_df" and bool(item.get("fallback_triggered", False)):
            panel_title = f"{file_name}\n{step_title} (FALLBACK MODE)"

        _draw_stage_boxes(
            ax=ax,
            image_rgb=image_rgb,
            boxes_df=df,
            title=panel_title,
            stage=stage,
            selected_only=selected_only,
            mask_lookup=mask_lookup,
        )

    plt.tight_layout()
    plt.show()
    plt.close()

# -----------------------------------------------------------------------------
# 3. Select up to 5 debug images
# -----------------------------------------------------------------------------
selected_debug_df = debug_images_df.head(POLE_DEBUG_IMAGE_COUNT).copy().reset_index(drop=True)

print("Multi-image pole-detection prototype:\n")
print(f"  POLE_DEBUG_IMAGE_COUNT         : {POLE_DEBUG_IMAGE_COUNT}")
print(f"  RUN_DEVICE                     : {RUN_DEVICE}")
print(f"  PROCESSOR_CONFIDENCE_THRESHOLD : {PROCESSOR_CONFIDENCE_THRESHOLD}")
print(f"  POLE_PROMPTS                   :")
for p in POLE_PROMPTS:
    print(f"    - {p}")

print("\nPrefilter config:")
print(f"  POLE_MIN_SCORE             : {POLE_MIN_SCORE}")
print(f"  POLE_MIN_AREA_FRAC         : {POLE_MIN_AREA_FRAC}")
print(f"  POLE_MIN_HEIGHT_FRAC       : {POLE_MIN_HEIGHT_FRAC}")
print(f"  POLE_MIN_ASPECT            : {POLE_MIN_ASPECT}")
print(f"  POLE_MAX_WIDTH_FRAC        : {POLE_MAX_WIDTH_FRAC}")
print(f"  POLE_MAX_BOX_W_PX          : {POLE_MAX_BOX_W_PX}")
print(f"  SHAFT_WIDTH_FRAC_THRESHOLD : {SHAFT_WIDTH_FRAC_THRESHOLD}")
print(f"  SHAFT_PENALTY_FACTOR       : {SHAFT_PENALTY_FACTOR}")

# -----------------------------------------------------------------------------
# 4. Run pole detection per image
# -----------------------------------------------------------------------------
all_raw_rows = []
all_feature_rows = []
all_scored_rows = []
all_selected_rows = []
step_results = []

POLE_5IMG_MASK_LOOKUP = {}

for row_idx in range(len(selected_debug_df)):
    row = selected_debug_df.iloc[row_idx]

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
            "image_id is None for one of the selected debug rows.\n"
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

    print("\n" + "=" * 90)
    print(f"Processing debug image {row_idx + 1}/{len(selected_debug_df)}")
    print("=" * 90)
    print(f"  image_id      : {image_id}")
    print(f"  file_name     : {file_name}")
    print(f"  image_path    : {image_path}")
    print(f"  image_size    : {image_w} x {image_h}")
    print(f"  original_mode : {original_mode}")

    # -------------------------------------------------------------------------
    # Prepare processor state
    # -------------------------------------------------------------------------
    if hasattr(processor, "device"):
        processor.device = RUN_DEVICE

    state = {}
    state = processor.set_image(image, state=state)

    # -------------------------------------------------------------------------
    # Run prompts and collect raw candidates
    # -------------------------------------------------------------------------
    candidate_rows = []

    for prompt_text in POLE_PROMPTS:
        reset_result = processor.reset_all_prompts(state)

        if reset_result is not None:
            state = reset_result

        state = processor.set_text_prompt(prompt_text, state)

        # ---------------------------------------------------------------------
        # plot_results diagnostic — show once across the whole cell
        # ---------------------------------------------------------------------
        if RUN_PLOT_RESULTS_DIAGNOSTIC and not _plot_results_shown:
            print("\nSAM3 plot_results diagnostic:")
            print(f"  image_id     : {image_id}")
            print(f"  prompt_text  : {prompt_text}")
            plot_results(image.copy(), state)
            _plot_results_shown = True

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

            bbox_w = max(1.0, x2 - x1)
            bbox_h = max(1.0, y2 - y1)

            mask_2d = masks_2d[det_idx] if det_idx < len(masks_2d) else None
            has_mask = isinstance(mask_2d, np.ndarray) and mask_2d.ndim == 2 and mask_2d.sum() > 0

            key = _candidate_key(image_id, prompt_text, det_idx)
            if has_mask:
                POLE_5IMG_MASK_LOOKUP[key] = mask_2d

            candidate_rows.append({
                "image_id": image_id,
                "file_name": file_name,
                "image_path": image_path,
                "image_w": int(image_w),
                "image_h": int(image_h),
                "prompt_text": prompt_text,
                "det_idx": int(det_idx),
                "score": float(scores[det_idx]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "has_mask": bool(has_mask),
            })

    raw_df = pd.DataFrame(candidate_rows)

    # -------------------------------------------------------------------------
    # Handle no-detection case for this image
    # -------------------------------------------------------------------------
    if raw_df.empty:
        print("  WARNING: No pole candidates were returned for this image.")

        feature_df = raw_df.copy()
        scored_df = raw_df.copy()
        selected_df = raw_df.copy()
        kept_only_df = raw_df.copy()
        fallback_triggered = False

        step_results.append({
            "image_id": image_id,
            "file_name": file_name,
            "image_path": image_path,
            "image_rgb": image_rgb,
            "raw_df": raw_df,
            "feature_df": feature_df,
            "scored_df": scored_df,
            "selected_df": selected_df,
            "kept_only_df": kept_only_df,
            "fallback_triggered": fallback_triggered,
        })

        continue

    # -------------------------------------------------------------------------
    # Add per-image pole features
    # -------------------------------------------------------------------------
    feature_df = raw_df.copy()

    feature_df["area"] = feature_df["bbox_w"] * feature_df["bbox_h"]
    feature_df["cx"] = (feature_df["x1"] + feature_df["x2"]) / 2.0
    feature_df["cy"] = (feature_df["y1"] + feature_df["y2"]) / 2.0
    feature_df["image_area"] = feature_df["image_w"] * feature_df["image_h"]

    feature_df["area_frac"] = (
        feature_df["area"] / feature_df["image_area"].clip(lower=1.0)
    )
    feature_df["height_frac"] = (
        feature_df["bbox_h"] / feature_df["image_h"].clip(lower=1.0)
    )
    feature_df["width_frac"] = (
        feature_df["bbox_w"] / feature_df["image_w"].clip(lower=1.0)
    )
    feature_df["aspect_ratio"] = (
        feature_df["bbox_h"] / feature_df["bbox_w"].clip(lower=1.0)
    )

    feature_df["center_dist_norm"] = (
        np.abs(feature_df["cx"] - image_cx) / max(image_w / 2.0, 1.0)
    )
    feature_df["x_center_score"] = 1.0 - np.clip(
        feature_df["center_dist_norm"], 0.0, 1.0
    )

    # IMPORTANT:
    # Per-image normalisation only.
    max_h = max(float(feature_df["bbox_h"].max()), 1.0)
    max_a = max(float(feature_df["area"].max()), 1.0)

    feature_df["height_score"] = feature_df["bbox_h"] / max_h
    feature_df["area_score"] = feature_df["area"] / max_a
    feature_df["conf_score"] = feature_df["score"]

    edge_margin = np.minimum.reduce([
        feature_df["x1"].values,
        feature_df["y1"].values,
        (feature_df["image_w"] - feature_df["x2"]).values,
        (feature_df["image_h"] - feature_df["y2"]).values,
    ])

    edge_norm_denom = 0.05 * np.minimum(
        feature_df["image_w"],
        feature_df["image_h"]
    )
    edge_norm_denom = edge_norm_denom.clip(lower=1.0)

    feature_df["edge_margin"] = edge_margin
    feature_df["edge_score"] = np.clip(
        feature_df["edge_margin"] / edge_norm_denom,
        0.0,
        1.0
    )

    # -------------------------------------------------------------------------
    # Prefilter candidates
    # -------------------------------------------------------------------------
    scored_df = feature_df.copy()

    scored_df["keep_score"] = scored_df["score"] >= POLE_MIN_SCORE
    scored_df["keep_area"] = scored_df["area_frac"] >= POLE_MIN_AREA_FRAC
    scored_df["keep_height"] = scored_df["height_frac"] >= POLE_MIN_HEIGHT_FRAC
    scored_df["keep_aspect"] = scored_df["aspect_ratio"] >= POLE_MIN_ASPECT
    scored_df["keep_width_frac"] = scored_df["width_frac"] <= POLE_MAX_WIDTH_FRAC
    scored_df["keep_width_px"] = scored_df["bbox_w"] <= POLE_MAX_BOX_W_PX

    scored_df["is_kept_after_prefilter"] = (
        scored_df["keep_score"] &
        scored_df["keep_area"] &
        scored_df["keep_height"] &
        scored_df["keep_aspect"] &
        scored_df["keep_width_frac"] &
        scored_df["keep_width_px"]
    )

    # -------------------------------------------------------------------------
    # Score candidates and select best pole
    # -------------------------------------------------------------------------
    scored_df["selection_mode"] = "not_kept"
    scored_df["final_score"] = np.nan
    scored_df["is_selected_pole"] = False

    kept_df = scored_df[
        scored_df["is_kept_after_prefilter"] == True
    ].copy()

    fallback_triggered = kept_df.empty

    if fallback_triggered:
        print(
            "  WARNING: All candidates failed prefilter for this image. "
            "Falling back to full candidate pool with shaft-penalty scoring."
        )
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
        by=["final_score", "score", "bbox_h", "x_center_score"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    if len(kept_df) > 0:
        kept_df.loc[0, "is_selected_pole"] = True

    score_cols = [
        "image_id",
        "prompt_text",
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
        on=["image_id", "prompt_text", "det_idx"],
        how="left",
    )

    scored_df["selection_mode"] = scored_df["selection_mode"].fillna("not_kept")
    scored_df["is_selected_pole"] = scored_df["is_selected_pole"].fillna(False)
    scored_df["shaft_penalty"] = scored_df["shaft_penalty"].fillna(np.nan)

    selected_df = scored_df[
        scored_df["is_selected_pole"] == True
    ].copy()

    kept_only_df = scored_df[
        scored_df["is_kept_after_prefilter"] == True
    ].copy()

    if len(selected_df) > 0:
        assert len(selected_df) == 1, (
            f"Unexpected number of selected poles for image_id={image_id}"
        )
        best_row = selected_df.iloc[0]
        print(
            f"  Selected pole -> det_idx={int(best_row['det_idx'])}, "
            f"score={float(best_row['score']):.3f}, "
            f"final_score={float(best_row['final_score']):.3f}, "
            f"has_mask={bool(best_row['has_mask'])}"
        )
    else:
        print("  No selected pole was produced for this image.")

    # -------------------------------------------------------------------------
    # Collect outputs for this image
    # -------------------------------------------------------------------------
    step_results.append({
        "image_id": image_id,
        "file_name": file_name,
        "image_path": image_path,
        "image_rgb": image_rgb,
        "raw_df": raw_df,
        "feature_df": feature_df,
        "scored_df": scored_df,
        "selected_df": selected_df,
        "kept_only_df": kept_only_df,
        "fallback_triggered": fallback_triggered,
    })

    if len(raw_df) > 0:
        all_raw_rows.append(raw_df)

    if len(feature_df) > 0:
        all_feature_rows.append(feature_df)

    if len(scored_df) > 0:
        all_scored_rows.append(scored_df)

    if len(selected_df) > 0:
        all_selected_rows.append(selected_df)

# -----------------------------------------------------------------------------
# 5. Combine outputs across all debug images
# -----------------------------------------------------------------------------
POLE_5IMG_RAW_CANDIDATES_DF = (
    pd.concat(all_raw_rows, ignore_index=True)
    if len(all_raw_rows) > 0 else
    pd.DataFrame()
)

POLE_5IMG_FEATURE_DEBUG_DF = (
    pd.concat(all_feature_rows, ignore_index=True)
    if len(all_feature_rows) > 0 else
    pd.DataFrame()
)

POLE_5IMG_CANDIDATES_DF = (
    pd.concat(all_scored_rows, ignore_index=True)
    if len(all_scored_rows) > 0 else
    pd.DataFrame()
)

POLE_5IMG_SELECTION_DF = (
    pd.concat(all_selected_rows, ignore_index=True)
    if len(all_selected_rows) > 0 else
    pd.DataFrame()
)

POLE_5IMG_RESULTS = step_results

# -----------------------------------------------------------------------------
# 6. Step outputs — tables
# -----------------------------------------------------------------------------
print("\n" + "=" * 100)
print("STEP A/B — RAW DETECTIONS TABLE")
print("=" * 100)
if POLE_5IMG_RAW_CANDIDATES_DF.empty:
    print("No raw detections found across the selected debug images.")
else:
    raw_cols = _existing_cols(
        POLE_5IMG_RAW_CANDIDATES_DF,
        [
            "image_id", "file_name", "prompt_text", "det_idx",
            "score", "has_mask", "x1", "y1", "x2", "y2"
        ]
    )
    _safe_display(
        POLE_5IMG_RAW_CANDIDATES_DF[raw_cols]
        .sort_values(["file_name", "score"], ascending=[True, False])
        .reset_index(drop=True)
    )

print("\n" + "=" * 100)
print("STEP C — FEATURE TABLE")
print("=" * 100)
if POLE_5IMG_FEATURE_DEBUG_DF.empty:
    print("No feature rows available.")
else:
    feature_cols = _existing_cols(
        POLE_5IMG_FEATURE_DEBUG_DF,
        [
            "image_id", "file_name", "det_idx", "score",
            "has_mask",
            "bbox_w", "bbox_h", "area",
            "width_frac",
            "cx", "cy",
            "x_center_score", "height_score", "area_score",
            "conf_score", "edge_score", "aspect_ratio"
        ]
    )
    _safe_display(
        POLE_5IMG_FEATURE_DEBUG_DF[feature_cols]
        .sort_values(["file_name", "score"], ascending=[True, False])
        .reset_index(drop=True)
    )

print("\n" + "=" * 100)
print("STEP D — PREFILTER TABLE")
print("=" * 100)
if POLE_5IMG_CANDIDATES_DF.empty:
    print("No prefilter rows available.")
else:
    prefilter_cols = _existing_cols(
        POLE_5IMG_CANDIDATES_DF,
        [
            "image_id", "file_name", "det_idx", "score",
            "has_mask",
            "area_frac", "height_frac", "width_frac", "aspect_ratio",
            "keep_score", "keep_area", "keep_height", "keep_aspect",
            "keep_width_frac", "keep_width_px",
            "is_kept_after_prefilter"
        ]
    )
    _safe_display(
        POLE_5IMG_CANDIDATES_DF[prefilter_cols]
        .sort_values(["file_name", "score"], ascending=[True, False])
        .reset_index(drop=True)
    )

print("\n" + "=" * 100)
print("STEP E/F — FINAL RANKING + SELECTED POLE TABLE")
print("=" * 100)
if POLE_5IMG_CANDIDATES_DF.empty:
    print("No scored rows available.")
else:
    final_cols = _existing_cols(
        POLE_5IMG_CANDIDATES_DF,
        [
            "image_id", "file_name", "det_idx", "score",
            "has_mask",
            "x_center_score", "height_score", "area_score",
            "conf_score", "edge_score", "shaft_penalty",
            "final_score", "selection_mode", "is_selected_pole"
        ]
    )
    _safe_display(
        POLE_5IMG_CANDIDATES_DF[final_cols]
        .sort_values(
            ["file_name", "is_selected_pole", "final_score", "score"],
            ascending=[True, False, False, False]
        )
        .reset_index(drop=True)
    )

# -----------------------------------------------------------------------------
# 7. Step outputs — galleries across up to 5 debug images
# -----------------------------------------------------------------------------
print("\n" + "=" * 100)
print("STEP A/B — RAW DETECTIONS OVERLAY")
print("=" * 100)
_show_multi_image_step_gallery(
    step_title="RAW detections",
    step_results=step_results,
    df_key="raw_df",
    stage="raw",
    selected_only=False,
    mask_lookup=POLE_5IMG_MASK_LOOKUP,
)

print("\n" + "=" * 100)
print("STEP D — KEPT CANDIDATES AFTER PREFILTER")
print("=" * 100)
_show_multi_image_step_gallery(
    step_title="KEPT after prefilter",
    step_results=step_results,
    df_key="kept_only_df",
    stage="kept",
    selected_only=False,
    mask_lookup=POLE_5IMG_MASK_LOOKUP,
)

print("\n" + "=" * 100)
print("STEP F — FINAL SELECTED CENTER POLE")
print("=" * 100)
_show_multi_image_step_gallery(
    step_title="SELECTED best pole",
    step_results=step_results,
    df_key="scored_df",
    stage="selected",
    selected_only=True,
    mask_lookup=POLE_5IMG_MASK_LOOKUP,
)

# -----------------------------------------------------------------------------
# 8. Selected pole table
# -----------------------------------------------------------------------------
if not POLE_5IMG_SELECTION_DF.empty:
    print("\nSelected poles:")

    selected_cols = _existing_cols(
        POLE_5IMG_SELECTION_DF,
        [
            "image_id", "file_name", "det_idx", "score",
            "has_mask",
            "x1", "y1", "x2", "y2",
            "cx", "cy",
            "bbox_w", "bbox_h", "width_frac",
            "shaft_penalty",
            "final_score", "selection_mode"
        ]
    )
    _safe_display(
        POLE_5IMG_SELECTION_DF[selected_cols].reset_index(drop=True)
    )
else:
    print("\nNo selected poles were produced across the chosen debug images.")

# -----------------------------------------------------------------------------
# 9. Final confirmation
# -----------------------------------------------------------------------------
print("\nCELL 14B completed successfully.")
print("Saved globals:")
print("  - POLE_5IMG_RAW_CANDIDATES_DF")
print("  - POLE_5IMG_FEATURE_DEBUG_DF")
print("  - POLE_5IMG_CANDIDATES_DF")
print("  - POLE_5IMG_SELECTION_DF")
print("  - POLE_5IMG_RESULTS")
print("  - POLE_5IMG_MASK_LOOKUP")