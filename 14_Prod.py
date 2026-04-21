# =============================================================================
# CELL 14 — PRODUCTION POLE-TOP FIXED CANVAS ROI + SHIFT + PAD + SAVE TO SILVER
# =============================================================================
# EXPLANATION:
# This cell builds one fixed-size pole-top ROI for every selected pole coming
# from CELL 13 production output.
#
# IMPORTANT DESIGN RULE:
# - DO NOT use the saved QA overlay PNGs as the crop source.
# - Use the original full-resolution source image from pole_selection_df["image_path"].
# - Use the full-resolution selected pole coordinates from pole_selection_df.
# - Keep the saved ROI crops clean (no pole mask overlay baked into the crop).
#
# WHAT THIS CELL DOES:
#   1) reads production pole_selection_df from CELL 13
#   2) keeps only rows where selection_status == "selected"
#   3) opens the original source image_path for each selected row
#   4) builds one fixed-size pole-top ROI request per image
#   5) shifts the ROI inside the image first when possible
#   6) pads only when needed so every saved ROI has the same final size
#   7) saves ROI PNGs into SILVER_POLE_ROIS
#   8) builds pole_rois_df for downstream production/debug cells
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
        "CELL 14 cannot run because some required variables are missing.\n"
        "Please run CELL 13 and CELL 10 first.\n"
        f"Missing globals: {missing_globals}"
    )

if not isinstance(pole_selection_df, pd.DataFrame):
    raise TypeError("pole_selection_df exists but is not a pandas DataFrame.")

if pole_selection_df.empty:
    raise ValueError("pole_selection_df is empty. Please check CELL 13.")

# -----------------------------------------------------------------------------
# 1. KEEP ONLY SELECTED POLES FROM CELL 13
# -----------------------------------------------------------------------------
# EXPLANATION:
# CELL 13 writes one row per input image into pole_selection_df.
# For ROI generation, we keep only rows where a reliable pole was selected.
# -----------------------------------------------------------------------------
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
        "Please check CELL 13 production output."
    )

# -----------------------------------------------------------------------------
# 2. REQUIRED SELECTED-ROW CONTRACT
# -----------------------------------------------------------------------------
required_selected_cols = [
    "image_path",
    "x1",
    "y1",
    "x2",
    "y2",
]

missing_selected_cols = [
    c for c in required_selected_cols if c not in selected_poles_df.columns
]

if missing_selected_cols:
    raise ValueError(
        "selected_poles_df is missing required columns from CELL 13.\n"
        f"Missing columns: {missing_selected_cols}"
    )

# -----------------------------------------------------------------------------
# 3. FIXED POLE-TOP ROI CONFIG
# -----------------------------------------------------------------------------
FIXED_ROI_WIDTH = int(globals().get("FIXED_ROI_WIDTH", 3500))
FIXED_ROI_HEIGHT = int(globals().get("FIXED_ROI_HEIGHT", 4250))
POLE_TOP_BUFFER_ABOVE = int(globals().get("POLE_TOP_BUFFER_ABOVE", 500))
PAD_RGB = tuple(globals().get("PAD_RGB", (0, 0, 0)))

# -----------------------------------------------------------------------------
# 3A. OVERWRITE CONTROL
# -----------------------------------------------------------------------------
OVERWRITE_POLE_ROIS = bool(globals().get("OVERWRITE_POLE_ROIS", True))

print("Fixed pole-top ROI config:")
print(f"  FIXED_ROI_WIDTH        : {FIXED_ROI_WIDTH}")
print(f"  FIXED_ROI_HEIGHT       : {FIXED_ROI_HEIGHT}")
print(f"  POLE_TOP_BUFFER_ABOVE  : {POLE_TOP_BUFFER_ABOVE}")
print(f"  PAD_RGB                : {PAD_RGB}")
print(f"  OVERWRITE_POLE_ROIS    : {OVERWRITE_POLE_ROIS}")
print(f"  selected_poles_df rows : {len(selected_poles_df)}")
print(f"  SILVER_POLE_ROIS       : {SILVER_POLE_ROIS}")

# -----------------------------------------------------------------------------
# 4. HELPER: CANDIDATE KEY
# -----------------------------------------------------------------------------
def _candidate_key(image_id, prompt, det_idx):
    """
    Build the same mask-lookup key used in CELL 13.

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
# 5. HELPER: EXTRACT TRAILING IMAGE-ID SUFFIX
# -----------------------------------------------------------------------------
def extract_image_suffix_id(image_id):
    """
    Extract the trailing numeric suffix from image_id.

    Examples:
        img_1298075_003_2060_V_0 -> "0"
        img_test_12             -> "12"

    Args:
        image_id:
            Stable image identifier from Cell 12.

    Returns:
        str:
            Trailing numeric suffix as a string.
    """
    if image_id is None:
        return "0"

    image_id_str = str(image_id).strip()
    parts = image_id_str.rsplit("_", 1)

    if len(parts) == 2 and parts[1].isdigit():
        return parts[1]

    return "0"

# -----------------------------------------------------------------------------
# 6. HELPER: BUILD OUTPUT PATH UNDER SILVER_POLE_ROIS
# -----------------------------------------------------------------------------
def build_roi_output_path(row, suffix_id):
    """
    Build the saved ROI output path.

    EXPLANATION:
    Preserve relative subfolder structure when possible so the Silver ROI folder
    stays organised the same way as the input image set.

    Filename strategy:
    - keep the original image stem for readability
    - append only the explicit numeric suffix id
    - avoid repeating the full stem twice

    Args:
        row:
            Selected pole row.

        suffix_id:
            Unique trailing numeric id, e.g. "0", "1", "2".

    Returns:
        tuple:
            (roi_file_name, roi_image_path)
    """
    relative_image_path = None

    if "relative_image_path" in row.index:
        relative_image_path = row["relative_image_path"]

    fallback_file_name = (
        row["file_name"]
        if "file_name" in row.index and pd.notna(row["file_name"])
        else "image"
    )

    if not isinstance(relative_image_path, str) or len(relative_image_path.strip()) == 0:
        relative_image_path = fallback_file_name

    relative_dir = os.path.dirname(relative_image_path)
    base_stem = os.path.splitext(os.path.basename(relative_image_path))[0]

    if relative_dir in ("", "."):
        target_dir = SILVER_POLE_ROIS
    else:
        target_dir = os.path.join(SILVER_POLE_ROIS, relative_dir)

    os.makedirs(target_dir, exist_ok=True)

    roi_file_name = f"{base_stem}__{suffix_id}__pole_roi.png"
    roi_image_path = os.path.join(target_dir, roi_file_name)

    return roi_file_name, roi_image_path

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

    if image_w >= box_w:
        if x1 < 0:
            x2 += (-x1)
            x1 = 0
        if x2 > image_w:
            x1 -= (x2 - image_w)
            x2 = image_w

    if image_h >= box_h:
        if y1 < 0:
            y2 += (-y1)
            y1 = 0
        if y2 > image_h:
            y1 -= (y2 - image_h)
            y2 = image_h

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
def build_pole_top_roi_request(row, image_w, image_h):
    """
    Build a fixed-size pole-top ROI request from the selected pole row.

    Args:
        row:
            Selected pole row from pole_selection_df.

        image_w, image_h:
            True source image dimensions.

    Returns:
        Dict[str, Any]:
            Requested fixed-size ROI geometry after shift-to-fit.
    """
    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])

    pole_w = max(x2 - x1, 1.0)
    pole_h = max(y2 - y1, 1.0)

    if "pole_cx" in row.index and pd.notna(row["pole_cx"]):
        pole_cx = float(row["pole_cx"])
    else:
        pole_cx = (x1 + x2) / 2.0

    req_x1 = pole_cx - (FIXED_ROI_WIDTH / 2.0)
    req_y1 = y1 - POLE_TOP_BUFFER_ABOVE

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

    IMPORTANT:
    This function creates a clean crop only.
    It does NOT draw the pole mask overlay onto the saved ROI image.

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

    src_x1 = max(0, req_x1)
    src_y1 = max(0, req_y1)
    src_x2 = min(image_w, req_x2)
    src_y2 = min(image_h, req_y2)

    overlap_w = max(0, src_x2 - src_x1)
    overlap_h = max(0, src_y2 - src_y1)

    dst_x1 = max(0, src_x1 - req_x1)
    dst_y1 = max(0, src_y1 - req_y1)

    roi_canvas = Image.new("RGB", (FIXED_ROI_WIDTH, FIXED_ROI_HEIGHT), PAD_RGB)

    if overlap_w > 0 and overlap_h > 0:
        src_crop = image_pil.crop((src_x1, src_y1, src_x2, src_y2))
        roi_canvas.paste(src_crop, (dst_x1, dst_y1))

    pad_left = int(max(0, -req_x1))
    pad_top = int(max(0, -req_y1))
    pad_right = int(max(0, req_x2 - image_w))
    pad_bottom = int(max(0, req_y2 - image_h))

    return {
        "roi_canvas": roi_canvas,
        "src_x1": int(src_x1),
        "src_y1": int(src_y1),
        "src_x2": int(src_x2),
        "src_y2": int(src_y2),
        "src_w": int(overlap_w),
        "src_h": int(overlap_h),
        "dst_x1": int(dst_x1),
        "dst_y1": int(dst_y1),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "pad_right": int(pad_right),
        "pad_bottom": int(pad_bottom),
        "was_padded": bool(
            (pad_left > 0) or (pad_top > 0) or
            (pad_right > 0) or (pad_bottom > 0)
        ),
    }

# -----------------------------------------------------------------------------
# 10. PREPARE / RESET THE SILVER ROI FOLDER
# -----------------------------------------------------------------------------
if os.path.isdir(SILVER_POLE_ROIS):
    existing_roi_items = os.listdir(SILVER_POLE_ROIS)

    if existing_roi_items and not OVERWRITE_POLE_ROIS:
        raise RuntimeError(
            "SILVER_POLE_ROIS already contains files.\n"
            "Re-running CELL 14 would overwrite the saved ROI working copy.\n"
            "If you really want to rebuild it, set:\n"
            "OVERWRITE_POLE_ROIS = True\n"
            "and then run CELL 14 again."
        )

    if existing_roi_items and OVERWRITE_POLE_ROIS:
        shutil.rmtree(SILVER_POLE_ROIS)

os.makedirs(SILVER_POLE_ROIS, exist_ok=True)

# -----------------------------------------------------------------------------
# 11. BUILD FIXED-SIZE ROI CANVASES AND SAVE TO SILVER
# -----------------------------------------------------------------------------
roi_rows = []

print(f"\nCreating fixed pole-top ROI crops for {len(selected_poles_df)} selected image(s)...")

for _, row in selected_poles_df.iterrows():
    image_id = row["image_id"] if "image_id" in row.index and pd.notna(row["image_id"]) else None
    file_name = row["file_name"] if "file_name" in row.index and pd.notna(row["file_name"]) else None
    image_path = row["image_path"] if "image_path" in row.index and pd.notna(row["image_path"]) else None

    if pd.isna(image_path) or not isinstance(image_path, str) or len(image_path.strip()) == 0:
        raise ValueError("A selected pole row is missing a valid image_path.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Original source image not found: {image_path}")

    if (
        pd.isna(file_name)
        or not isinstance(file_name, str)
        or len(file_name.strip()) == 0
    ):
        file_name = os.path.basename(image_path)

    prompt = row["prompt"] if "prompt" in row.index and pd.notna(row["prompt"]) else None

    if "det_idx" in row.index and pd.notna(row["det_idx"]):
        det_idx = int(row["det_idx"])
    else:
        raise ValueError(
            "A selected pole row is missing det_idx.\n"
            "Selected rows from CELL 13 are expected to carry a valid det_idx."
        )

    selection_mode = row["selection_mode"] if "selection_mode" in row.index and pd.notna(row["selection_mode"]) else None
    has_mask = bool(row["has_mask"]) if "has_mask" in row.index and pd.notna(row["has_mask"]) else False
    final_score = float(row["final_score"]) if "final_score" in row.index and pd.notna(row["final_score"]) else np.nan
    raw_score = float(row["score"]) if "score" in row.index and pd.notna(row["score"]) else np.nan

    mask_lookup_hit = False
    if (
        "pole_mask_lookup" in globals()
        and isinstance(globals().get("pole_mask_lookup"), dict)
        and prompt is not None
    ):
        mask_lookup_hit = _candidate_key(image_id, prompt, det_idx) in pole_mask_lookup

    # -------------------------------------------------------------------------
    # Compute the explicit suffix id once and carry it forward.
    # -------------------------------------------------------------------------
    suffix_id = extract_image_suffix_id(image_id)

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        image_w, image_h = img.size

        roi_request = build_pole_top_roi_request(
            row=row,
            image_w=image_w,
            image_h=image_h,
        )

        roi_file_name, roi_image_path = build_roi_output_path(row, suffix_id)

        roi_render = render_fixed_canvas_roi(
            image_pil=img,
            roi_request=roi_request,
        )

        roi_canvas = roi_render["roi_canvas"]
        roi_canvas.save(roi_image_path, format="PNG")
        roi_canvas.close()

    roi_rows.append({
        "image_id": image_id,
        "roi_suffix_id": suffix_id,
        "file_name": file_name,
        "relative_image_path": row["relative_image_path"] if "relative_image_path" in row.index else None,
        "image_path": image_path,

        # ---------------------------------------------------------------------
        # Carry-forward selected-pole identity fields
        # ---------------------------------------------------------------------
        "selection_status": "selected",
        "selection_mode": selection_mode,
        "prompt": prompt,
        "det_idx": det_idx,
        "score": raw_score,
        "final_score": final_score,
        "has_mask": has_mask,
        "mask_lookup_hit": bool(mask_lookup_hit),

        # ---------------------------------------------------------------------
        # Full source image geometry
        # ---------------------------------------------------------------------
        "image_w": int(image_w),
        "image_h": int(image_h),

        # ---------------------------------------------------------------------
        # Selected pole geometry
        # ---------------------------------------------------------------------
        "x1": float(row["x1"]),
        "y1": float(row["y1"]),
        "x2": float(row["x2"]),
        "y2": float(row["y2"]),
        "box_w": float(row["box_w"]) if "box_w" in row.index and pd.notna(row["box_w"]) else float(roi_request["pole_w"]),
        "box_h": float(row["box_h"]) if "box_h" in row.index and pd.notna(row["box_h"]) else float(roi_request["pole_h"]),
        "pole_cx": float(row["pole_cx"]) if "pole_cx" in row.index and pd.notna(row["pole_cx"]) else np.nan,
        "pole_cy": float(row["pole_cy"]) if "pole_cy" in row.index and pd.notna(row["pole_cy"]) else np.nan,
        "pole_w": float(roi_request["pole_w"]),
        "pole_h": float(roi_request["pole_h"]),
        "pole_cx_used": float(roi_request["pole_cx_used"]),

        # ---------------------------------------------------------------------
        # Requested fixed ROI geometry
        # ---------------------------------------------------------------------
        "req_x1": int(roi_request["req_x1"]),
        "req_y1": int(roi_request["req_y1"]),
        "req_x2": int(roi_request["req_x2"]),
        "req_y2": int(roi_request["req_y2"]),
        "req_w": int(roi_request["req_w"]),
        "req_h": int(roi_request["req_h"]),

        # ---------------------------------------------------------------------
        # Actual source overlap / paste geometry
        # ---------------------------------------------------------------------
        "src_x1": int(roi_render["src_x1"]),
        "src_y1": int(roi_render["src_y1"]),
        "src_x2": int(roi_render["src_x2"]),
        "src_y2": int(roi_render["src_y2"]),
        "src_w": int(roi_render["src_w"]),
        "src_h": int(roi_render["src_h"]),
        "dst_x1": int(roi_render["dst_x1"]),
        "dst_y1": int(roi_render["dst_y1"]),
        "pad_left": int(roi_render["pad_left"]),
        "pad_top": int(roi_render["pad_top"]),
        "pad_right": int(roi_render["pad_right"]),
        "pad_bottom": int(roi_render["pad_bottom"]),
        "was_padded": bool(roi_render["was_padded"]),

        # ---------------------------------------------------------------------
        # Saved fixed canvas ROI output
        # ---------------------------------------------------------------------
        "roi_w": int(FIXED_ROI_WIDTH),
        "roi_h": int(FIXED_ROI_HEIGHT),
        "roi_file_name": roi_file_name,
        "roi_image_path": roi_image_path,
        "source_layer": "silver",
        "source_folder": SILVER_POLE_ROIS,
    })

    if (
        (len(roi_rows) % 20 == 0)
        or (len(roi_rows) == 1)
        or (len(roi_rows) == len(selected_poles_df))
    ):
        print(f"  [{len(roi_rows)}/{len(selected_poles_df)}] {roi_file_name}")

    del roi_canvas
    del roi_render

gc.collect()

# -----------------------------------------------------------------------------
# 12. BUILD THE ROI MANIFEST
# -----------------------------------------------------------------------------
pole_rois_df = pd.DataFrame(roi_rows)

if pole_rois_df.empty:
    raise ValueError("pole_rois_df ended up empty. Please check the crop/save loop.")

# -----------------------------------------------------------------------------
# 13. VERIFY SAVED ROI FILES EXIST
# -----------------------------------------------------------------------------
missing_roi_files = [
    p for p in pole_rois_df["roi_image_path"].tolist()
    if not os.path.exists(p)
]

if len(missing_roi_files) > 0:
    raise RuntimeError(
        "Some ROI files were expected but were not found on disk.\n"
        f"Missing files: {missing_roi_files[:10]}"
    )

# -----------------------------------------------------------------------------
# 14. PRINT SUMMARY
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
print("Saved ROI type           : clean crops only (no pole mask overlay)")

# -----------------------------------------------------------------------------
# 15. SAVE OUTPUTS
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
            "FIXED_ROI_WIDTH": FIXED_ROI_WIDTH,
            "FIXED_ROI_HEIGHT": FIXED_ROI_HEIGHT,
            "POLE_TOP_BUFFER_ABOVE": POLE_TOP_BUFFER_ABOVE,
            "PAD_RGB": list(PAD_RGB),
            "OVERWRITE_POLE_ROIS": OVERWRITE_POLE_ROIS,
        },
        nb_globals=globals(),
    )
else:
    print(
        "Note: save_state not available in this Databricks notebook; "
        "outputs remain in globals only."
    )

print("\nCELL 14 completed.")
print("Saved outputs:")
print("  - pole_rois_df")
print("  - clean ROI PNG crops in SILVER_POLE_ROIS")
