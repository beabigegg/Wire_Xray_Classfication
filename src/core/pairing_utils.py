"""
Wire TOP/SIDE Pairing Utilities

This module provides utilities for handling paired TOP/SIDE images of the same wire.
Implements the quick fix approach for wire pairing without database schema changes.

Filename Convention:
- TOP images: {wire_id}_TOP.jpg
- SIDE images: {wire_id}_SIDE.jpg

Example:
- 001_TOP.jpg and 001_SIDE.jpg are paired (wire_id = "001")
- 042_TOP.jpg and 042_SIDE.jpg are paired (wire_id = "042")
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import re


def extract_wire_id(filename: str) -> Optional[str]:
    """
    Extract wire ID from filename.

    Args:
        filename: Image filename (e.g., "001_TOP.jpg", "042_SIDE.jpg")

    Returns:
        Wire ID string if valid format, None otherwise

    Examples:
        >>> extract_wire_id("001_TOP.jpg")
        "001"
        >>> extract_wire_id("042_SIDE.jpg")
        "042"
        >>> extract_wire_id("invalid.jpg")
        None
    """
    # Pattern: {wire_id}_TOP.jpg or {wire_id}_SIDE.jpg
    pattern = r"^(.+)_(TOP|SIDE)\.(jpg|jpeg|png)$"
    match = re.match(pattern, filename, re.IGNORECASE)

    if match:
        return match.group(1)  # Wire ID
    return None


def get_view_type(filename: str) -> Optional[str]:
    """
    Extract view type (TOP or SIDE) from filename.

    Args:
        filename: Image filename

    Returns:
        "TOP" or "SIDE" if valid format, None otherwise

    Examples:
        >>> get_view_type("001_TOP.jpg")
        "TOP"
        >>> get_view_type("042_SIDE.jpg")
        "SIDE"
    """
    pattern = r"^.+_(TOP|SIDE)\.(jpg|jpeg|png)$"
    match = re.match(pattern, filename, re.IGNORECASE)

    if match:
        return match.group(1).upper()
    return None


def find_pair_filename(filename: str) -> Optional[str]:
    """
    Find the paired filename for a given TOP or SIDE image.

    Args:
        filename: Image filename

    Returns:
        Paired filename if input is valid, None otherwise

    Examples:
        >>> find_pair_filename("001_TOP.jpg")
        "001_SIDE.jpg"
        >>> find_pair_filename("042_SIDE.jpg")
        "042_TOP.jpg"
    """
    wire_id = extract_wire_id(filename)
    view_type = get_view_type(filename)

    if not wire_id or not view_type:
        return None

    # Get file extension
    ext = Path(filename).suffix

    # Swap view type
    paired_view = "SIDE" if view_type == "TOP" else "TOP"

    return f"{wire_id}_{paired_view}{ext}"


def group_images_by_wire(filenames: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Group image filenames by wire ID, organizing TOP/SIDE pairs.

    Args:
        filenames: List of image filenames

    Returns:
        Dictionary mapping wire_id to {"TOP": filename, "SIDE": filename}
        Only includes wires that have both views.

    Examples:
        >>> group_images_by_wire(["001_TOP.jpg", "001_SIDE.jpg", "002_TOP.jpg"])
        {"001": {"TOP": "001_TOP.jpg", "SIDE": "001_SIDE.jpg"}}
    """
    wire_groups = {}

    for filename in filenames:
        wire_id = extract_wire_id(filename)
        view_type = get_view_type(filename)

        if not wire_id or not view_type:
            continue

        if wire_id not in wire_groups:
            wire_groups[wire_id] = {}

        wire_groups[wire_id][view_type] = filename

    # Filter to only complete pairs
    complete_pairs = {
        wire_id: views
        for wire_id, views in wire_groups.items()
        if "TOP" in views and "SIDE" in views
    }

    return complete_pairs


def validate_pairing(filenames: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate wire pairing and identify incomplete pairs.

    Args:
        filenames: List of image filenames

    Returns:
        Tuple of (complete_pairs, incomplete_singles)
        - complete_pairs: List of wire IDs with both TOP and SIDE
        - incomplete_singles: List of filenames without pairs

    Examples:
        >>> validate_pairing(["001_TOP.jpg", "001_SIDE.jpg", "002_TOP.jpg"])
        (["001"], ["002_TOP.jpg"])
    """
    wire_groups = {}

    for filename in filenames:
        wire_id = extract_wire_id(filename)
        view_type = get_view_type(filename)

        if not wire_id or not view_type:
            continue

        if wire_id not in wire_groups:
            wire_groups[wire_id] = {}

        wire_groups[wire_id][view_type] = filename

    complete_pairs = []
    incomplete_singles = []

    for wire_id, views in wire_groups.items():
        if "TOP" in views and "SIDE" in views:
            complete_pairs.append(wire_id)
        else:
            # Add incomplete singles
            if "TOP" in views:
                incomplete_singles.append(views["TOP"])
            if "SIDE" in views:
                incomplete_singles.append(views["SIDE"])

    return complete_pairs, incomplete_singles


def get_pair_defect_labels(
    wire_id: str,
    images_by_wire: Dict[str, Dict[str, str]],
    image_to_defect: Dict[str, str]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get defect labels for both views of a wire pair.

    Args:
        wire_id: Wire identifier
        images_by_wire: Wire grouping from group_images_by_wire()
        image_to_defect: Mapping from filename to defect label

    Returns:
        Tuple of (top_defect, side_defect) or (None, None) if not found
    """
    if wire_id not in images_by_wire:
        return None, None

    views = images_by_wire[wire_id]
    top_defect = image_to_defect.get(views.get("TOP"))
    side_defect = image_to_defect.get(views.get("SIDE"))

    return top_defect, side_defect


def combine_pair_predictions(
    top_result: Dict,
    side_result: Dict,
    strategy: str = "worst_case"
) -> Dict:
    """
    Combine TOP and SIDE inference results into final decision.

    Args:
        top_result: Inference result for TOP view
        side_result: Inference result for SIDE view
        strategy: Combination strategy
            - "worst_case": Fail if either view shows defect
            - "majority": Use more severe defect
            - "confidence": Use higher confidence prediction

    Returns:
        Combined result dictionary

    Note:
        Defect severity order: PASS < 沖線 < 晃動 < 碰觸
    """
    defect_severity = {
        "PASS": 0,
        "沖線": 1,
        "晃動": 2,
        "碰觸": 3
    }

    top_defect = top_result.get("defect_type", "PASS")
    side_defect = side_result.get("defect_type", "PASS")

    if strategy == "worst_case":
        # Choose more severe defect
        if defect_severity.get(top_defect, 0) > defect_severity.get(side_defect, 0):
            final_defect = top_defect
            final_confidence = top_result.get("defect_confidence", 0.0)
            decisive_view = "TOP"
        else:
            final_defect = side_defect
            final_confidence = side_result.get("defect_confidence", 0.0)
            decisive_view = "SIDE"

    elif strategy == "confidence":
        # Choose higher confidence prediction
        top_conf = top_result.get("defect_confidence", 0.0)
        side_conf = side_result.get("defect_confidence", 0.0)

        if top_conf > side_conf:
            final_defect = top_defect
            final_confidence = top_conf
            decisive_view = "TOP"
        else:
            final_defect = side_defect
            final_confidence = side_conf
            decisive_view = "SIDE"

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return {
        "defect_type": final_defect,
        "defect_confidence": final_confidence,
        "decisive_view": decisive_view,
        "top_result": top_result,
        "side_result": side_result
    }
