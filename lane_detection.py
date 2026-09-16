"""
lane_detection.py

Core computer vision pipeline for detecting road lane lines in an image
or a video frame using classical (non-deep-learning) techniques:

    grayscale -> Gaussian blur -> Canny edges -> Region of Interest (ROI)
    -> Hough Line Transform -> average/extrapolate left & right lanes
    -> draw lines on the original frame

Every function does exactly one step of the pipeline so the whole flow
is easy to read, test, and explain (e.g. in a viva).
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Step 1: Grayscale conversion
# ---------------------------------------------------------------------------
def to_grayscale(frame):
    """Convert a BGR frame to single-channel grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Step 2: Noise reduction
# ---------------------------------------------------------------------------
def apply_gaussian_blur(gray_frame, kernel_size=5):
    """
    Smooth the grayscale image to suppress small noise/texture that would
    otherwise create false edges during Canny edge detection.
    """
    return cv2.GaussianBlur(gray_frame, (kernel_size, kernel_size), 0)


# ---------------------------------------------------------------------------
# Step 3: Edge detection
# ---------------------------------------------------------------------------
def detect_edges(blurred_frame, low_threshold=50, high_threshold=150):
    """
    Apply the Canny edge detector. Pixels with gradient above
    high_threshold are treated as strong edges; pixels between the two
    thresholds are kept only if connected to a strong edge.
    """
    return cv2.Canny(blurred_frame, low_threshold, high_threshold)


# ---------------------------------------------------------------------------
# Step 4: Region of Interest (ROI)
# ---------------------------------------------------------------------------
def region_of_interest(edges_frame):
    """
    Mask out everything except a triangular/trapezoidal region in the
    lower half of the frame, since that is where the road (and hence the
    lane lines) is expected to appear in a forward-facing dashcam shot.

    The polygon coordinates are defined as fractions of the frame's
    width/height so this works for any input resolution.
    """
    height, width = edges_frame.shape[:2]

    polygon = np.array([[
        (int(0.10 * width), height),                 # bottom-left
        (int(0.90 * width), height),                 # bottom-right
        (int(0.55 * width), int(0.60 * height)),      # top-right (near horizon)
        (int(0.45 * width), int(0.60 * height)),      # top-left  (near horizon)
    ]], dtype=np.int32)

    mask = np.zeros_like(edges_frame)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges_frame, mask)
    return masked_edges, polygon


# ---------------------------------------------------------------------------
# Step 5: Hough Line Transform
# ---------------------------------------------------------------------------
def detect_hough_lines(masked_edges):
    """
    Run the Probabilistic Hough Line Transform on the masked edge image
    and return an array of line segments (x1, y1, x2, y2).
    """
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,                # distance resolution in pixels
        theta=np.pi / 180,    # angle resolution in radians
        threshold=50,         # minimum number of votes (intersections)
        minLineLength=40,     # discard segments shorter than this
        maxLineGap=100        # max gap between segments to join them
    )
    return lines


# ---------------------------------------------------------------------------
# Step 6: Separate & average left / right lane lines
# ---------------------------------------------------------------------------
def average_slope_intercept(frame, lines):
    """
    Group raw Hough segments into a left lane and a right lane using the
    sign of their slope, average each group, and extrapolate a single
    line per side spanning from the bottom of the frame to the ROI apex.

    Returns a list containing up to two lines: [left_line, right_line].
    Each line is [x1, y1, x2, y2]. A side is skipped if no segments with
    a reasonable slope were found for it (e.g. dashed line not detected).
    """
    left_fit = []
    right_fit = []

    if lines is None:
        return []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue  # ignore perfectly vertical segments (undefined slope)

        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)

        # Ignore near-horizontal segments (slope close to 0) since real
        # lane markings on a forward-facing camera are never flat.
        if abs(slope) < 0.3:
            continue

        # In image coordinates, y increases downward.
        # A left lane line rises to the right  -> negative slope.
        # A right lane line rises to the left   -> positive slope.
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    lane_lines = []
    height = frame.shape[0]
    y1 = height                     # bottom of the frame
    y2 = int(height * 0.60)         # near the ROI horizon line

    if left_fit:
        left_avg = np.average(left_fit, axis=0)
        lane_lines.append(_make_line_points(y1, y2, left_avg))

    if right_fit:
        right_avg = np.average(right_fit, axis=0)
        lane_lines.append(_make_line_points(y1, y2, right_avg))

    return lane_lines


def _make_line_points(y1, y2, line_params):
    """Convert (slope, intercept) into pixel endpoints (x1, y1, x2, y2)."""
    slope, intercept = line_params
    if slope == 0:
        slope = 0.0001  # avoid division by zero for a near-flat fit

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Step 7: Draw results
# ---------------------------------------------------------------------------
def draw_lines(frame, lines, color=(0, 0, 255), thickness=10):
    """Draw the given lines on a transparent overlay, then blend it with frame."""
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    return cv2.addWeighted(frame, 0.8, line_image, 1.0, 0)


def draw_roi_outline(frame, polygon, color=(0, 255, 255), thickness=2):
    """Draw the ROI polygon boundary on the frame (useful for debugging/demo)."""
    outlined = frame.copy()
    cv2.polylines(outlined, polygon, isClosed=True, color=color, thickness=thickness)
    return outlined


# ---------------------------------------------------------------------------
# Full pipeline for a single frame (used for both images and video frames)
# ---------------------------------------------------------------------------
def process_frame(frame, show_roi=False):
    """
    Run the complete lane detection pipeline on a single BGR frame and
    return the frame with detected lane lines drawn on it.
    """
    gray = to_grayscale(frame)
    blurred = apply_gaussian_blur(gray)
    edges = detect_edges(blurred)
    masked_edges, roi_polygon = region_of_interest(edges)
    hough_lines = detect_hough_lines(masked_edges)
    lane_lines = average_slope_intercept(frame, hough_lines)
    result = draw_lines(frame, lane_lines)

    if show_roi:
        result = draw_roi_outline(result, roi_polygon)

    return result
