# Project Statement

## Title

Automatic Road Lane Line Detection using Canny Edge Detection and Hough
Transform

## Problem Statement

Lane detection is a foundational task in road-scene understanding and
Advanced Driver Assistance Systems (ADAS). Given a photo or video frame
from a forward-facing road camera, the goal is to automatically identify
and highlight the boundaries of the current driving lane, without any
manual annotation.

This project implements a lightweight, classical computer vision
pipeline that detects the left and right lane markings in a road image
or video using **Canny edge detection** and the **Hough Line Transform**
— two well-established, non-learning-based techniques from image
processing.

## Scope

This project is intentionally scoped as a small, self-contained,
academic demonstration of a classical CV pipeline. It is:

- Limited to **2D straight-line lane detection** on images/short videos
  with a forward-facing camera angle.
- Built entirely with **OpenCV and NumPy** — no deep learning frameworks,
  no pretrained/external models, no large datasets.
- A single-purpose command-line script, not a production system, web
  app, or long-term deployment pipeline.

**Out of scope:**
- Curved-lane modeling (polynomial/spline fitting)
- Multi-lane or lane-change detection
- Vehicle, pedestrian, or traffic-sign detection
- Real-time performance guarantees on embedded hardware
- Any form of vehicle control or autonomous decision-making

This is a **lane-line detection system**, not a complete autonomous
driving or ADAS system.

## Target Users

- Undergraduate Computer Science / AI-ML students learning classical
  image processing techniques (edge detection, Hough transforms, ROI
  masking).
- Anyone wanting a minimal, readable reference implementation of a
  classical lane-detection pipeline, e.g. for a college mini-project or
  viva demonstration.

## High-Level Features

- Accepts a road **image** or **video** as input via a simple CLI.
- Grayscale conversion and Gaussian blurring for noise reduction.
- Edge detection using the Canny algorithm.
- A configurable trapezoidal **Region of Interest (ROI)** mask focused
  on the road area.
- Line detection using the Probabilistic Hough Line Transform.
- Slope-based separation of raw line segments into left/right lane
  groups, followed by averaging and extrapolation into two clean lane
  lines.
- Overlay of the final detected lane lines on the original frame.
- Saves annotated output (image or video) to an `output/` folder.
- Optional ROI-boundary overlay for debugging and demonstration
  purposes.
