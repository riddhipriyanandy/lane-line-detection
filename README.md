# Automatic Road Lane Line Detection using Canny Edge Detection and Hough Transform

## Overview

This project detects road lane lines in an image (or video) using classical
computer vision techniques — **no deep learning or pretrained models are
used**. Given a forward-facing road photo/video, the program:

1. Converts the frame to grayscale.
2. Smooths it with a Gaussian blur to reduce noise.
3. Detects edges using the **Canny Edge Detector**.
4. Masks the edge image to a **Region of Interest (ROI)** — the trapezoid
   of the frame where the road is expected to be.
5. Detects straight line segments in that region using the
   **Probabilistic Hough Line Transform**.
6. Separates the segments into a left lane and a right lane based on
   their slope, averages each group, and extrapolates a single line per
   side.
7. Draws the final lane lines on top of the original frame and saves the
   result.

This is a classical, rule-based pipeline. It is **a lane-line detector,
not a complete autonomous driving / lane-keeping system** — it does not
do vehicle detection, path planning, or control.

## Features

- Works on a single road **image**.
- Also works on a short road **video** (processed frame-by-frame).
- Fully configurable via a simple command-line interface.
- Optional overlay of the Region of Interest boundary for debugging/demo.
- Modular code: each pipeline step is a separate, well-commented function.
- No GUI framework, no web server, no database — just a script.

## Technologies / Tools Used

- Python 3.9+
- [OpenCV](https://opencv.org/) (`opencv-python`) — image/video I/O, Canny,
  Hough Transform, drawing
- [NumPy](https://numpy.org/) — array operations and line-fitting (`polyfit`)

## Project Structure

```text
lane-line-detection/
│
├── README.md                  # This file
├── statement.md                # Problem statement & project scope
├── requirements.txt            # Python dependencies
├── main.py                     # CLI entry point (image/video handling)
├── lane_detection.py           # Core CV pipeline (all processing steps)
├── generate_sample_input.py    # Creates a synthetic test image (optional)
├── input/                      # Put your road image/video here
│   └── test_road.jpg           # Small synthetic sample image (included)
├── output/                     # Annotated results are saved here
└── screenshots/                # Example output screenshots for the README
```

## Installation

### 1. Prerequisites

- Python 3.9 or newer installed
- `pip` available on your PATH

Check your Python version:

```bash
python3 --version
```

### 2. Get the project

```bash
git clone <your-repo-url>
cd lane-line-detection
```

(Or simply download/unzip the project folder.)

### 3. (Recommended) Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run the Project

A small synthetic sample image is already included at
`input/test_road.jpg`, so you can run the project immediately:

```bash
python main.py --input input/test_road.jpg
```

The annotated output is saved to `output/test_road.jpg`.

### Useful options

| Option        | Description                                                              |
|---------------|---------------------------------------------------------------------------|
| `--input`     | Path to the input image or video (required)                              |
| `--output`    | Custom path to save the result (defaults to `output/<input filename>`)   |
| `--show-roi`  | Also draws the yellow Region-of-Interest boundary on the output          |
| `--display`   | Opens a live preview window (only works if you have a desktop/GUI)       |

Example with all options:

```bash
python main.py --input input/test_road.jpg --output output/result.jpg --show-roi --display
```

### Providing your own input

- **Image:** Drop a `.jpg`/`.jpeg`/`.png`/`.bmp` road photo into `input/`
  and pass its path with `--input`. Prefer a forward-facing shot where
  the road takes up the lower half of the frame (similar to a dashcam).
- **Video:** Drop a short `.mp4`/`.avi`/`.mov`/`.mkv` clip into `input/`
  and run the same command — `main.py` auto-detects video files and
  processes them frame by frame, saving an annotated video to `output/`.

```bash
python main.py --input input/my_video.mp4
```

> **Tip:** If your video/image resolution or camera angle is very
> different from a typical dashcam view, the ROI polygon in
> `lane_detection.py` (`region_of_interest` function) may need small
> tweaks — see the *Limitations* section below.

## (Optional) Regenerating the sample input

The included `input/test_road.jpg` was generated synthetically so the
project works out of the box without needing an external dataset. To
regenerate it (or see how it was made):

```bash
python generate_sample_input.py
```

## Testing Instructions

Since this is a small CV script (not a library), "testing" is done by
running it end-to-end and visually verifying the output:

1. Run the pipeline on the included sample:
   ```bash
   python main.py --input input/test_road.jpg --show-roi
   ```
2. Open `output/test_road.jpg` and confirm:
   - The yellow ROI trapezoid roughly outlines the road area.
   - A red line is drawn over (or very close to) each visible lane
     marking.
3. Try it on 2–3 of your own road photos taken from roughly eye level,
   facing forward, and confirm the lane lines are picked up reasonably
   well.
4. (Optional) Try a short (5–10 second) road video clip and confirm the
   output video plays with lane lines tracked across frames.

There are no automated unit tests — the pipeline is short enough that
manual visual verification is the standard and expected way to validate
it for a project of this scope.

## Expected Output

- For an **image** input: an annotated `.jpg`/`.png` in `output/` with
  the detected lane lines drawn in red on top of the original photo.
- For a **video** input: an annotated `.mp4` in `output/` with lane
  lines drawn on every processed frame.
- Console output reporting where the result was saved (and, for videos,
  how many frames were processed).

See `screenshots/sample_output.jpg` for an example of the annotated
output on the included sample image.

## Limitations

- This is a **classical, rule-based** approach — not a learned model.
  It assumes reasonably visible, high-contrast lane markings.
- The **Region of Interest** is a fixed trapezoid defined as a fraction
  of the frame size. It works well for a typical forward-facing
  dashcam-style shot but may need manual adjustment for very different
  camera angles/positions.
- Performance can degrade in poor conditions: heavy shadows, faded or
  missing lane paint, rain/glare, sharp curves, or lane changes.
- Only straight (or near-straight) lane segments are modeled; the
  extrapolated lines are a linear fit, so they do not follow strongly
  curved roads.
- This project detects lane lines only — it does **not** perform
  vehicle/pedestrian detection, path planning, or any driving decision.
  It is **not** a complete autonomous-driving or ADAS system.

## Screenshots

- `screenshots/sample_output.jpg` — Result on the included synthetic
  sample image (`input/test_road.jpg`), run with `--show-roi`.
