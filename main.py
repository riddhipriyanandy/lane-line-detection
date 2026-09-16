"""
main.py

Command-line entry point for the lane detection project.

Usage:
    python main.py --input input/test_road.jpg
    python main.py --input input/test_road.jpg --show-roi
    python main.py --input input/test_road.mp4

The script auto-detects whether the input is an image or a video based
on its file extension, runs the lane detection pipeline (lane_detection.py)
on it, and saves the annotated result to the output/ folder.
"""

import argparse
import os
import sys

import cv2

from lane_detection import process_frame

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def process_image(input_path, output_path, show_roi, display):
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Error: could not read image '{input_path}'. Check the file path/format.")
        sys.exit(1)

    result = process_frame(frame, show_roi=show_roi)

    cv2.imwrite(output_path, result)
    print(f"Saved annotated image to: {output_path}")

    if display:
        # Only attempted when --display is passed, since a live window
        # needs a desktop environment and is not available on headless
        # machines/servers (e.g. college lab PCs run over SSH, CI graders).
        cv2.imshow("Lane Detection - Result", result)
        print("Press any key on the image window to close it...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(input_path, output_path, show_roi, display):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: could not open video '{input_path}'. Check the file path/format.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(frame, show_roi=show_roi)
        writer.write(result)
        frame_count += 1

        if display:
            cv2.imshow("Lane Detection - Video", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Interrupted by user (q pressed).")
                break

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames.")
    print(f"Saved annotated video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Automatic Road Lane Line Detection using Canny Edge Detection and Hough Transform"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to an input road image or video (e.g. input/test_road.jpg)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save the annotated output. Defaults to output/<input_filename>."
    )
    parser.add_argument(
        "--show-roi", action="store_true",
        help="Draw the Region of Interest boundary on the output for debugging/demo purposes."
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Also open a live preview window (requires a desktop/GUI environment)."
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' does not exist.")
        sys.exit(1)

    os.makedirs("output", exist_ok=True)
    ext = os.path.splitext(args.input)[1].lower()
    output_path = args.output or os.path.join("output", os.path.basename(args.input))

    if ext in IMAGE_EXTENSIONS:
        process_image(args.input, output_path, args.show_roi, args.display)
    elif ext in VIDEO_EXTENSIONS:
        process_video(args.input, output_path, args.show_roi, args.display)
    else:
        print(f"Error: unsupported file extension '{ext}'. "
              f"Supported images: {IMAGE_EXTENSIONS}, videos: {VIDEO_EXTENSIONS}")
        sys.exit(1)


if __name__ == "__main__":
    main()
