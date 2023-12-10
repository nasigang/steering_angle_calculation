# steering_angle_calculation

## Lane Detection using OpenCV

This Python script implements a lane detection algorithm using OpenCV for real-time video processing.

### Features

* **Color and Edge Detection:**
    * Converts input image to HLS color space for improved lane visibility.
    * Applies thresholding and Gaussian blur to capture lane edge information.
* **Region of Interest Masking:**
    * Focuses processing on relevant areas of the image (e.g., bottom half of the road).
* **Perspective Transformation (Bird's Eye View):**
    * Warps the image to a bird's eye view for improved lane line detection.
* **Sliding Window Method:**
    * Identifies lane lines based on pixel intensity within sliding windows.
* **Curve Fitting and Off-center Calculation:**
    * Estimates lane curvature and vehicle position relative to the lane center.
* **Moving Average:**
    * Smooths out lane detection results for increased accuracy.
* **Video Input and Output:**
    * Reads video frames, processes them, and outputs the final results.

### Usage

1. Install OpenCV and numpy libraries.
2. Download the provided Python script and video file (images/LaneToSteeringChallenge.mp4).
3. Run the script using the following command:

```python
python lane_detection.py


4. The script will process the video and generate a new video (images/LaneToSteeringChallenge.avi) with lane lines and curvature information overlaid.

### Requirements

* Python 3.x
* OpenCV library
* numpy library

### Disclaimer

This script is for educational purposes only and should not be used in real-world self-driving applications. It may not be robust to various lighting conditions, road markings, and other environmental factors.
