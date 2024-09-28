
"""
image_search.py

This script searches for a template image within a larger screen image using OpenCV.
It's optimized for pixelated graphics, such as those found in Roblox games, through
preprocessing techniques like resizing and edge detection.

Usage:
    python image_search.py <template_path> <screen_path> [threshold] [min_scale] [max_scale] [scale_step]

Arguments:
    template_path (str): Path to the template image (e.g., 'template.png').
    screen_path (str): Path to the screen image where the template will be searched.
    threshold (float, optional): Matching threshold between 0 and 1 (default is 0.6).
    min_scale (float, optional): Minimum scale to use in multi-scale search (default is 0.3).
    max_scale (float, optional): Maximum scale to use in multi-scale search (default is 2.0).
    scale_step (float, optional): Step size for scaling (default is 0.05).

Output:
    Prints the coordinates of the top-left corner where the template is found,
    or 'Not Found' if no match meets the threshold.
"""

import cv2
import numpy as np
import sys
import os

def preprocess_image(image_path):
    """
    Preprocess the image by applying edge detection.

    Parameters:
        image_path (str): The path to the image file to preprocess.

    Returns:
        numpy.ndarray: The preprocessed image.
    """
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at '{image_path}'.")
        return None

    # Apply Canny edge detection
    edges = cv2.Canny(image, 50, 150)
    
    return edges

def multi_scale_template_matching(screen_path, template_path, min_scale=0.3, max_scale=2.0, scale_step=0.05, threshold=0.6):
    """
    Perform multi-scale template matching to handle differences in size between the template and screen image.
    
    Parameters:
        screen_path (str): Path to the screen image.
        template_path (str): Path to the template image.
        min_scale (float): The minimum scale factor to try.
        max_scale (float): The maximum scale factor to try.
        scale_step (float): The step size for scaling.
        threshold (float): Matching threshold between 0 and 1.
        
    Returns:
        tuple or None: Coordinates of the found template, or None if not found.
    """
    # Preprocess images
    processed_screen = preprocess_image(screen_path)
    processed_template = preprocess_image(template_path)

    if processed_template is None or processed_screen is None:
        print("Error: One or more images could not be processed.")
        return None

    # Initialize variables to track the best match
    best_match_val = -np.inf
    best_match_coords = None

    # Iterate over the specified range of scales
    for scale in np.arange(min_scale, max_scale, scale_step):
        # Resize the template image to the current scale
        resized_template = cv2.resize(processed_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # Get the size of the resized template
        t_height, t_width = resized_template.shape[:2]

        # Ensure the resized template can fit within the screen image
        if t_height > processed_screen.shape[0] or t_width > processed_screen.shape[1]:
            continue

        # Perform template matching
        result = cv2.matchTemplate(processed_screen, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Check if this match is better than the previous best match
        if max_val > best_match_val and max_val >= threshold:
            best_match_val = max_val
            best_match_coords = max_loc

    # Report the best match
    if best_match_coords:
        x, y = best_match_coords
        print(f"Found at: {x},{y} with match value: {best_match_val}")
        return (x, y)
    else:
        print("Not Found")
        return None

if __name__ == "__main__":
    # Ensure the correct number of arguments
    if len(sys.argv) < 3:
        print("Usage: python image_search.py <template_path> <screen_path> [threshold] [min_scale] [max_scale] [scale_step]")
        sys.exit(1)

    # Parse command-line arguments
    template_path = sys.argv[1]
    screen_path = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    min_scale = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    max_scale = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
    scale_step = float(sys.argv[6]) if len(sys.argv) > 6 else 0.05

    # Validate file paths
    if not os.path.isfile(template_path):
        print(f"Error: Template file '{template_path}' does not exist.")
        sys.exit(1)

    if not os.path.isfile(screen_path):
        print(f"Error: Screen file '{screen_path}' does not exist.")
        sys.exit(1)

    # Validate the numeric inputs
    if not (0 <= threshold <= 1):
        print("Error: Threshold must be between 0 and 1.")
        sys.exit(1)

    if min_scale <= 0 or max_scale <= 0 or scale_step <= 0:
        print("Error: Scale values must be greater than 0.")
        sys.exit(1)

    # Perform multi-scale template matching
    multi_scale_template_matching(screen_path, template_path, min_scale, max_scale, scale_step, threshold)


