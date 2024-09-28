import cv2
import sys

def visualize_detection(screen_path, template_path, found_coords):
    """
    Draw a rectangle on the screen image at the found coordinates to verify the match.
    
    Parameters:
        screen_path (str): Path to the screen image.
        template_path (str): Path to the template image.
        found_coords (tuple): The (x, y) coordinates where the template was found.
    """
    # Load the original screen image in color
    screen_image = cv2.imread(screen_path)
    if screen_image is None:
        print(f"Error: Unable to load screen image from {screen_path}")
        return
    
    # Load the template to get its size
    template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template_image is None:
        print(f"Error: Unable to load template image from {template_path}")
        return

    # Get template dimensions
    t_height, t_width = template_image.shape

    # Get the top-left coordinates of the found match
    x, y = found_coords

    # Draw a rectangle around the detected area
    cv2.rectangle(screen_image, (x, y), (x + t_width, y + t_height), (0, 255, 0), 2)

    # Save the result
    result_path = "result_visualization.png"
    cv2.imwrite(result_path, screen_image)
    print(f"Verification image saved at: {result_path}")

if __name__ == "__main__":
    # Check if the correct number of arguments are passed
    if len(sys.argv) != 5:
        print("Usage: python visualize_detection.py <screen_path> <template_path> <found_x> <found_y>")
        sys.exit(1)

    # Retrieve arguments from the command line
    screen_path = sys.argv[1]
    template_path = sys.argv[2]
    found_x = int(sys.argv[3])
    found_y = int(sys.argv[4])

    # Call the function with the provided arguments
    visualize_detection(screen_path, template_path, (found_x, found_y))
