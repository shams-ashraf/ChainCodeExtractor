import cv2
import numpy as np


def get_chain_code(contour, sector_size):
    """Extracts chain code from the given contour using 8-connectivity and outputs it in a formatted way."""
    directions = []
    points_accessed = []

    direction_map = {
        (1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
        (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7
    }

    prev = contour[0][0]  # Starting point in pixels
    prev_sector = (prev[1] // sector_size, prev[0] // sector_size)  # Convert to sector coordinates
    points_accessed.append(prev_sector)

    for point in contour[1:]:
        curr = point[0]
        move = (curr[0] - prev[0], curr[1] - prev[1])

        if move in direction_map:
            directions.append(direction_map[move])

        curr_sector = (curr[1] // sector_size, curr[0] // sector_size)
        if curr_sector != prev_sector:
            points_accessed.append(curr_sector)  # Store accessed sector coordinates

        prev = curr
        prev_sector = curr_sector

    formatted_chain_code = " ".join(map(str, directions))  # Format output

    return formatted_chain_code, [(int(row), int(col)) for row, col in points_accessed]


# Automatically set the image path
image_path = "image2.png"  # Change this to your actual image filename

# Load the image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Error: Could not open the image. Ensure it's a valid image file.")

# Convert to binary (black & white)
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Define sector size based on image dimensions
h, w = binary_image.shape
sector_size_x = max(w // 10, 1)  # Avoid division by zero
sector_size_y = max(h // 10, 1)

if contours:
    for i, contour in enumerate(contours):
        chain_code, accessed_sectors = get_chain_code(contour, sector_size_x)

        print(f"\nContour {i + 1}:")
        print("Chain Code:", chain_code)
        print("Accessed Sectors (Row, Column):", accessed_sectors)
