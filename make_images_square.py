import os
from PIL import Image, ImageOps

def add_whitespace_to_square(image_path, output_path, color=(255, 255, 255)):
    # Open the image
    image = Image.open(image_path)
    
    # Get the dimensions of the image
    width, height = image.size
    
    # Determine the size of the new square image
    max_dim = max(width, height)
    
    # Create a new image with a white background
    new_image = Image.new("RGB", (max_dim, max_dim), color)
    
    # Calculate the position to paste the original image onto the new image
    paste_position = ((max_dim - width) // 2, (max_dim - height) // 2)
    
    # Paste the original image onto the new image
    new_image.paste(image, paste_position)
    
    # Save the new image
    new_image.save(output_path)

def process_images_in_folder(input_folder, output_folder, color=(255, 255, 255)):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)
            add_whitespace_to_square(input_image_path, output_image_path, color)

# Example usage
input_folder = 'images/'  # Replace with your input folder path
output_folder = 'images_square/'  # Replace with your desired output folder path
process_images_in_folder(input_folder, output_folder)