from PIL import Image
import os

def compress_image(input_file_path, output_file_path, quality=20):
    # Open an image file
    with Image.open(input_file_path) as img:
        # Compress the image
        img.save(output_file_path, "JPEG", quality=quality)

if __name__ == "__main__":
    input_path = "path/to/your/input/image.jpg"  # Change this to your input image path
    output_path = "path/to/your/output/image_compressed.jpg"  # Change this to your desired output path
    compress_image(input_path, output_path)