import os
import cv2

# Constants
INPUT_FOLDER = r"C:\Users\odezz\source\MinecraftAI2\scripts\gen7\test_images"
OUTPUT_FOLDER = r"C:\Users\odezz\source\MinecraftAI2\scripts\gen7\resized_images"
IMAGE_WIDTH = 120
IMAGE_HEIGHT = 120

# Define interpolation methods
INTERPOLATION_METHODS = {
    'INTER_NEAREST': cv2.INTER_NEAREST,
    'INTER_LINEAR': cv2.INTER_LINEAR,
    'INTER_CUBIC': cv2.INTER_CUBIC,
    'INTER_LANCZOS4': cv2.INTER_LANCZOS4
}

def create_output_directories(output_base, methods):
    """
    Create subdirectories for each interpolation method.
    """
    for method in methods:
        method_dir = os.path.join(output_base, method)
        os.makedirs(method_dir, exist_ok=True)
        print(f"Checked/Created directory: {method_dir}")

def is_image_file(filename):
    """
    Check if a file is an image based on its extension.
    """
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
    return filename.lower().endswith(IMAGE_EXTENSIONS)

def resize_and_save_image(image_path, output_base, methods, width, height):
    """
    Resize an image using different interpolation methods and save them.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Unable to read image: {image_path}")
            return

        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)

        for method_name, method in methods.items():
            resized_img = cv2.resize(img, (width, height), interpolation=method)
            output_dir = os.path.join(output_base, method_name)
            output_path = os.path.join(output_dir, f"{name}_{method_name}{ext}")
            cv2.imwrite(output_path, resized_img)
            print(f"Saved resized image to: {output_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def main():
    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"Input folder does not exist: {INPUT_FOLDER}")
        return

    # Create output directories
    create_output_directories(OUTPUT_FOLDER, INTERPOLATION_METHODS.keys())

    # List all files in the input directory
    files = os.listdir(INPUT_FOLDER)
    image_files = [f for f in files if is_image_file(f)]

    if not image_files:
        print("No image files found in the input directory.")
        return

    print(f"Found {len(image_files)} image(s) to process.")

    # Process each image
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(INPUT_FOLDER, image_file)
        print(f"\nProcessing image {idx}/{len(image_files)}: {image_path}")
        resize_and_save_image(image_path, OUTPUT_FOLDER, INTERPOLATION_METHODS, IMAGE_WIDTH, IMAGE_HEIGHT)

    print("\nImage resizing completed successfully.")

if __name__ == "__main__":
    main()