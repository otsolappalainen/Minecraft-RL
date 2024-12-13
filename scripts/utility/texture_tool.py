import os
from tkinter import Tk, filedialog
from PIL import Image, ImageEnhance

def balance_contrast(image_path, output_dir):
    # Open the image
    with Image.open(image_path) as img:
        # Convert to grayscale (if necessary)
        if img.mode != 'L':
            img = img.convert('L')

        # Apply contrast balancing
        # Here, we reduce the contrast slightly to balance dark and light areas
        enhancer = ImageEnhance.Contrast(img)
        balanced_image = enhancer.enhance(0.3)  # Reduce contrast (factor < 1)

        # Save the new image in the specified directory
        file_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, file_name)
        balanced_image.save(output_path)
        print(f"Processed image saved at: {output_path}")

def main():
    # Directory to save the modified images
    output_dir = r"C:\\Users\\odezz\\Desktop\\mine textures"
    os.makedirs(output_dir, exist_ok=True)

    # Create a file dialog to select an image
    Tk().withdraw()  # Hide the main tkinter window
    image_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
    )

    if not image_path:
        print("No image selected. Exiting.")
        return

    # Process the selected image
    try:
        balance_contrast(image_path, output_dir)
    except Exception as e:
        print(f"An error occurred while processing the image: {e}")

if __name__ == "__main__":
    main()
