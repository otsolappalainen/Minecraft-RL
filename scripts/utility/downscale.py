import tkinter as tk
from tkinter import filedialog
import cv2
import os
from pathlib import Path

def main():
    # Hide root window
    root = tk.Tk()
    root.withdraw()

    # Create output directory
    output_dir = Path("downscaled_images")
    output_dir.mkdir(exist_ok=True)

    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select image to downscale",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp")
        ]
    )

    if not file_path:
        print("No file selected")
        return

    # Load and downscale image
    img = cv2.imread(file_path)
    if img is None:
        print("Failed to load image")
        return

    height, width = img.shape[:2]
    new_size = (width//2, height//2)
    downscaled = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)

    # Create output filename
    original_name = Path(file_path).stem
    output_path = output_dir / f"{original_name}_downscaled.png"

    # Save downscaled image
    cv2.imwrite(str(output_path), downscaled)
    print(f"Saved downscaled image to: {output_path}")

if __name__ == "__main__":
    main()