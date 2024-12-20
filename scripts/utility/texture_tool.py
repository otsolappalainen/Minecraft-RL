import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageEnhance, ImageTk

class TextureEditor:
    def __init__(self):
        self.window = tk.Tk()
        self.window.withdraw()  # Hide main window initially
        
        # Image state
        self.original_image = None
        self.current_image = None
        self.photo_image = None
        self.scale_factor = 20
        
        # Output directory
        self.output_dir = r"C:\Users\odezz\Desktop\mine textures"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initial image selection
        if self.load_initial_image():
            self.window.deiconify()  # Show window only if image loaded
            self.window.title("Texture Editor")
            self.setup_ui()
        else:
            self.window.destroy()
            return
            
    def load_initial_image(self):
        image_path = filedialog.askopenfilename(
            title="Select texture to edit",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
        )
        
        if not image_path:
            messagebox.showerror("Error", "No image selected. Exiting.")
            return False
            
        try:
            self.original_image = Image.open(image_path)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            return False
    
    def setup_ui(self):
        # Control Frame
        control_frame = tk.Frame(self.window)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Sliders
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        
        tk.Label(control_frame, text="Brightness:").pack()
        tk.Scale(control_frame, from_=0.0, to=2.0, resolution=0.1,
                variable=self.brightness_var, orient=tk.HORIZONTAL,
                command=self.update_image).pack()
                
        tk.Label(control_frame, text="Contrast:").pack()
        tk.Scale(control_frame, from_=0.0, to=2.0, resolution=0.1,
                variable=self.contrast_var, orient=tk.HORIZONTAL,
                command=self.update_image).pack()
        
        # Buttons
        tk.Button(control_frame, text="Load Different Image", command=self.load_different_image).pack(pady=5)
        tk.Button(control_frame, text="Save Image", command=self.save_image).pack(pady=5)
        
        # Preview Frame
        self.preview_frame = tk.Label(self.window)
        self.preview_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Show initial image
        self.update_image()
        
    def load_different_image(self):
        if self.load_initial_image():
            self.update_image()
            
    def update_image(self, *args):
        if self.original_image is None:
            return
            
        # Apply adjustments
        img = self.original_image.copy()
        
        # Brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(self.brightness_var.get())
        
        # Contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.contrast_var.get())
        
        self.current_image = img
        
        # Scale up for preview
        width, height = img.size
        scaled = img.resize(
            (width * self.scale_factor, height * self.scale_factor),
            Image.NEAREST
        )
        
        # Convert for Tkinter
        self.photo_image = ImageTk.PhotoImage(scaled)
        self.preview_frame.config(image=self.photo_image)
        
    def save_image(self):
        if self.current_image is None:
            return
            
        file_name = "processed_texture.png"
        output_path = os.path.join(self.output_dir, file_name)
        self.current_image.save(output_path, "PNG")
        print(f"Saved to: {output_path}")
        
    def run(self):
        if self.original_image:  # Only run if image was loaded
            self.window.mainloop()

if __name__ == "__main__":
    editor = TextureEditor()
    editor.run()