import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from stable_baselines3 import PPO
import warnings
import datetime
from matplotlib.colors import LinearSegmentedColormap
import cv2
from pathlib import Path
import re

# Update VIDEO_CONFIG with better codec settings
VIDEO_CONFIG = {
    "FPS": 15,
    "INPUT_HEIGHT": 120,
    "INPUT_WIDTH": 240,
    "OUTPUT_HEIGHT": 480,
    "OUTPUT_WIDTH": 960,
    "CODEC": "mp4v",  # Changed from avc1 to more widely supported mp4v
    "BITRATE": "5000k"
}

def remove_alpha_channel(image):
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    return image


def get_model_path():
    import tkinter as tk
    from tkinter import filedialog
    
    # Initialize tkinter root
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    # Set initial directory
    initial_dir = Path(r"E:\PPO_BC_MODELS\models_ppo_240")
    
    # Open file dialog
    model_path = filedialog.askopenfilename(
        title="Select Model File",
        initialdir=initial_dir,
        filetypes=[("ZIP files", "*.zip")]
    )
    
    if not model_path:
        raise ValueError("No model selected")
        
    return model_path

class CNNVisualizer:
    def __init__(self, model_path, image_dir, force_cpu=True):
        # Load model
        if force_cpu:
            self.device = torch.device("cpu")
            model = PPO.load(model_path, device=self.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = PPO.load(model_path)
        
        # Extract only CNN layers
        self.cnn = model.policy.features_extractor.img_head.to(self.device)
        self.cnn.eval()
        
        # Simple transform matching env normalization
        self.transform = transforms.Compose([
            transforms.Resize((VIDEO_CONFIG['INPUT_HEIGHT'], VIDEO_CONFIG['INPUT_WIDTH'])),
            transforms.ToTensor(),  # This divides by 255 automatically
        ])

    def get_feature_maps(self, image_path):
        try:
            image = Image.open(image_path)
            image = remove_alpha_channel(image)
            image = self.transform(image).unsqueeze(0)
            image = image.to(self.device)
            
            activations = []
            
            def hook_fn(module, input, output):
                activations.append(output.detach())
            
            hooks = []
            # Add hooks for conv layers
            for module in self.cnn.modules():
                if isinstance(module, nn.Conv2d):
                    hooks.append(module.register_forward_hook(hook_fn))
            
            with torch.no_grad():
                _ = self.cnn(image)
            
            for hook in hooks:
                hook.remove()
                
            return activations
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def visualize_feature_maps(self, image_path, save_dir):
        activations = self.get_feature_maps(image_path)
        if (activations is None):
            return
            
        layer_names = ['conv1_16', 'conv2_32', 'conv3_64', 'conv4_32']
        
        for i, (layer_activations, layer_name) in enumerate(zip(activations, layer_names)):
            n_features = layer_activations.shape[1]
            size = int(np.ceil(np.sqrt(n_features)))
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig, axes = plt.subplots(size, size, figsize=(12, 12))
                fig.suptitle(f'{layer_name} Features', fontsize=16)
                
                if size == 1:
                    axes = np.array([[axes]])
                elif len(axes.shape) == 1:
                    axes = axes.reshape(size, size)
                
                for idx in range(n_features):
                    ax = axes[idx//size, idx%size]
                    feature_map = layer_activations[0, idx].cpu().numpy()
                    ax.imshow(feature_map, cmap='viridis')
                    ax.axis('off')
                
                for idx in range(n_features, size*size):
                    axes[idx//size, idx%size].axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(save_dir, f'{layer_name}_features.png')
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()

    def visualize_filters(self, save_dir):
        for i, module in enumerate(self.cnn.modules()):
            if isinstance(module, nn.Conv2d):
                filters = module.weight.data.cpu()
                n_filters = filters.shape[0]
                size = int(np.ceil(np.sqrt(n_filters)))
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fig, axes = plt.subplots(size, size, figsize=(12, 12))
                    
                    if size == 1:
                        axes = np.array([[axes]])
                    elif len(axes.shape) == 1:
                        axes = axes.reshape(size, size)
                    
                    for idx in range(n_filters):
                        ax = axes[idx//size, idx%size]
                        filter_img = filters[idx].mean(0)
                        ax.imshow(filter_img.numpy(), cmap='viridis')
                        ax.axis('off')
                    
                    for idx in range(n_filters, size*size):
                        axes[idx//size, idx%size].axis('off')
                    
                    plt.tight_layout()
                    save_path = os.path.join(save_dir, f'conv_layer_{i}_filters.png')
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.close()

    def create_heatmap(self, image_path, save_dir, overlay_alpha=0.7):
        """
        Create heatmap from final layer activations and overlay on original image
        Args:
            overlay_alpha: Float 0-1, higher means more heatmap visible
        """
        activations = self.get_feature_maps(image_path)
        if activations is None:
            return
            
        # Get final layer activations
        final_layer = activations[-1]  # Shape: [1, 32, H, W]
        
        # Combine all feature maps and normalize
        heatmap = final_layer.mean(dim=1)[0]  # Average across channels
        heatmap = heatmap.cpu().numpy()
        
        # Normalize to 0-1
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / heatmap.max()
        
        # Resize heatmap to original image size
        original_image = Image.open(image_path)
        original_image = remove_alpha_channel(original_image)
        heatmap = cv2.resize(heatmap, original_image.size)
        
        # Create colormap
        colors = [(0, 0, 0, 0), (1, 0, 0, 1)]  # Transparent to red
        cmap = LinearSegmentedColormap.from_list('custom', colors)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.axis('off')
        plt.title('Original Image')
        
        # Plot overlay
        plt.subplot(1, 2, 2)
        plt.imshow(original_image)
        plt.imshow(heatmap, cmap=cmap, alpha=overlay_alpha)
        plt.axis('off')
        plt.title('Activation Heatmap Overlay')
        
        # Save
        save_path = os.path.join(save_dir, 'heatmap_overlay.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def process_batch(visualizer, image_files):
    """Process all images and find global min/max for normalization"""
    results = []
    total = len(image_files)
    global_min = float('inf')
    global_max = float('-inf')
    
    # First pass - collect data and find global min/max
    for idx, img_path in enumerate(image_files, 1):
        print(f"\rProcessing image {idx}/{total}", end="")
        
        original = Image.open(img_path)
        original = remove_alpha_channel(original)
        original = np.array(original)
        
        activations = visualizer.get_feature_maps(str(img_path))
        if activations is None:
            continue
            
        final_layer = activations[-1]
        heatmap = final_layer.mean(dim=1)[0].cpu().numpy()
        
        # Update global min/max
        global_min = min(global_min, heatmap.min())
        global_max = max(global_max, heatmap.max())
        
        results.append({
            'original': original,
            'activations': activations,
            'heatmap': heatmap,  # Store raw values
            'path': img_path
        })
    
    print("\nBatch processing complete!")
    return results, global_min, global_max

# Modify create_side_by_side_video function
def create_side_by_side_video(results, save_path, config=VIDEO_CONFIG, global_min=None, global_max=None):
    output_height = config['OUTPUT_HEIGHT']
    output_width = config['OUTPUT_WIDTH']
    
    # Try different codec options in order of preference
    codec_options = [
        ('mp4v', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi')
    ]
    
    out = None
    for codec, ext in codec_options:
        try:
            save_path = str(save_path).rsplit('.', 1)[0] + ext
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(
                save_path,
                fourcc,
                config['FPS'],
                (output_width * 2, output_height),
                isColor=True
            )
            if out.isOpened():
                print(f"Successfully initialized video writer with codec: {codec}")
                break
        except Exception as e:
            print(f"Failed to initialize codec {codec}: {e}")
            continue
    
    if out is None or not out.isOpened():
        raise RuntimeError("Could not initialize any video codec")
        
    for idx, result in enumerate(results, 1):
        print(f"\rCreating frame {idx}/{len(results)}", end="")
        
        # Resize maintaining aspect ratio
        original = cv2.resize(result['original'], 
                            (output_width, output_height), 
                            interpolation=cv2.INTER_NEAREST)
        original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        
        # Use global normalization
        heatmap = result['heatmap']
        heatmap = cv2.resize(heatmap, (output_width, output_height), 
                           interpolation=cv2.INTER_NEAREST)
        
        # Scale to 0-255 using global min/max
        heatmap = ((heatmap - global_min) / (global_max - global_min) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Blend heatmap with original
        overlay = cv2.addWeighted(original, 0.7, heatmap, 0.3, 0)
        
        # Combine side by side
        combined = np.hstack([original, overlay])
        out.write(combined)
    
    print("\nVideo creation complete!")
    out.release()


def create_feature_map_videos(results, save_dir, config=VIDEO_CONFIG):
    """Creates feature map videos from pre-processed results"""
    layer_names = ['conv1_16', 'conv2_32', 'conv3_64', 'conv4_32']
    
    # Initialize video writers
    writers = {}
    first_features = results[0]['activations']
    
    for i, (layer_acts, layer_name) in enumerate(zip(first_features, layer_names)):
        n_features = layer_acts.shape[1]
        size = int(np.ceil(np.sqrt(n_features)))
        # Adjust cell size based on output height
        cell_size_h = config['OUTPUT_HEIGHT']//4  
        cell_size_w = config['OUTPUT_WIDTH']//4
        grid_size = (size * cell_size_w, size * cell_size_h)
        
        save_path = save_dir / f"{layer_name}_features.mp4"
        fourcc = cv2.VideoWriter_fourcc(*config['CODEC'])
        writer = cv2.VideoWriter(str(save_path), fourcc, config['FPS'], 
                               grid_size, isColor=True)
            
        writers[layer_name] = writer
    
    # Process each result
    for frame_idx, result in enumerate(results, 1):
        print(f"\rCreating feature maps frame {frame_idx}/{len(results)}", end="")
        
        for layer_acts, layer_name in zip(result['activations'], layer_names):
            n_features = layer_acts.shape[1]
            size = int(np.ceil(np.sqrt(n_features)))
            cell_size_h = config['OUTPUT_HEIGHT']//4
            cell_size_w = config['OUTPUT_WIDTH']//4
            
            # Create feature grid
            grid = np.zeros((size * cell_size_h, size * cell_size_w))
            for feat_idx in range(n_features):
                row = feat_idx // size
                col = feat_idx % size
                feature_map = layer_acts[0, feat_idx].cpu().numpy()
                feature_map = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX)
                # Upscale each feature map
                feature_map = cv2.resize(feature_map, (cell_size_w, cell_size_h), 
                                      interpolation=cv2.INTER_LANCZOS4)
                grid[row*cell_size_h:(row+1)*cell_size_h, 
                     col*cell_size_w:(col+1)*cell_size_w] = feature_map
            
            grid = cv2.applyColorMap(grid.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            writers[layer_name].write(grid)
        
    
    print("\nFeature map videos complete!")
    for writer in writers.values():
        writer.release()
    
    print("\nFeature map videos complete!")
    for writer in writers.values():
        writer.release()

def sort_images_by_timestamp(image_dir):
    """Sort images by timestamp in filename"""
    def extract_timestamp(filename):
        # Extract timestamp from format: step_XXXXXX_portXXXX_TIMESTAMP.png
        match = re.search(r'step_\d+_port\d+_(\d+)', filename)
        if match:
            return int(match.group(1))  # Use actual timestamp for sorting
        return 0
    
    image_files = [f for f in image_dir.iterdir() 
                   if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    return sorted(image_files, key=lambda x: extract_timestamp(x.name))

def create_quad_video(results, save_path, config=VIDEO_CONFIG, global_min=None, global_max=None):
    output_height = config['OUTPUT_HEIGHT'] // 2
    output_width = config['OUTPUT_WIDTH'] // 2
    
    codec_options = [
        ('mp4v', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi')
    ]
    
    out = None
    for codec, ext in codec_options:
        try:
            save_path = str(save_path).rsplit('.', 1)[0] + ext
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(
                save_path,
                fourcc,
                config['FPS'],
                (output_width * 2, output_height * 2),
                isColor=True
            )
            if out.isOpened():
                print(f"Successfully initialized video writer with codec: {codec}")
                break
        except Exception as e:
            print(f"Failed to initialize codec {codec}: {e}")
            continue
    
    if out is None or not out.isOpened():
        raise RuntimeError("Could not initialize any video codec")
        
    for idx, result in enumerate(results, 1):
        print(f"\rCreating frame {idx}/{len(results)}", end="")
        
        # 1. Original Image
        original = cv2.resize(result['original'], 
                            (output_width, output_height), 
                            interpolation=cv2.INTER_NEAREST)
        original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        
        # 2. Standard Heatmap
        heatmap = result['heatmap']
        heatmap = cv2.resize(heatmap, (output_width, output_height), 
                           interpolation=cv2.INTER_NEAREST)
        heatmap = ((heatmap - global_min) / (global_max - global_min) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        standard_overlay = cv2.addWeighted(original, 0.7, heatmap_color, 0.3, 0)
        
        # 3. Lanczos4 Heatmap
        heatmap_lanczos = cv2.resize(result['heatmap'], 
                                   (output_width, output_height), 
                                   interpolation=cv2.INTER_LANCZOS4)
        heatmap_lanczos = ((heatmap_lanczos - global_min) / (global_max - global_min) * 255).astype(np.uint8)
        heatmap_lanczos_color = cv2.applyColorMap(heatmap_lanczos, cv2.COLORMAP_JET)
        lanczos_overlay = cv2.addWeighted(original, 0.7, heatmap_lanczos_color, 0.3, 0)
        
        # 4. Hot-area Filter
        # Normalize heatmap to 0-1 for mask
        mask = ((heatmap_lanczos / 255) > 0.5).astype(np.float32)  # Threshold at 0.5
        mask = np.expand_dims(mask, axis=2)  # Add channel dimension
        mask = np.repeat(mask, 3, axis=2)  # Repeat for RGB
        hot_area_view = original * mask
        
        # Combine into 2x2 grid
        top_row = np.hstack([original, standard_overlay])
        bottom_row = np.hstack([lanczos_overlay, hot_area_view])
        combined = np.vstack([top_row, bottom_row])
        
        out.write(combined)
    
    print("\nVideo creation complete!")
    out.release()

def create_transparent_overlay_video(results, save_path, config=VIDEO_CONFIG, global_min=None, global_max=None):
    output_height = config['OUTPUT_HEIGHT']
    output_width = config['OUTPUT_WIDTH']
    
    codec_options = [
        ('mp4v', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi')
    ]
    
    out = None
    for codec, ext in codec_options:
        try:
            save_path = str(save_path).rsplit('.', 1)[0] + ext
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(
                save_path,
                fourcc,
                config['FPS'],
                (output_width * 2, output_height),
                isColor=True
            )
            if out.isOpened():
                print(f"Successfully initialized video writer with codec: {codec}")
                break
        except Exception as e:
            print(f"Failed to initialize codec {codec}: {e}")
            continue
    
    if out is None or not out.isOpened():
        raise RuntimeError("Could not initialize any video codec")
        
    for idx, result in enumerate(results, 1):
        print(f"\rCreating frame {idx}/{len(results)}", end="")
        
        # Original Image
        original = cv2.resize(result['original'], 
                            (output_width, output_height), 
                            interpolation=cv2.INTER_NEAREST)
        original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        
        # Create transparency mask from heatmap
        heatmap = result['heatmap']
        heatmap = cv2.resize(heatmap, (output_width, output_height), 
                           interpolation=cv2.INTER_NEAREST)
        
        # Normalize heatmap to 0-1
        mask = (heatmap - global_min) / (global_max - global_min)
        
        # Create black background
        black_bg = np.zeros_like(original)
        
        # Blend based on heatmap values
        masked = original * mask[..., None] + black_bg * (1 - mask[..., None])
        masked = masked.astype(np.uint8)
        
        # Combine side by side
        combined = np.hstack([original, masked])
        out.write(combined)
    
    print("\nVideo creation complete!")
    out.release()

def create_stacked_comparison_video(results, save_path, config=VIDEO_CONFIG, global_min=None, global_max=None):
    def get_side_by_side_frame(result, output_height, output_width):
        # Top row: Original | Colored heatmap overlay
        original = cv2.resize(result['original'], 
                            (output_width, output_height), 
                            interpolation=cv2.INTER_NEAREST)
        original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        
        heatmap = result['heatmap']
        heatmap = cv2.resize(heatmap, (output_width, output_height), 
                           interpolation=cv2.INTER_NEAREST)
        heatmap = ((heatmap - global_min) / (global_max - global_min) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.7, heatmap_color, 0.3, 0)
        return np.hstack([original, overlay])

    def get_bottom_row_frame(result, output_height, output_width):
        # Left side: Pure heatmap on black background
        heatmap = result['heatmap']
        heatmap = cv2.resize(heatmap, (output_width, output_height), 
                           interpolation=cv2.INTER_NEAREST)
        heatmap = ((heatmap - global_min) / (global_max - global_min) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Right side: Transparent overlay version
        original = cv2.resize(result['original'], 
                            (output_width, output_height), 
                            interpolation=cv2.INTER_NEAREST)
        original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        
        # Create black background
        black_bg = np.zeros_like(original)
        
        # Normalize heatmap to 0-1
        mask = (heatmap - global_min) / (global_max - global_min)
        
        # Blend based on heatmap values
        masked = original * mask[..., None] + black_bg * (1 - mask[..., None])
        masked = masked.astype(np.uint8)
        
        return np.hstack([heatmap_color, masked])

    # Setup video writer
    output_height = config['OUTPUT_HEIGHT'] // 2  # Half height for each row
    output_width = config['OUTPUT_WIDTH']
    
    codec_options = [
        ('mp4v', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi')
    ]
    
    out = None
    for codec, ext in codec_options:
        try:
            save_path = str(save_path).rsplit('.', 1)[0] + ext
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(
                save_path,
                fourcc,
                config['FPS'],
                (output_width * 2, output_height * 2),  # Full size for 2x2 grid
                isColor=True
            )
            if out.isOpened():
                print(f"Successfully initialized video writer with codec: {codec}")
                break
        except Exception as e:
            print(f"Failed to initialize codec {codec}: {e}")
            continue
    
    if out is None or not out.isOpened():
        raise RuntimeError("Could not initialize any video codec")

    # Create frames
    for idx, result in enumerate(results, 1):
        print(f"\rCreating frame {idx}/{len(results)}", end="")
        
        # Get both rows
        top_row = get_side_by_side_frame(result, output_height, output_width)
        bottom_row = get_bottom_row_frame(result, output_height, output_width)
        
        # Stack them vertically
        combined = np.vstack([top_row, bottom_row])
        out.write(combined)
    
    print("\nVideo creation complete!")
    out.release()

def create_calibration_window(image_path, visualizer, global_min, global_max):
    def on_trackbar(x):
        # Get current values
        threshold = cv2.getTrackbarPos('Threshold', 'Calibration') / 100
        contrast = cv2.getTrackbarPos('Contrast', 'Calibration') / 10
        
        # Process image with current values
        original = cv2.imread(str(image_path))
        original = cv2.resize(original, (VIDEO_CONFIG['OUTPUT_WIDTH']//2, VIDEO_CONFIG['OUTPUT_HEIGHT']//2))
        
        activations = visualizer.get_feature_maps(str(image_path))
        heatmap = activations[-1].mean(dim=1)[0].cpu().numpy()
        heatmap = cv2.resize(heatmap, (VIDEO_CONFIG['OUTPUT_WIDTH']//2, VIDEO_CONFIG['OUTPUT_HEIGHT']//2))
        
        # Normalize and apply mask
        mask = (heatmap - global_min) / (global_max - global_min)
        mask = 1 / (1 + np.exp(-contrast * (mask - threshold)))
        
        # Create visualization
        black_bg = np.zeros_like(original)
        masked = original * mask[..., None] + black_bg * (1 - mask[..., None])
        masked = masked.astype(np.uint8)
        
        # Show side by side
        combined = np.hstack([original, masked])
        cv2.imshow('Calibration', combined)

    # Create window
    cv2.namedWindow('Calibration')
    
    # Create trackbars (threshold 0-100 represents 0-1, contrast 0-100 represents 0-10)
    cv2.createTrackbar('Threshold', 'Calibration', 50, 100, on_trackbar)
    cv2.createTrackbar('Contrast', 'Calibration', 10, 100, on_trackbar)
    
    # Initial render
    on_trackbar(0)
    
    print("\nPress 'S' to save values and exit")
    print("Press 'ESC' to exit without saving")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return None, None
        elif key == ord('s'):
            threshold = cv2.getTrackbarPos('Threshold', 'Calibration') / 100
            contrast = cv2.getTrackbarPos('Contrast', 'Calibration') / 10
            cv2.destroyAllWindows()
            return threshold, contrast

def main():
    try:
        # Ask for mode
        print("\nSelect mode:")
        print("1. Calibrate parameters")
        print("2. Process videos")
        mode = input("Enter choice (1/2): ").strip()
        
        model_path = get_model_path()
        image_dir = Path(r"C:\Users\odezz\source\Minecraft-RL\scripts\PPO\env_screenshots")
        save_dir = Path(r"E:\PPO_BC_MODELS\visualizations")
        
        # Initialize visualizer
        print("Initializing visualizer...")
        visualizer = CNNVisualizer(model_path, str(image_dir), force_cpu=True)
        
        # Get image files
        image_files = sort_images_by_timestamp(image_dir)
        
        if mode == "1":
            # Calibration mode
            print("Calibration mode selected...")
            
            # Process one random image for calibration
            random_image = np.random.choice(image_files)
            activations = visualizer.get_feature_maps(str(random_image))
            heatmap = activations[-1].mean(dim=1)[0].cpu().numpy()
            global_min, global_max = heatmap.min(), heatmap.max()
            
            # Open calibration window
            threshold, contrast = create_calibration_window(
                random_image, visualizer, global_min, global_max)
                
            if threshold is None:
                print("Calibration cancelled")
                return
                
            print(f"\nCalibrated values: threshold={threshold:.2f}, contrast={contrast:.2f}")
            return
            
        # Normal processing mode
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = save_dir / "videos" / f"run_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # ...rest of main() code...
        if not image_dir.exists():
            print(f"Image directory {image_dir} not found!")
            return
            
        print("Initializing visualizer...")
        visualizer = CNNVisualizer(model_path, str(image_dir), force_cpu=True)
        
        # Sort images by timestamp
        image_files = sort_images_by_timestamp(image_dir)
        print(f"Found {len(image_files)} images")
        
        # Process all images first with global normalization
        print("Processing images...")
        results, global_min, global_max = process_batch(visualizer, image_files)
        
        # Create side by side video with global normalization
        print("Creating side-by-side comparison video...")
        side_by_side_path = save_dir / "side_by_side.mp4"
        create_side_by_side_video(results, side_by_side_path, global_min=global_min, global_max=global_max)
        
        # Create feature map videos
        #print("Creating feature map videos...")
        #create_feature_map_videos(results, save_dir)
        
        # Replace quad video with stacked comparison
        print("Creating stacked comparison video...")
        stacked_path = save_dir / "stacked_comparison.mp4"
        create_stacked_comparison_video(
            results, 
            stacked_path, 
            global_min=global_min, 
            global_max=global_max,
            #mask_threshold=0.4,  # Adjust these values to control transparency
            #mask_contrast=7.5    # Higher = sharper transition
        )
        
        # Create transparent overlay video
        print("Creating transparent overlay video...")
        transparent_path = save_dir / "transparent_overlay.mp4"
        create_transparent_overlay_video(results, transparent_path, global_min=global_min, global_max=global_max)
        
        print(f"Videos saved to {save_dir}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()