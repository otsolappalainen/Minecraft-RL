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


def remove_alpha_channel(image):
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    return image

def compute_mean_std(image_dir, transform):
    mean = 0.0
    std = 0.0
    count = 0
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, img_name)
            image = Image.open(img_path)
            image = remove_alpha_channel(image)
            image = transform(image)
            image = image.unsqueeze(0)
            mean += image.mean([0, 2, 3])
            std += image.std([0, 2, 3])
            count += 1
    if count == 0:
        raise ValueError(f"No images found in {image_dir}")
    mean /= count
    std /= count
    return mean.tolist(), std.tolist()

def get_model_path():
    # Use same path structure as train_ppo_v5.py
    BASE_DIR = Path(__file__).parent.parent  # Go up one level from script location
    MODELS_DIR = BASE_DIR / "models"
    PPO_MODELS_DIR = MODELS_DIR / "ppo" / "v5"
    
    # Ensure directory exists
    if not PPO_MODELS_DIR.exists():
        raise ValueError(f"Models directory not found: {PPO_MODELS_DIR}")
    
    # Get model files
    models = [f for f in os.listdir(PPO_MODELS_DIR) if f.endswith('.zip')]
    
    if not models:
        raise ValueError("No models found in directory")
    
    print("\nAvailable models:")
    for idx, name in enumerate(models, 1):
        print(f"{idx}. {name}")
    
    while True:
        try:
            choice = int(input("\nSelect model number: ")) - 1
            if 0 <= choice < len(models):
                return str(PPO_MODELS_DIR / models[choice])
            print("Invalid selection")
        except ValueError:
            print("Please enter a number")



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
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
        ])
        
        print("Computing normalization parameters...")
        self.mean, self.std = compute_mean_std(image_dir, self.transform)
        print(f"Mean: {self.mean}")
        print(f"Std: {self.std}")
        
        self.transform = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
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
        if activations is None:
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

def main():
    try:
        model_path = get_model_path()
        
        # Use consistent base directory structure
        BASE_DIR = Path(__file__).parent.parent
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = BASE_DIR / "visualizations" / "cnn" / f"run_{timestamp}"
        image_dir = BASE_DIR / "test_images"
        
        # Add heatmap overlay strength parameter
        HEATMAP_ALPHA = 0.7  # Adjust this value between 0-1
        
        os.makedirs(save_dir, exist_ok=True)
        
        if not image_dir.exists():
            print(f"Image directory {image_dir} not found!")
            return
            
        print("Initializing visualizer...")
        visualizer = CNNVisualizer(model_path, image_dir, force_cpu=True)
        
        print("Visualizing CNN filters...")
        visualizer.visualize_filters(save_dir)
        
        print(f"Processing images from {image_dir}...")
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for i, image_file in enumerate(image_files, 1):
            print(f"Processing image {i}/{len(image_files)}: {image_file}")
            image_path = os.path.join(image_dir, image_file)
            img_save_dir = os.path.join(save_dir, os.path.splitext(image_file)[0])
            os.makedirs(img_save_dir, exist_ok=True)
            
            visualizer.visualize_feature_maps(image_path, img_save_dir)
            visualizer.create_heatmap(image_path, img_save_dir, HEATMAP_ALPHA)
        
        print(f"Visualizations saved to {save_dir}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
