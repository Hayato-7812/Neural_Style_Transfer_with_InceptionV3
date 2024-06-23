import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import io
import os
import cv2
from pytorch_fid import fid_score
import torch
import random
import matplotlib.pyplot as plt

# Define paths
image_folder = 'images/'
temp_content_folder = 'temp_content_images/'
temp_stylized_folders = {
    'vgg': 'temp_stylized_images_vgg/',
    'inception': 'temp_stylized_images_inception/',
    'resnet': 'temp_stylized_images_resnet/'
}

# Ensure the temporary folders exist
os.makedirs(temp_content_folder, exist_ok=True)
for folder in temp_stylized_folders.values():
    os.makedirs(folder, exist_ok=True)

# Define transformations
def load_image(image_path):
    try:
        image = io.imread(image_path)
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        image = cv2.resize(image, (224, 224))  # Resize images to 224x224
        image = image / 255.0  # Normalize to [0, 1] range
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Calculate SSIM and PSNR
def calculate_ssim_psnr(content_img, stylized_img):
    if content_img is None or stylized_img is None:
        raise ValueError("One of the images is None. Check the paths and loading process.")
    
    # SSIM
    ssim_value = ssim(content_img, stylized_img, data_range=1.0, channel_axis=-1)

    # PSNR
    mse = np.mean((content_img - stylized_img) ** 2)
    if mse == 0:
        psnr_value = 100
    else:
        PIXEL_MAX = 1.0
        psnr_value = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    return ssim_value, psnr_value

# Function to display images
def display_images(images, titles, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()
    for img, title, ax in zip(images, titles, axes):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Aggregate results
results = {'vgg': {'ssim': [], 'psnr': [], 'fid': []},
           'inception': {'ssim': [], 'psnr': [], 'fid': []},
           'resnet': {'ssim': [], 'psnr': [], 'fid': []}}

# Save paths for FID calculation
content_image_paths = []

for i in range(1, 7):
    content_image_path = os.path.join(image_folder, f'{i}c.jpg')
    content_image = load_image(content_image_path)
    if content_image is None:
        print(f"Skipping content image: {content_image_path}")
        continue
    
    temp_content_path = os.path.join(temp_content_folder, f'{i}c_temp.jpg')
    io.imsave(temp_content_path, (content_image * 255).astype(np.uint8))
    content_image_paths.append(temp_content_path)

    for model_name, stylized_folder in temp_stylized_folders.items():
        stylized_image_path = os.path.join(image_folder, f'{i}o_{model_name}.jpg')
        stylized_image = load_image(stylized_image_path)
        if stylized_image is None:
            print(f"Skipping stylized image: {stylized_image_path}")
            continue

        try:
            ssim_value, psnr_value = calculate_ssim_psnr(content_image, stylized_image)
            results[model_name]['ssim'].append(ssim_value)
            results[model_name]['psnr'].append(psnr_value)
            
            temp_stylized_path = os.path.join(stylized_folder, f'{i}o_{model_name}_temp.jpg')
            io.imsave(temp_stylized_path, (stylized_image * 255).astype(np.uint8))
        except Exception as e:
            print(f"Error calculating metrics for {content_image_path} and {stylized_image_path}: {e}")

# Display a random set of images to verify loading
random_indices = random.sample(range(1, 7), 3)
images_to_display = []
titles = []
for i in random_indices:
    content_image_path = os.path.join(image_folder, f'{i}c.jpg')
    content_image = load_image(content_image_path)
    if content_image is not None:
        images_to_display.append(content_image)
        titles.append(f'Content {i}')

    for model_name in temp_stylized_folders.keys():
        stylized_image_path = os.path.join(image_folder, f'{i}o_{model_name}.jpg')
        stylized_image = load_image(stylized_image_path)
        if stylized_image is not None:
            images_to_display.append(stylized_image)
            titles.append(f'Stylized {i} ({model_name})')

# Display the images
display_images(images_to_display, titles, rows=3, cols=4)

# Calculate FID for each model separately
for model_name, stylized_folder in temp_stylized_folders.items():
    if content_image_paths and os.listdir(stylized_folder):  # Ensure there are images to compare
        fid_value = fid_score.calculate_fid_given_paths(
            [temp_content_folder, stylized_folder],
            batch_size=6,  # Adjusted batch size
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dims=2048,  # Dimension of Inception v3 features
            num_workers=0  # Set num_workers to 0
        )
        results[model_name]['fid'].append(fid_value)

# Compute averages
average_results = {model_name: {} for model_name in results.keys()}
for model_name, metrics in results.items():
    for metric_name, values in metrics.items():
        if values:  # Check if there are any values to average
            average_results[model_name][metric_name] = np.mean(values)
        else:
            average_results[model_name][metric_name] = float('nan')

# Display results
for model_name, metrics in average_results.items():
    print(f"Results for {model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# Cleanup temporary files
for folder in [temp_content_folder] + list(temp_stylized_folders.values()):
    for file in os.listdir(folder):
        os.remove(os.path.join(folder, file))