import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== 1. Model Definition (Same as training) ====================
class CLAHETransform:
    """Apply CLAHE for contrast enhancement on grayscale medical images"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        enhanced = clahe.apply(img_np)
        return transforms.functional.to_pil_image(enhanced)

class ResNetWithDropout(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.4):
        super(ResNetWithDropout, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        
        # Remove original FC layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Custom classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.resnet(x)
        return self.classifier(features)

# ==================== 2. GradCAM Implementation ====================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling on gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # [H, W]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), target_class, output

# ==================== 3. Visualization Functions ====================
def apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET):
    """
    Apply heatmap on image
    """
    # Resize activation map to match image size
    heatmap = cv2.resize(activation_map, (org_img.shape[1], org_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Convert to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap on image
    superimposed = cv2.addWeighted(org_img, 0.6, heatmap, 0.4, 0)
    
    return heatmap, superimposed

def create_visualization(original_img, cam, prediction, confidence, class_names, save_path):
    """
    Create comprehensive visualization with original, heatmap, and overlay
    """
    # Convert PIL to numpy if needed
    if isinstance(original_img, Image.Image):
        original_img = np.array(original_img)
    
    # Ensure RGB
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    elif original_img.shape[2] == 4:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)
    
    # Generate heatmap and overlay
    heatmap, superimposed = apply_colormap_on_image(original_img, cam)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap)
    axes[1].set_title('GradCAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(superimposed)
    axes[2].set_title(f'Prediction: {class_names[prediction]}\nConfidence: {confidence:.2f}%', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return superimposed

# ==================== 4. Prediction Pipeline ====================
def predict_with_gradcam(image_path, model, transform, gradcam, class_names):
    """
    Make prediction and generate GradCAM
    """
    # Load and preprocess image
    original_img = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_img).unsqueeze(0).to(device)
    
    # Generate GradCAM
    cam, predicted_class, output = gradcam.generate_cam(input_tensor)
    
    # Get confidence
    probabilities = F.softmax(output, dim=1)
    confidence = probabilities[0, predicted_class].item() * 100
    
    return original_img, cam, predicted_class, confidence

# ==================== 5. Batch Processing ====================
def process_folder(input_folder, output_folder, model_path, class_names):
    """
    Process all images in a folder and save GradCAM visualizations
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = ResNetWithDropout(num_classes=len(class_names), dropout_rate=0.4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Initialize GradCAM
    target_layer = model.resnet.layer4[-1]  # Last layer of ResNet layer4
    gradcam = GradCAM(model, target_layer)
    
    # Define transforms (same as validation)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
        image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process...")
    
    # Process each image
    results = []
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Generate prediction and GradCAM
            original_img, cam, prediction, confidence = predict_with_gradcam(
                str(img_path), model, transform, gradcam, class_names
            )
            
            # Create output filename
            output_filename = f"{img_path.stem}_gradcam.png"
            output_path = os.path.join(output_folder, output_filename)
            
            # Create and save visualization
            create_visualization(original_img, cam, prediction, confidence, class_names, output_path)
            
            # Store results
            results.append({
                'filename': img_path.name,
                'prediction': class_names[prediction],
                'confidence': confidence
            })
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            continue
    
    # Save results summary
    results_path = os.path.join(output_folder, 'predictions_summary.txt')
    with open(results_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PREDICTION SUMMARY\n")
        f.write("="*70 + "\n\n")
        for result in results:
            f.write(f"Image: {result['filename']}\n")
            f.write(f"  Prediction: {result['prediction']}\n")
            f.write(f"  Confidence: {result['confidence']:.2f}%\n")
            f.write("-"*70 + "\n")
    
    print(f"\n✅ Processing complete!")
    print(f"📁 GradCAM visualizations saved to: {output_folder}")
    print(f"📄 Predictions summary saved to: {results_path}")
    print(f"📊 Processed {len(results)}/{len(image_files)} images successfully")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("PREDICTION DISTRIBUTION")
    print("="*70)
    from collections import Counter
    pred_counts = Counter([r['prediction'] for r in results])
    for class_name, count in pred_counts.items():
        percentage = (count / len(results)) * 100
        print(f"{class_name:15s}: {count:3d} ({percentage:5.1f}%)")
    print("="*70)

# ==================== 6. Main Execution ====================
if __name__ == '__main__':
    # Configuration
    INPUT_FOLDER = 'D:/ml/brain/testmain'  # Folder containing images to predict
    OUTPUT_FOLDER = 'D:/ml/brain/gradcam_results'  # Folder to save GradCAM results
    MODEL_PATH = 'best_model.pth'  # Path to trained model
    CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']  # Your class names
    
    # Create input folder if it doesn't exist (for testing)
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
        print("Please train the model first using main.py")
        exit(1)
    
    # Check if input folder has images
    if not any(Path(INPUT_FOLDER).iterdir()):
        print(f"❌ Error: No images found in '{INPUT_FOLDER}'")
        print(f"Please add images to this folder and run again.")
        exit(1)
    
    # Process all images
    print("="*70)
    print("🔬 BRAIN TUMOR GRADCAM PREDICTOR")
    print("="*70)
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Model: {MODEL_PATH}")
    print(f"Classes: {CLASS_NAMES}")
    print("="*70 + "\n")
    
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, MODEL_PATH, CLASS_NAMES)