# prediction/predict_brain.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import os
import cv2
import numpy as np
import base64
from io import BytesIO

# ========== DEVICE CONFIGURATION ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[predict_brain] Using device: {device}")

# ========== CLAHE TRANSFORM ==========
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

# ========== MODEL ARCHITECTURE ==========
class ResNetWithDropout(nn.Module):
    """Brain tumor classifier with dropout - matches your training code"""
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

# ========== GRADCAM IMPLEMENTATION ==========
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
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
        
        return cam.cpu().numpy(), target_class

# ========== IMAGE PREPROCESSING ==========
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

# ========== CLASS DEFINITIONS ==========
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no tumor",
    "pituitary"
]

CLASS_INFO = {
    "glioma": {
        "full_name": "Glioma",
        "description": "Most common malignant brain tumor arising from glial cells",
        "severity": "High"
    },
    "meningioma": {
        "full_name": "Meningioma",
        "description": "Usually benign tumor arising from the meninges",
        "severity": "Moderate"
    },
    "no tumor": {
        "full_name": "No Tumor Detected",
        "description": "Normal brain tissue without detectable tumors",
        "severity": "None"
    },
    "pituitary": {
        "full_name": "Pituitary Tumor",
        "description": "Tumor in the pituitary gland, usually benign",
        "severity": "Moderate"
    }
}

# ========== MODEL LOADING ==========
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model_brain.pth"

def load_model():
    """Load the trained brain tumor classification model"""
    if not MODEL_PATH.exists():
        print(f"[predict_brain] ❌ Model not found at: {MODEL_PATH}")
        return None, None
    
    try:
        # Initialize model with correct architecture
        model = ResNetWithDropout(num_classes=4, dropout_rate=0.4).to(device)
        
        # Load saved weights
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        # Set to evaluation mode
        model.eval()
        
        # Initialize GradCAM with layer4 of resnet
        target_layer = model.resnet.layer4[-1]
        gradcam = GradCAM(model, target_layer)
        
        print(f"[predict_brain] ✓ Model loaded successfully from: {MODEL_PATH}")
        return model, gradcam
        
    except Exception as e:
        print(f"[predict_brain] ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Load model once when module is imported
_brain_model, _gradcam = load_model()

# ========== GRADCAM VISUALIZATION ==========
def apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET):
    """Apply heatmap on image"""
    # Resize activation map to match image size
    heatmap = cv2.resize(activation_map, (org_img.shape[1], org_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Convert to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap on image
    superimposed = cv2.addWeighted(org_img, 0.6, heatmap, 0.4, 0)
    
    return heatmap, superimposed

def generate_gradcam_image(original_img, cam):
    """Generate GradCAM visualization as base64 string"""
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
    
    # Convert to base64
    superimposed_pil = Image.fromarray(superimposed)
    buffered = BytesIO()
    superimposed_pil.save(buffered, format="PNG")
    gradcam_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{gradcam_base64}"

# ========== PREDICTION FUNCTION ==========
def predict_image(image_path):
    """
    Predict brain tumor type from MRI image with GradCAM
    
    Args:
        image_path (str): Path to the MRI image file
    
    Returns:
        dict: Prediction results with GradCAM visualization
    """
    # Check if model is loaded
    if _brain_model is None or _gradcam is None:
        return {
            "success": False,
            "error": f"Model file not found. Please ensure 'best_model_brain.pth' exists at: {MODEL_PATH}"
        }
    
    # Check if image exists
    if not os.path.exists(image_path):
        return {
            "success": False,
            "error": f"Image file not found: {image_path}"
        }
    
    try:
        # Load original image
        original_img = Image.open(image_path).convert('RGB')
        
        # Preprocess image
        image_tensor = transform(original_img).unsqueeze(0).to(device)
        
        # Generate GradCAM
        cam, predicted_idx = _gradcam.generate_cam(image_tensor)
        
        # Make prediction
        with torch.no_grad():
            outputs = _brain_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence = probabilities[0, predicted_idx].item()
        
        # Get predicted class
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence_percent = float(confidence * 100.0)
        
        # Get all class probabilities
        all_probs = probabilities.cpu().numpy()[0]
        prob_dict = {
            CLASS_NAMES[i]: float(all_probs[i] * 100.0)
            for i in range(len(CLASS_NAMES))
        }
        
        # Sort probabilities by confidence
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
        
        # Generate GradCAM visualization
        gradcam_image = generate_gradcam_image(original_img, cam)
        
        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": round(confidence_percent, 2),
            "all_probabilities": sorted_probs,
            "class_info": CLASS_INFO.get(predicted_class, {}),
            "gradcam_image": gradcam_image,
            "raw_outputs": outputs.cpu().numpy().tolist()
        }
        
    except Exception as e:
        print(f"[predict_brain] Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Prediction error: {str(e)}"
        }

# ========== BATCH PREDICTION (OPTIONAL) ==========
def predict_batch(image_paths):
    """Predict multiple images at once"""
    results = []
    for img_path in image_paths:
        results.append(predict_image(img_path))
    return results

# ========== TESTING ==========
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Testing Brain Tumor Classification Model")
    print(f"{'='*60}\n")
    
    print(f"Model path: {MODEL_PATH}")
    print(f"Model loaded: {'Yes' if _brain_model is not None else 'No'}")
    print(f"GradCAM loaded: {'Yes' if _gradcam is not None else 'No'}")
    print(f"Device: {device}")
    print(f"Classes: {CLASS_NAMES}")
    
    test_image = BASE_DIR / "static" / "uploads" / "test_brain.jpg"
    
    if test_image.exists():
        print(f"\nTesting with image: {test_image}")
        result = predict_image(str(test_image))
        
        if result["success"]:
            print(f"\n✓ Prediction: {result['predicted_class']}")
            print(f"✓ Confidence: {result['confidence']:.2f}%")
            print(f"\nAll Probabilities:")
            for disease, prob in result['all_probabilities'].items():
                print(f"  {disease}: {prob:.2f}%")
            print(f"\n✓ GradCAM generated: Yes")
        else:
            print(f"\n✗ Error: {result['error']}")
    else:
        print(f"\nNo test image found at: {test_image}")