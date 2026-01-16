# prediction/predict_skin.py
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import os

# ========== DEVICE CONFIGURATION ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[predict_skin] Using device: {device}")

# ========== MODEL ARCHITECTURE ==========
class TransferLearningClassifier(nn.Module):
    """
    ResNet-50 based transfer learning classifier for skin cancer detection
    Classes: basal cell carcinoma, melanoma, nevus, normal, pigmented benign keratosis
    """
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super(TransferLearningClassifier, self).__init__()
        # Load pretrained ResNet-50 backbone
        self.backbone = models.resnet50(pretrained=False)
        
        # Get number of features from backbone
        num_features = self.backbone.fc.in_features  # 2048 for ResNet-50
        
        # Replace final fully connected layer with custom head
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ========== IMAGE PREPROCESSING ==========
# Standard ImageNet normalization for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ========== CLASS DEFINITIONS ==========
CLASS_NAMES = [
    "basal cell carcinoma",
    "melanoma",
    "nevus",
    "normal",
    "pigmented benign keratosis"
]

# Additional information about each class
CLASS_INFO = {
    "basal cell carcinoma": {
        "full_name": "Basal Cell Carcinoma",
        "description": "Most common type of skin cancer, slow-growing and rarely spreads",
        "severity": "Moderate",
        "recommendation": "Consult a dermatologist for evaluation and treatment options"
    },
    "melanoma": {
        "full_name": "Melanoma",
        "description": "Most dangerous type of skin cancer, can spread rapidly if untreated",
        "severity": "High",
        "recommendation": "URGENT: Seek immediate dermatologist consultation"
    },
    "nevus": {
        "full_name": "Nevus (Mole)",
        "description": "Common benign skin growth, usually harmless",
        "severity": "Benign",
        "recommendation": "Monitor for changes; routine check-up if it changes in size, color, or shape"
    },
    "normal": {
        "full_name": "Normal Skin",
        "description": "Healthy skin tissue without concerning features",
        "severity": "None",
        "recommendation": "Continue regular skin self-examinations"
    },
    "pigmented benign keratosis": {
        "full_name": "Pigmented Benign Keratosis",
        "description": "Non-cancerous skin growth, common in older adults",
        "severity": "Benign",
        "recommendation": "Generally harmless; consult doctor if irritated or changing"
    }
}

# ========== MODEL LOADING ==========
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_skin_classifier_transfer.pth"

def load_model():
    """
    Load the trained skin cancer classification model
    Returns: loaded model or None if loading fails
    """
    if not MODEL_PATH.exists():
        print(f"[predict_skin] ❌ Model not found at: {MODEL_PATH}")
        print(f"[predict_skin] Expected path: {MODEL_PATH.absolute()}")
        return None
    
    try:
        # Initialize model with same architecture as training
        model = TransferLearningClassifier(num_classes=5, dropout_rate=0.5).to(device)
        
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
        
        print(f"[predict_skin] ✓ Model loaded successfully from: {MODEL_PATH}")
        return model
        
    except Exception as e:
        print(f"[predict_skin] ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

# Load model once when module is imported
_skin_model = load_model()

# ========== PREDICTION FUNCTION ==========
def predict_image(image_path):
    """
    Predict skin lesion type from dermoscopic image
    
    Args:
        image_path (str): Path to the skin lesion image file
    
    Returns:
        dict: Prediction results
            {
                "success": bool,
                "predicted_class": str,
                "confidence": float,
                "all_probabilities": dict,
                "class_info": dict,
                "error": str (only if success=False)
            }
    """
    # Check if model is loaded
    if _skin_model is None:
        return {
            "success": False,
            "error": f"Model file not found. Please ensure 'best_skin_classifier_transfer.pth' exists in the models folder at: {MODEL_PATH}"
        }
    
    # Check if image exists
    if not os.path.exists(image_path):
        return {
            "success": False,
            "error": f"Image file not found: {image_path}"
        }
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = _skin_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get predicted class
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_percent = float(confidence.item() * 100.0)
        
        # Get all class probabilities
        all_probs = probabilities.cpu().numpy()[0]
        prob_dict = {
            CLASS_NAMES[i]: float(all_probs[i] * 100.0)
            for i in range(len(CLASS_NAMES))
        }
        
        # Sort probabilities by confidence
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": round(confidence_percent, 2),
            "all_probabilities": sorted_probs,
            "class_info": CLASS_INFO.get(predicted_class, {}),
            "disease_info": CLASS_INFO.get(predicted_class, {}),  # Alias for compatibility
            "raw_outputs": outputs.cpu().numpy().tolist()
        }
        
    except Exception as e:
        print(f"[predict_skin] Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Prediction error: {str(e)}"
        }

# ========== BATCH PREDICTION (OPTIONAL) ==========
def predict_batch(image_paths):
    """
    Predict multiple images at once
    
    Args:
        image_paths (list): List of image file paths
    
    Returns:
        list: List of prediction dictionaries
    """
    results = []
    for img_path in image_paths:
        results.append(predict_image(img_path))
    return results

# ========== TESTING ==========
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Testing Skin Cancer Classification Model")
    print(f"{'='*60}\n")
    
    # Print model info
    print(f"Model path: {MODEL_PATH}")
    print(f"Model loaded: {'Yes' if _skin_model is not None else 'No'}")
    print(f"Device: {device}")
    print(f"Classes: {CLASS_NAMES}")
    
    # Test with a sample image if available
    test_image = BASE_DIR / "static" / "uploads" / "test_skin.jpg"
    
    if test_image.exists():
        print(f"\nTesting with image: {test_image}")
        result = predict_image(str(test_image))
        
        if result["success"]:
            print(f"\n✓ Prediction: {result['predicted_class']}")
            print(f"✓ Confidence: {result['confidence']:.2f}%")
            print(f"✓ Severity: {result['class_info'].get('severity', 'Unknown')}")
            print(f"\nAll Probabilities:")
            for disease, prob in result['all_probabilities'].items():
                print(f"  {disease}: {prob:.2f}%")
            print(f"\nRecommendation: {result['class_info'].get('recommendation', 'N/A')}")
        else:
            print(f"\n✗ Error: {result['error']}")
    else:
        print(f"\nNo test image found at: {test_image}")
        print("Upload an image through the web interface to test.")