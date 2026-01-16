import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Transfer Learning Model (same as training)
class TransferLearningClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5, freeze_layers=True):
        super(TransferLearningClassifier, self).__init__()
        
        self.backbone = models.resnet50(pretrained=False)
        num_features = self.backbone.fc.in_features
        
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

# Image preprocessing (same as test transform)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names (must match training order)
CLASS_NAMES = [
    'basal cell carcinoma',
    'melanoma',
    'nevus',
    'normal',
    'pigmented benign keratosis'
]

# Disease information
DISEASE_INFO = {
    'basal cell carcinoma': {
        'full_name': 'Basal Cell Carcinoma',
        
        
    },
    'melanoma': {
        'full_name': 'Melanoma',
        
        
    },
    'nevus': {
        'full_name': 'Nevus (Mole)',
        
    },
    'normal': {
        'full_name': 'Normal Skin',
        
    },
    'pigmented benign keratosis': {
        'full_name': 'Pigmented Benign Keratosis',

    }
}

def load_model(model_path, num_classes=5):
    """Load the trained model"""
    model = TransferLearningClassifier(num_classes=num_classes, dropout_rate=0.5).to(device)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print("Please train the model first using the training script.")
        return None
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✓ Model loaded successfully from {model_path}\n")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_image(model, image_path, transform):
    """Predict single image with confidence scores"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get all class probabilities
            all_probs = probabilities[0].cpu().numpy()
        
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item() * 100
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'all_probabilities': all_probs,
            'image_size': original_size,
            'success': True
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

def print_prediction_result(image_name, result, show_all_probs=False):
    """Print formatted prediction result"""
    print("=" * 100)
    print(f"📸 IMAGE: {image_name}")
    print("=" * 100)
    
    if not result['success']:
        print(f"❌ Error: {result['error']}\n")
        return
    
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    info = DISEASE_INFO[predicted_class]
    
    # Main prediction
    print(f"\n🔍 PREDICTION: {info['full_name']}")
    print(f"📊 CONFIDENCE: {confidence:.2f}%")
    
    # Confidence bar
    bar_length = 50
    filled_length = int(bar_length * confidence / 100)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    print(f"   [{bar}] {confidence:.2f}%")
    
    # Severity indicator
    severity_colors = {
        'CRITICAL': '🔴',
        'HIGH': '🟠',
        'LOW': '🟢',
        'NONE': '⚪'
    }
    print(f"\n⚠️  SEVERITY: {severity_colors.get(info['severity'], '⚪')} {info['severity']}")
    print(f"📝 DESCRIPTION: {info['description']}")
    print(f"💡 RECOMMENDATION: {info['recommendation']}")
    
    # Show all probabilities if requested
    if show_all_probs:
        print(f"\n📈 ALL CLASS PROBABILITIES:")
        print("-" * 70)
        sorted_indices = np.argsort(result['all_probabilities'])[::-1]
        for idx in sorted_indices:
            class_name = CLASS_NAMES[idx]
            prob = result['all_probabilities'][idx] * 100
            bar_len = int(30 * prob / 100)
            bar = '█' * bar_len + '░' * (30 - bar_len)
            print(f"  {class_name:30s} [{bar}] {prob:5.2f}%")
    
    print("\n" + "=" * 100 + "\n")

def predict_folder(model, folder_path, show_all_probs=False, save_results=True):
    """Predict all images in a folder"""
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Error: Folder not found: {folder_path}")
        return
    
    # Check if folder has subdirectories (class-based structure)
    subdirs = [d for d in folder.iterdir() if d.is_dir()]
    
    if subdirs:
        print(f"📁 Found {len(subdirs)} subdirectories. Processing all images...\n")
        for subdir in subdirs:
            for file in subdir.iterdir():
                if file.suffix.lower() in image_extensions:
                    image_files.append(file)
    else:
        for file in folder.iterdir():
            if file.suffix.lower() in image_extensions:
                image_files.append(file)
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return
    
    print(f"✓ Found {len(image_files)} images to process\n")
    
    # Process all images
    results = []
    predictions_summary = {class_name: 0 for class_name in CLASS_NAMES}
    
    for img_file in image_files:
        result = predict_image(model, str(img_file), test_transform)
        
        if result['success']:
            result['filename'] = img_file.name
            result['filepath'] = str(img_file)
            results.append(result)
            predictions_summary[result['predicted_class']] += 1
            print_prediction_result(img_file.name, result, show_all_probs)
        else:
            print(f"❌ Failed to process: {img_file.name} - {result['error']}\n")
    
    # Print summary
    print("\n" + "=" * 100)
    print("📊 PREDICTION SUMMARY")
    print("=" * 100)
    print(f"\nTotal images processed: {len(results)}/{len(image_files)}")
    print(f"\nDisease Distribution:")
    print("-" * 70)
    
    for class_name, count in predictions_summary.items():
        percentage = (count / len(results) * 100) if results else 0
        info = DISEASE_INFO[class_name]
        bar_len = int(40 * percentage / 100)
        bar = '█' * bar_len + '░' * (40 - bar_len)
        print(f"{info['full_name']:35s} [{bar}] {count:3d} ({percentage:5.1f}%)")
    
    # High priority cases
    high_priority = [r for r in results if DISEASE_INFO[r['predicted_class']]['severity'] in ['CRITICAL', 'HIGH']]
    if high_priority:
        print(f"\n⚠️  HIGH PRIORITY CASES: {len(high_priority)}")
        print("-" * 70)
        for r in high_priority:
            severity = DISEASE_INFO[r['predicted_class']]['severity']
            print(f"  • {r['filename']:40s} → {r['predicted_class']:30s} ({r['confidence']:.1f}%) [{severity}]")
    
    print("\n" + "=" * 100)
    
    # Save results to file
    if save_results and results:
        output_file = "prediction_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("SKIN DISEASE PREDICTION RESULTS\n")
            f.write("=" * 100 + "\n\n")
            
            for r in results:
                info = DISEASE_INFO[r['predicted_class']]
                f.write(f"Image: {r['filename']}\n")
                f.write(f"Prediction: {info['full_name']}\n")
                f.write(f"Confidence: {r['confidence']:.2f}%\n")
                f.write(f"Severity: {info['severity']}\n")
                f.write(f"Recommendation: {info['recommendation']}\n")
                f.write("-" * 100 + "\n\n")
            
            f.write("\nSUMMARY\n")
            f.write("=" * 100 + "\n")
            f.write(f"Total images: {len(results)}\n\n")
            for class_name, count in predictions_summary.items():
                percentage = (count / len(results) * 100) if results else 0
                f.write(f"{DISEASE_INFO[class_name]['full_name']:35s}: {count:3d} ({percentage:5.1f}%)\n")
        
        print(f"\n✓ Results saved to: {output_file}")

def main():
    print("\n" + "=" * 100)
    print("🩺 SKIN DISEASE CLASSIFICATION - PREDICTION TOOL")
    print("=" * 100 + "\n")
    
    # Configuration
    MODEL_PATH = 'best_skin_classifier_transfer.pth'
    TEST_FOLDER = r'D:\ml\skin_cancer\Test'
    
    # You can also test on individual images
    # SINGLE_IMAGE = r'D:\ml\skin_cancer\skin_with_types\Test\melanoma\image1.jpg'
    
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        return
    
    print("Choose prediction mode:")
    print("1. Predict all images in Test folder")
    print("2. Predict single image")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        show_all = input("Show all class probabilities? (y/n): ").strip().lower() == 'y'
        predict_folder(model, TEST_FOLDER, show_all_probs=show_all, save_results=True)
    
    elif choice == '2':
        image_path = input("Enter image path: ").strip().strip('"')
        if os.path.exists(image_path):
            result = predict_image(model, image_path, test_transform)
            print_prediction_result(os.path.basename(image_path), result, show_all_probs=True)
        else:
            print(f"❌ Image not found: {image_path}")
    
    else:
        print("Invalid choice!")
    
    
    print("=" * 100 + "\n")

if __name__ == '__main__':
    main()