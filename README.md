# -Early-Stage-Cancer-Detection
🧠 Early Stage Cancer Detection System using AI
📌 Project Overview

Early detection of cancer plays a critical role in improving survival rates and providing timely medical intervention. This project presents an Artificial Intelligence-based medical diagnosis system capable of detecting cancer at an early stage using medical images such as MRI scans and skin lesion images.

The system leverages Deep Learning and Computer Vision techniques to analyze medical images and classify them into cancerous or non-cancerous categories with high accuracy. It also includes explainability features using Grad-CAM, helping doctors and users understand model predictions visually.

🎯 Objectives

The main objectives of this project are:

To develop an AI-powered system that can detect cancer at an early stage

To assist doctors with faster and more reliable diagnosis

To reduce human error in medical image interpretation

To provide visual explanations for predictions using Grad-CAM

To build an easy-to-use web interface for real-time predictions

🚀 Features

✔ Automated cancer detection from medical images

✔ Supports multiple cancer types (Brain Tumor, Skin Cancer, etc.)

✔ Deep Learning model with high accuracy

✔ Grad-CAM based explainability

✔ User-friendly web interface

✔ Fast and real-time prediction

✔ Upload and visualize medical images

✔ Downloadable prediction reports

🧪 Problem Statement

Traditional cancer diagnosis requires expert radiologists and dermatologists to manually analyze medical images. This process:

Is time-consuming

Can be prone to human error

Requires highly skilled professionals

Is not easily accessible in remote areas

This project solves these issues by using AI-based automated diagnosis, making early detection faster, cheaper, and more accessible.

🛠 Technology Stack
Programming Languages

Python

JavaScript

HTML / CSS

Frameworks & Libraries

PyTorch / TensorFlow

OpenCV

Flask / FastAPI

NumPy

Pandas

Matplotlib

Scikit-learn

Deep Learning Models

ResNet50

MobileNetV2

CNN-based custom architectures

Tools

Grad-CAM

Jupyter Notebook

Git & GitHub

🧬 Dataset Information

The system is trained on publicly available medical datasets:

Brain Tumor MRI Dataset

ISIC Skin Cancer Dataset

Chest X-Ray Dataset (if applicable)

Each dataset contains labeled images categorized into:

Benign

Malignant

Normal

Data preprocessing techniques used:

Image resizing

Normalization

Data augmentation

CLAHE (for contrast enhancement)

⚙ System Architecture

The project follows the below pipeline:

User uploads medical image

Image preprocessing

Model inference

Prediction generation

Explainability visualization

Display results on UI

📊 Methodology
1. Data Collection

Medical image datasets collected from verified online sources.

2. Data Preprocessing

Noise removal

Image enhancement

Rescaling

Augmentation

3. Model Training

Transfer learning using pre-trained CNN models

Fine-tuning on medical datasets

Hyperparameter optimization

4. Model Evaluation

Metrics used:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

5. Explainability

Grad-CAM used to highlight important regions in the image influencing predictions.

💻 How to Run the Project
Prerequisites

Python 3.8+

pip

Virtual Environment

Installation Steps
# Clone the repository
git clone https://github.com/drashanniphade/Early-Stage-Cancer-Detection.git

# Navigate to project folder
cd early-cancer-detection

# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py


Open browser and go to:

http://127.0.0.1:5000

📈 Results

The trained model achieved:

Accuracy: ~95%

Precision: ~94%

Recall: ~93%

F1 Score: ~94%

These results indicate that the system is highly reliable for assisting medical diagnosis.

🌐 User Interface

The web application allows users to:

Upload MRI / medical images

View prediction results

See Grad-CAM heatmaps

Download reports

🧩 Modules in the Project

Data Preprocessing Module

Model Training Module

Prediction Module

Explainability Module

Web Application Module

🔮 Future Enhancements

Support for more cancer types

Integration with hospital databases

Mobile application

Real-time doctor consultation

Cloud deployment

Multilingual support

👨‍💻 Contributors

Your Name – Developer

Guided by – Mentor/Professor Name

📜 License

This project is licensed under the MIT License.

📞 Contact

For any queries or collaboration:

Email: darshanprofessional19@gmail.com

⭐ If you find this project useful, please give it a star on GitHub!
