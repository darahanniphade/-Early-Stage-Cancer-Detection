// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewImage = document.getElementById('previewImage');
const removeImageBtn = document.getElementById('removeImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsCard = document.getElementById('resultsCard');
const resultsContent = document.getElementById('resultsContent');
const loadingState = document.getElementById('loadingState');
const resultsDisplay = document.getElementById('resultsDisplay');
const errorState = document.getElementById('errorState');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const retryBtn = document.getElementById('retryBtn');

// State
let selectedFile = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

// Setup Event Listeners
function setupEventListeners() {
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary-color)';
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--border-color)';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--border-color)';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    analyzeBtn.addEventListener('click', analyzeBrain);
    newAnalysisBtn.addEventListener('click', resetAll);
    retryBtn.addEventListener('click', analyzeBrain);
}

// Handle File Selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload an image file.');
        return;
    }

    if (file.size > 16 * 1024 * 1024) {
        showError('File is too large. Maximum size is 16MB.');
        return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewArea.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Analyze Brain MRI
async function analyzeBrain() {
    if (!selectedFile) return;

    resultsContent.style.display = 'block';
    loadingState.style.display = 'block';
    resultsDisplay.style.display = 'none';
    errorState.style.display = 'none';
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict/brain', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            displayResults(result);
        } else {
            showError(result.error || 'Analysis failed');
        }
    } catch (error) {
        showError('Network error. Please check your connection and try again.');
        console.error('Analysis error:', error);
    } finally {
        analyzeBtn.disabled = false;
    }
}

// Display Results
function displayResults(data) {
    loadingState.style.display = 'none';
    resultsDisplay.style.display = 'block';

    document.getElementById('resultTitle').textContent = 'Diagnosis Result';
    document.getElementById('resultTime').textContent = new Date().toLocaleString();

    const prediction = data.predicted_class;
    document.getElementById('predictionValue').textContent = formatDiseaseName(prediction);

    const confidence = data.confidence;
    document.getElementById('confidenceValue').textContent = `${confidence.toFixed(2)}%`;
    document.getElementById('confidenceFill').style.width = `${confidence}%`;

    const confidenceFill = document.getElementById('confidenceFill');
    if (confidence >= 80) {
        confidenceFill.style.background = 'var(--success-color)';
    } else if (confidence >= 60) {
        confidenceFill.style.background = 'var(--warning-color)';
    } else {
        confidenceFill.style.background = 'var(--danger-color)';
    }

    // Display GradCAM if available
    if (data.gradcam_image) {
        displayGradCAM(data.gradcam_image);
    }

    displayProbabilities(data.all_probabilities || {});
}

// Display GradCAM Visualization
function displayGradCAM(gradcamBase64) {
    // Check if GradCAM section already exists
    let gradcamSection = document.getElementById('gradcamSection');
    
    if (!gradcamSection) {
        // Create GradCAM section
        gradcamSection = document.createElement('div');
        gradcamSection.id = 'gradcamSection';
        gradcamSection.className = 'gradcam-section';
        gradcamSection.innerHTML = `
            <h4><i class="fas fa-brain"></i> GradCAM Visualization</h4>
            <p class="gradcam-description">Regions highlighted in red-yellow indicate areas the AI focused on for diagnosis</p>
            <div class="gradcam-container">
                <img id="gradcamImage" alt="GradCAM Heatmap">
            </div>
        `;
        
        // Insert before probabilities section
        const probabilitiesSection = document.querySelector('.probabilities-section');
        probabilitiesSection.parentNode.insertBefore(gradcamSection, probabilitiesSection);
        
        // Add CSS styles dynamically
        if (!document.getElementById('gradcam-styles')) {
            const style = document.createElement('style');
            style.id = 'gradcam-styles';
            style.textContent = `
                .gradcam-section {
                    background: var(--dark-bg);
                    padding: 1.5rem;
                    border-radius: 12px;
                    margin-bottom: 2rem;
                    border: 1px solid var(--border-color);
                }
                
                .gradcam-section h4 {
                    margin-bottom: 0.5rem;
                    color: var(--text-primary);
                }
                
                .gradcam-section h4 i {
                    color: var(--primary-color);
                    margin-right: 0.5rem;
                }
                
                .gradcam-description {
                    color: var(--text-secondary);
                    font-size: 0.9rem;
                    margin-bottom: 1rem;
                }
                
                .gradcam-container {
                    background: var(--card-bg);
                    border-radius: 8px;
                    padding: 1rem;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                
                #gradcamImage {
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: var(--shadow-md);
                }
            `;
            document.head.appendChild(style);
        }
    }
    
    // Set GradCAM image
    const gradcamImage = document.getElementById('gradcamImage');
    gradcamImage.src = gradcamBase64;
}

// Display Probabilities
function displayProbabilities(probabilities) {
    const probabilityList = document.getElementById('probabilityList');
    probabilityList.innerHTML = '';

    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    sorted.forEach(([name, value]) => {
        const item = document.createElement('div');
        item.className = 'probability-item';
        
        item.innerHTML = `
            <span class="probability-name">${formatDiseaseName(name)}</span>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${value}%"></div>
            </div>
            <span class="probability-value">${value.toFixed(2)}%</span>
        `;
        
        probabilityList.appendChild(item);
    });
}

// Show Error
function showError(message) {
    resultsContent.style.display = 'block';
    loadingState.style.display = 'none';
    resultsDisplay.style.display = 'none';
    errorState.style.display = 'block';
    document.getElementById('errorMessage').textContent = message;
}

// Reset Upload
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    uploadArea.style.display = 'flex';
    previewArea.style.display = 'none';
    previewImage.src = '';
    analyzeBtn.disabled = true;
}

// Reset All
function resetAll() {
    resetUpload();
    resultsContent.style.display = 'none';
    document.querySelector('.results-placeholder').style.display = 'block';
    
    // Remove GradCAM section if it exists
    const gradcamSection = document.getElementById('gradcamSection');
    if (gradcamSection) {
        gradcamSection.remove();
    }
}

// Utility: Format disease name
function formatDiseaseName(name) {
    return name.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
}