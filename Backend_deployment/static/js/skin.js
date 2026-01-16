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

// Severity mapping
const SEVERITY_MAP = {
    'melanoma': { level: 'high', label: 'High Risk', description: 'This is the most dangerous type of skin cancer. Immediate consultation with a dermatologist is strongly recommended.' },
    'basal cell carcinoma': { level: 'moderate', label: 'Moderate Risk', description: 'Common skin cancer that grows slowly. Should be evaluated by a dermatologist soon.' },
    'pigmented benign keratosis': { level: 'low', label: 'Benign', description: 'This is a non-cancerous growth. Regular monitoring is recommended.' },
    'nevus': { level: 'low', label: 'Benign', description: 'Common mole, typically harmless. Monitor for any changes.' },
    'normal': { level: 'low', label: 'Normal', description: 'No concerning features detected. Continue regular skin checks.' }
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

// Setup Event Listeners
function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--secondary-color)';
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

    // Remove image
    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    // Analyze button
    analyzeBtn.addEventListener('click', analyzeSkin);

    // New analysis
    newAnalysisBtn.addEventListener('click', resetAll);

    // Retry button
    retryBtn.addEventListener('click', analyzeSkin);
}

// Handle File Selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload an image file.');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File is too large. Maximum size is 16MB.');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewArea.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Analyze Skin Lesion
async function analyzeSkin() {
    if (!selectedFile) return;

    // Show loading state
    resultsContent.style.display = 'block';
    loadingState.style.display = 'block';
    resultsDisplay.style.display = 'none';
    errorState.style.display = 'none';
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict/skin', {
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

    // Update result title and time
    document.getElementById('resultTitle').textContent = 'Analysis Result';
    document.getElementById('resultTime').textContent = new Date().toLocaleString();

    // Update prediction
    const prediction = data.predicted_class;
    document.getElementById('predictionValue').textContent = formatDiseaseName(prediction);

    // Update confidence
    const confidence = data.confidence;
    document.getElementById('confidenceValue').textContent = `${confidence.toFixed(2)}%`;
    document.getElementById('confidenceFill').style.width = `${confidence}%`;

    // Set confidence color based on value
    const confidenceFill = document.getElementById('confidenceFill');
    if (confidence >= 80) {
        confidenceFill.style.background = 'var(--success-color)';
    } else if (confidence >= 60) {
        confidenceFill.style.background = 'var(--warning-color)';
    } else {
        confidenceFill.style.background = 'var(--danger-color)';
    }

    // Display severity
    displaySeverity(prediction);

    // Display all probabilities
    displayProbabilities(data.all_probabilities || {});
}

// Display Severity
function displaySeverity(prediction) {
    const severityInfo = SEVERITY_MAP[prediction.toLowerCase()] || SEVERITY_MAP['normal'];
    const severityBadge = document.getElementById('severityBadge');
    const severityDescription = document.getElementById('severityDescription');

    severityBadge.textContent = severityInfo.label;
    severityBadge.className = `severity-badge ${severityInfo.level}`;
    severityDescription.textContent = severityInfo.description;
}

// Display Probabilities
function displayProbabilities(probabilities) {
    const probabilityList = document.getElementById('probabilityList');
    probabilityList.innerHTML = '';

    // Sort probabilities by value
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
}

// Utility: Format disease name
function formatDiseaseName(name) {
    return name.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
}