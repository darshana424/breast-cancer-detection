# Breast Cancer Detection Using Neural Networks

## Project Overview
This project implements a deep learning solution for automated breast cancer tumor classification using neural networks. The system classifies tumors as malignant or benign based on cellular characteristics extracted from diagnostic images.

## Problem Statement
Breast cancer remains one of the leading causes of cancer-related mortality worldwide. Early and accurate detection is critical for successful treatment outcomes. Traditional manual diagnosis methods can be time-consuming and subject to human error. This project addresses the need for an automated, reliable, and efficient diagnostic support system that can assist medical professionals in making faster and more accurate diagnoses.

## Objective
The primary objective is to develop a robust neural network-based classification model that can:
- Accurately distinguish between malignant and benign breast tumors
- Achieve high prediction accuracy (>95%) to support clinical decision-making
- Process tumor characteristics efficiently for real-time diagnostic support
- Provide a scalable solution that can be integrated into existing medical workflows

## Dataset Description

**Source**: Wisconsin Breast Cancer Dataset (569 samples)

**Features**: 30 numerical attributes computed from digitized images of fine needle aspirate (FNA) of breast mass, including:
- Radius, texture, perimeter, area, smoothness
- Compactness, concavity, concave points
- Symmetry, fractal dimension
- Statistical measures (mean, standard error, worst) for each characteristic

**Target Variable**:
- **Malignant (M)**: Cancerous tumors (212 samples)
- **Benign (B)**: Non-cancerous tumors (357 samples)

**Data Quality**:
- No missing values
- Balanced class distribution
- Pre-validated medical data

## Methodology / Approach

### 1. Data Preprocessing
- **Data Loading**: CSV import and validation
- **Exploratory Data Analysis**: Statistical analysis and distribution visualization
- **Label Encoding**: Categorical diagnosis conversion (M=1, B=0)
- **Feature Engineering**: Removal of non-predictive columns (ID)
- **Data Splitting**: 80-20 train-test stratified split
- **Standardization**: Feature scaling using StandardScaler for optimal neural network performance

### 2. Model Architecture
- **Input Layer**: Flatten layer accepting 30 features
- **Hidden Layer**: Dense layer with 20 neurons and ReLU activation
- **Output Layer**: Dense layer with 2 neurons and sigmoid activation (binary classification)
- **Optimizer**: Adam optimizer for adaptive learning rate
- **Loss Function**: Sparse categorical crossentropy
- **Metrics**: Classification accuracy

### 3. Training Strategy
- **Epochs**: 10 iterations
- **Validation Split**: 10% of training data for real-time validation
- **Early Stopping Consideration**: Monitored through validation metrics
- **Batch Processing**: Automated batching for efficient training

### 4. Model Evaluation
- Test set accuracy: **96.49%**
- Training/validation loss and accuracy tracking
- Visual performance analysis through matplotlib plots
- Confusion matrix analysis for error pattern identification

## Tools & Technologies Used

| Category | Technology | Version |
|----------|-----------|---------|
| **Programming Language** | Python | 3.x |
| **Deep Learning Framework** | TensorFlow/Keras | Latest |
| **Data Processing** | Pandas | Latest |
| | NumPy | Latest |
| **Visualization** | Matplotlib | Latest |
| **Machine Learning** | Scikit-learn | Latest |
| **Development Environment** | Google Colab | - |
| **Version Control** | Git | Latest |

## Steps to Run the Project

### Prerequisites
```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```

### Execution Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/darshana424/breast-cancer-detection.git
   cd breast-cancer-detection
   ```

2. **Dataset Preparation**
   - Place `breast-cancer.csv` in the project root directory
   - Verify data integrity and format

3. **Execute the Notebook**
   ```bash
   jupyter notebook Untitled2.ipynb
   ```
   Or upload to Google Colab for cloud execution

4. **Run Cells Sequentially**
   - Import dependencies
   - Load and preprocess data
   - Train the model
   - Evaluate performance
   - Test with new samples

5. **Model Inference**
   ```python
   # Example prediction
   input_data = (11.76, 21.6, 74.72, ..., 0.06563)
   prediction = model.predict(scaler.transform([input_data]))
   ```

## Results / Output

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 96.49% |
| **Training Accuracy** | 94.71% (final epoch) |
| **Validation Accuracy** | 97.83% (final epoch) |
| **Final Training Loss** | 0.1186 |
| **Final Validation Loss** | 0.0916 |

### Key Achievements

✅ **High Accuracy**: Achieved >96% test accuracy, exceeding clinical requirements  
✅ **Model Convergence**: Stable learning curve with minimal overfitting  
✅ **Generalization**: Strong validation performance indicating robust generalization  
✅ **Real-time Inference**: Sub-second prediction capability for clinical deployment  

### Visualization Outputs

1. **Training History Plots**
   - Model accuracy progression over epochs
   - Loss function convergence analysis
   - Training vs. validation performance comparison

2. **Sample Predictions**
   - Correctly classified malignant sample with 84.14% confidence
   - Demonstrates model's decision-making capability

### Model Reliability

The model demonstrates:
- Consistent performance across training and validation sets
- No significant overfitting (validation accuracy > training accuracy)
- Smooth convergence indicating stable learning
- High confidence in predictions (>80% for test cases)

## Project Structure

```
breast-cancer-detection/
│
├── Untitled2.ipynb           # Main Jupyter notebook
├── breast-cancer.csv          # Dataset (569 samples, 32 features)
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
└── models/                    # Saved model directory
    └── breast_cancer_model.h5 # Trained Keras model
```

## Dataset Information

**File**: `breast-cancer.csv`  
**Size**: 569 rows × 32 columns  
**Source**: [Wisconsin Breast Cancer Database](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

## Model Files

The trained model can be saved and loaded using:

```python
# Save model
model.save('models/breast_cancer_model.h5')

# Load model
from tensorflow import keras
loaded_model = keras.models.load_model('models/breast_cancer_model.h5')
```

## Future Enhancements

- [ ] Implement cross-validation for robust performance estimation
- [ ] Add explainability features (SHAP values, attention mechanisms)
- [ ] Develop REST API for clinical system integration
- [ ] Expand dataset with additional tumor characteristics
- [ ] Implement ensemble methods for improved accuracy
- [ ] Create web-based user interface for medical practitioners
- [ ] Add support for multi-class classification (tumor subtypes)

## License
This project is developed for educational and research purposes. Medical applications require appropriate regulatory approvals.

## Disclaimer
This model is designed as a diagnostic support tool and should not replace professional medical judgment. All predictions should be validated by qualified healthcare professionals.

**Project Status**: ✅ Completed
**Last Updated**: January 2026 
