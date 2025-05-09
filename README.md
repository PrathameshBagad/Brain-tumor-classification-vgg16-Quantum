
# Brain Tumor Classification using VGG16 + Quantum Circuit

This project combines classical deep learning using VGG16 with quantum feature extraction using PennyLane to classify brain MRI scans into three tumor types:
- Meningioma
- Glioma
- Pituitary Tumor

## 📁 Dataset
- Source: [Figshare Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
- Format: `.mat` MATLAB v7.3 files
- Each `.mat` file contains:
  - `image`: 2D grayscale brain MRI slice
  - `label`: Integer representing tumor class
  - `tumorMask`: Binary mask of tumor region (not used in this version)

## 🧠 Model Architecture

### 1. **VGG16 Branch**
- Uses pretrained ImageNet weights
- Fine-tunes last 8 layers
- Extracts classical features from RGB-converted MRI slices

### 2. **Quantum Circuit Branch**
- Uses `pennylane` with 8 qubits
- Encodes first 8 pixels (flattened image) into quantum states
- Extracts quantum features via RX, RY, and CNOT gates

### 3. **Combined Model**
- Concatenates classical and quantum features
- Final classifier uses dense layers with dropout
- Optimizer: AdamW
- Loss: Categorical Crossentropy

## 🧪 Evaluation
- Evaluation metrics: Accuracy, Precision, Recall, F1-score
- Uses scikit-learn for classification report and confusion matrix

## 🧾 Requirements
- TensorFlow
- Pennylane
- scikit-learn
- h5py
- OpenCV
- Matplotlib

## 🚀 How to Run
1. Place `.mat` files in: `/content/dataset/figshare-brain-tumor-dataset/dataset/data`
2. Run the notebook sections sequentially:
   - Data loading and preprocessing
   - Quantum feature extraction
   - Combined model building and training
   - Evaluation and visualization
   - Model saving

## 📦 Output
- Trained `.h5` model saved at: `/content/combined_vgg16_quantum_model.h5`
- Confusion matrix and classification report plotted

## Validation on Unseen dataset
📂 1. Using a New Dataset
The new dataset is organized into folders (Glioma, Meningioma, Pituitary).

It has never been used in training or prior validation.

This setup simulates a domain shift — testing the model on data from different acquisition settings or patient populations.

Dataset used for validation : #link(https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download&select=Testing)

🧪 2. Fine-Tuning on 10% of the New Data
We load pretrained weights from our VGG16 + Quantum model.

Fine-tuning is done on just 10% of the new dataset.

This helps the model adapt to subtle changes in MRI characteristics without overfitting.

✅ 3. Validation on Remaining 90%
The model is evaluated on the remaining 90% of the unseen dataset.

We compute accuracy, classification metrics, and confusion matrix to assess performance.

## Comparison of various models
-Run comparison file
🧠 Models Evaluated:
VGG16
VGGNet
ResNet50
DenseNet
Quantum-enhanced versions of each
| Model               | Accuracy   | Recall | Precision | F1-score |
| ------------------- | ---------- | ------ | --------- | -------- |
| **VGG16**           | 94.94%     | 94%    | 95%       | 94%      |
| **VGG16 + Quantum** | **97.55%** | 97%    | 98%       | 98%      |
| VGGNet              | 90.21%     | 89%    | 90%       | 90%      |
| VGGNet + Quantum    | 96.57%     | 96%    | 97%       | 97%      |
| ResNet50            | 91.35%     | 91%    | 91%       | 91%      |
| ResNet50 + Quantum  | 91.35%     | 91%    | 91%       | 91%      |
| DenseNet            | 90.21%     | 90%    | 90%       | 90%      |
| DenseNet + Quantum  | 90.70%     | 91%    | 91%       | 91%      |

