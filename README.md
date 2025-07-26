# Brain MRI Tumor Segmentation using UNETR (ViT-based CNN)

This project presents a deep learning pipeline for segmenting brain tumors from MRI scans using a transformer-based architecture known as **UNETR** (U-Net with Transformers as Encoders). UNETR combines the global attention mechanism of Vision Transformers (ViT) with the localization strength of CNN-based decoders, delivering state-of-the-art results on medical image segmentation tasks.

## 🎯 Objective

To automate brain tumor segmentation from volumetric MRI scans using UNETR, improving early diagnosis and assisting radiologists in clinical decision-making.

## 🧠 About UNETR

UNETR (U-Net Transformer) is an advanced architecture for medical image segmentation introduced by MONAI (Medical Open Network for AI). It uses a ViT encoder to learn global features, directly connecting transformer encoder layers to the decoder via skip connections—enabling precise spatial localization.

## 📁 Dataset

- **Source**: Brain Tumor Segmentation (BraTS) Dataset
- **Modalities**: T1, T1-Gd, T2, and FLAIR MRI sequences
- **Labels**: Multiclass tumor segmentation – enhancing tumor, edema, and necrotic core
- **Format**: NIfTI (`.nii.gz`) volumetric images
- **Split**: Train, Validation, Test (customizable)

## 🛠️ Tech Stack

- **Framework**: TensorFlow, Keras
- **Backbone**: UNETR (Transformer-based U-Net)
- **Libraries**: MONAI, NiBabel, NumPy, OpenCV, Matplotlib
- **Language**: Python 3.9+
- **Visualization**: Matplotlib, OpenCV
- **Environment**: VS Code / Python scripts

## ⚙️ Features

- ✅ Volumetric image loading and preprocessing using NiBabel and MONAI
- ✅ Patch-based ViT encoder with positional embeddings
- ✅ U-Net-style decoder with upsampling and skip connections
- ✅ Loss: Combined Dice + Binary Cross Entropy (BCE)
- ✅ Evaluation Metrics: Dice Score, IoU, Precision, Recall
- ✅ Real-time segmentation visualization for slices
- ✅ Configurable training pipeline

## 🧪 Model Architecture

- **Encoder**: Vision Transformer (ViT)
- **Decoder**: CNN-based with transpose convolutions and skip connections
- **Skip Connections**: Transformer outputs fused with upsampled feature maps
- **Activation**: Sigmoid for binary mask prediction
- **Loss Function**: `DiceLoss + BCEWithLogitsLoss`

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/nishantwestside/Brain-MRI-Tumor-Segmentation-using-UNETR-ViT-based-CNN-.git
   cd Brain-MRI-Tumor-Segmentation-using-UNETR-ViT-based-CNN-
2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
3. **Download the Dataset**
   - Download BraTS dataset
   - Place it under `./files/`
4. **Run the Training Script**
   ```bash
   python train.py
5. **Test the Model**
   ```bash
   python test.py

## 📊 Evaluation Metrics

- Dice Coefficient: Measures overlap between predicted and ground truth masks

- IoU (Jaccard Index)

- Pixel Accuracy

- Precision & Recall

## 💡 Key Contributions

- Leveraged a hybrid ViT-CNN architecture (UNETR) to improve medical segmentation accuracy

- Designed an end-to-end MRI preprocessing, training, and visualization pipeline

- Integrated MONAI's medical image I/O and ViT components for faster experimentation
