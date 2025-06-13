# Plant Disease Prediction from Leaf Images 🌱🔍

This project focuses on the detection and classification of plant diseases from leaf images using deep learning techniques. By leveraging Convolutional Neural Networks (CNNs), the model can accurately identify various diseases, aiding in early detection and prevention of crop damage.
TRAINED MODLE LINK - https://drive.google.com/file/d/1hCxxf7Uoa8cI_4pnJ0lB7cNVN5EIAq7y/view?usp=sharing 



## 📖 About the Project
The primary goal is to build a deep learning model capable of identifying plant diseases by analyzing leaf images. This automated approach helps farmers and agricultural experts quickly diagnose problems, improving crop yield and reducing losses.
![Screenshot 2025-06-13 192215](https://github.com/user-attachments/assets/0cbadae0-6de5-4775-8d58-80870eb341f5)
![Screenshot 2025-06-13 180801](https://github.com/user-attachments/assets/c158ec88-06b6-42d8-bcf8-9e1afb1fe781)
![Screenshot 2025-06-13 164944](https://github.com/user-attachments/assets/ce03fd5f-e285-4895-b382-e1e42cf31ed3)
![Screenshot 2025-06-13 192039](https://github.com/user-attachments/assets/3007bc2f-a3e1-449b-97a3-359602d8e60f)




Key Features:
- CNN-based image classification
- 38-class disease detection
- High accuracy model (~97%)
- Easy-to-follow Jupyter Notebook implementation

## 📂 Dataset
The model uses the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) from Kaggle containing thousands of images across:

- 38 different classes (diseases + healthy leaves)
- Multiple plant species
- Color images (256×256 pixels)

Dataset is downloaded and unzipped automatically using the Kaggle API in the notebook.

## 🛠️ Methodology
### CNN Architecture
Built using Keras with TensorFlow backend:

1. **Conv2D layers** - Feature extraction (edges, textures)
2. **MaxPooling2D layers** - Downsampling feature maps
3. **Flatten layer** - Converts 2D features to 1D vector
4. **Dense layers** - Final classification with softmax activation

### Training Configuration:
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy
- Epochs: 20

## 📦 Dependencies
Install required packages:
pip install tensorflow numpy matplotlib pillow

## Core Libraries:

-TensorFlow/Keras - Model building and training

-NumPy - Numerical operations

-Matplotlib - Visualization

-Pillow (PIL) - Image processing


## Results
Final performance metrics:

Metric          	    Training	 Validation
**Accuracy**          	97.97%	 87.35%
**Loss**	                0.06	 0.09



