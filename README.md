Fashion Product Classification using Deep Learning

**📌 Project Overview**

This project aims to classify fashion products into multiple categories using a Convolutional Neural Network (CNN). Given an image of a fashion product, the model predicts the following attributes:

- Color of the product

- Type (e.g., T-shirt, shoes, dress, etc.)

- Season suitable for the product

- Gender (Men, Women, Unisex)

The dataset used for this project is the Myntra Fashion Dataset, which includes images and metadata of various fashion products.

**🚀 Features**

- Multi-label classification of fashion products

- CNN-based deep learning model

- Trained using GPU acceleration on Kaggle

- Model evaluation using accuracy and confusion matrices

- Visualization of predictions on test images

**📂 Project Structure**

├── dataset/                     # Dataset directory (images & CSV file)
├── models/                      # Trained model & label encoders
│   ├── fashion_model.pth        # Saved PyTorch model weights
│   ├── label_encoders.pkl       # Encoded labels for classes
├── notebooks/                   # Jupyter notebooks
│   ├── fashion_classification.ipynb  # Main training notebook
├── utils/                       # Helper functions
│   ├── preprocess.py            # Data preprocessing functions
│   ├── visualization.py         # Functions to visualize predictions
├── requirements.txt             # Required Python libraries
├── README.md                    # Project documentation

📊 Dataset

Images Directory: /kaggle/input/data-set/myntradataset/images

Data File: /kaggle/input/data-set/myntradataset/styles.csv

The dataset consists of images of fashion products along with metadata, including color, type, season, and gender labels.

🔧 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/yourusername/fashion-product-classification.git
cd fashion-product-classification

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Download Dataset

If using Kaggle:

kaggle datasets download -d dataset-name

Alternatively, manually download and place it in the dataset/ directory.

**📖 Training the Model**

To train the model from scratch, run the Jupyter Notebook:

jupyter notebook notebooks/fashion_classification.ipynb

Make sure to enable GPU acceleration if using Kaggle or Colab.

The trained model will be saved as fashion_model.pth.

🎯** Testing the Model**

To test the model on sample images, run:

    from utils.visualization import show_predictions

    image_path = "dataset/images/sample.jpg"
    show_predictions(image_path, model, transform, label_encoders)

This will display the image along with predicted labels.

📊 Model Evaluation

The model is evaluated on a test set using accuracy and confusion matrices.

The confusion matrix helps visualize model performance on each category.

To generate confusion matrices:

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    def plot_confusion_matrix(y_true, y_pred, labels, title):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

🔗 Model & Dataset Links

Dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

📌 Future Improvements

- Improve accuracy by experimenting with different CNN architectures (ResNet, EfficientNet, etc.)

- Fine-tune the model with a larger dataset

- Deploy the model as a web app using Flask or FastAPI

🤝 Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue.

📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

💡 Acknowledgments

The Myntra Fashion Dataset

PyTorch and OpenAI for deep learning frameworks

Kaggle for providing computing resources

Predicted using Amazon Screenshots.
