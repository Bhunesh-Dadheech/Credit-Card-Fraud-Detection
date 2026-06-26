# Credit-Card-Fraud-Detection

### Securing Financial Transactions via Advanced Neural Networks

This project deploys a high-precision deep learning model to identify fraudulent credit card transactions. By leveraging a specialized neural network architecture, it effectively tackles the challenge of class imbalance to distinguish between legitimate and fraudulent activities with high accuracy.

---

##  Project Scope

* **Binary Classification:** Distinguishes strictly between Valid (`0`) and Fraudulent (`1`) transactions.
* **Imbalance Handling:** Utilizes SMOTE (Synthetic Minority Over-sampling Technique) to normalize the dataset distribution.
* **Performance Metrics:** Optimizes for high precision and recall, validated via AUC-ROC scores.

---

##  Neural Network Architecture

The solution implements a robust Deep Neural Network (DNN) designed for stability:
* **Input Layer:** Processes 28 PCA-transformed feature vectors.
* **Hidden Layers:**
  * Dense layers with ReLU activation.
  * Dropout Regularization to mitigate overfitting.
* **Output Layer:** Sigmoid activation for probability scoring.
* **Optimization:** Trained using the Adam optimizer and Binary Cross-Entropy loss.

---

##  Dataset Details

* **File:** `creditcard_2023.csv`
* **Privacy:** Fully anonymized data.
* **Features:** 28 numerical columns (PCA transformed) + Time & Amount.
* **Labels:** `0` (Legitimate) / `1` (Fraud).

---

##  Tech Stack

* **Language:** Python 3.8+
* **Deep Learning:** TensorFlow, Keras
* **Data Manipulation:** Pandas, NumPy, Scikit-learn
* **Analysis:** Matplotlib, Seaborn, SHAP
* **Version Control:** Git LFS (Large File Storage)

---

##  Getting Started

### Prerequisites
Ensure Git LFS is installed on your local machine to properly pull and manage the dataset.

### Installation

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/AayushGala-git/Credit-Card-Fraud-Detection-Using-Deep-Learning.git](https://github.com/AayushGala-git/Credit-Card-Fraud-Detection-Using-Deep-Learning.git)
   cd Credit-Card-Fraud-Detection-Using-Deep-Learning
   
2. **Environment Setup:**
   ```bash
   python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

3. **Fetch Dataset via Git LFS:**
   ```bash
   git lfs install
   git lfs pull

## LICENSE:
This project is open-source and licensed under the MIT License.

