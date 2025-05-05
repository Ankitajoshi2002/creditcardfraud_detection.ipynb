# 💳 Credit Card Fraud Detection

An intelligent machine learning system for detecting fraudulent credit card transactions using the Random Forest Classifier. This project applies data exploration, visualization, and modeling techniques to detect fraud from an imbalanced dataset.

<p align="center">
  <img src="https://user-images.githubusercontent.com/your-image-url/credit-card-fraud.png" alt="Credit Card Fraud Detection" width="70%">
</p>

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Evaluation Metrics](#evaluation-metrics)
- [Confusion Matrix](#confusion-matrix)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Author](#author)

---

## 📖 Overview

This project aims to detect fraudulent credit card transactions using supervised machine learning. The key challenge is handling the highly imbalanced dataset where fraud cases are significantly lower than legitimate ones.

---

## 🧾 Dataset

- 📂 **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 🔢 **Rows**: 284,807 transactions
- 📉 **Fraudulent transactions**: ~0.17%
- 🎯 **Features**:
  - `Time`, `V1` to `V28` (anonymized PCA components)
  - `Amount`, `Class` (0 = Valid, 1 = Fraud)

---

## 🧮 Confusion Matrix
<p align="center"> <img src="https://user-images.githubusercontent.com/your-image-url/confusion-matrix.png" alt="Confusion Matrix" width="50%"> </p>

---
## 🛠 How to Run

- **Mount Google Drive:**
python
Copy
Edit
from google.colab import drive
drive.mount('/content/drive')

- **Load the dataset:**
python
Copy
Edit
data = pd.read_csv("/content/drive/MyDrive/finaldataset.csv")

- **Run the notebook cells in sequence to:**

- **Perform EDA**

- **Train the model**

- **Evaluate performance**

- **Visualize results**

## 💻 Technologies Used
- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- Google Colab

## 🎯 Results
- The Random Forest model achieved high accuracy and recall.
- Successfully detected a large portion of fraudulent transactions while minimizing false positives.
- The model is suitable for real-world applications where reducing financial loss is critical.

## 👩‍💻 Author
Ankita Joshi

🎓 MCA Student, Graphic Era Deemed to be University, Dehradun

💼 Aspiring Data Scientist & ML Engineer

🌐 GitHub: ankitajoshi2002

"Detecting fraud isn't just a technical challenge—it's a necessity for securing our financial systems."

## 📌 License
This project is open-source and available under the MIT License.

## 📊 Exploratory Data Analysis

- Checked class distribution
- Visualized feature correlation with a heatmap
- Compared statistical summaries of fraud vs valid transactions

<p align="center">
  <img src="https://user-images.githubusercontent.com/your-image-url/heatmap.png" alt="Heatmap" width="60%">
</p>

---

## 🤖 Modeling

- **Model**: Random Forest Classifier
- **Split**: 80% training / 20% testing
- Handled class imbalance by evaluating precision and recall instead of just accuracy

---

## 📈 Evaluation Metrics

```python
Accuracy      : 99.9%
Precision     : High (reduces false positives)
Recall        : High (detects most frauds)
F1 Score      : Balanced metric
Matthews Corr.: Effective for imbalanced datasets
