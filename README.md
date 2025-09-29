## 📉 Customer Churn Prediction using Deep Learning
### 📌 Problem

Customer churn, when clients stop doing business with a company, it directly impacts revenue and growth. In this project, I developed a deep learning model to predict customer churn in a marketing dataset.

The challenge was severe class imbalance: most customers stay, but only a small minority churn. Without handling this, models tend to ignore churners, making them useless for retention strategies.

## 🛠️ Approach
🔹 Data Preparation

Cleaned missing values and encoded categorical variables.

Normalized continuous features for stable training.

Tackled imbalance using:

SMOTE oversampling to generate synthetic churn cases.

Class weights to penalize misclassification of churners.

Focal loss to make the model focus on hard-to-classify minority samples.

## 🔹 Model Architecture

Input: 30 features

Hidden layers: Fully connected layers with ReLU, Batch Normalization, and Dropout

Output: 1 neuron with sigmoid activation (probability of churn)

Loss: Binary crossentropy (baseline) and Focal loss (alternative for imbalance)

Optimizer: Adam

🔹 Training & Evaluation

Epochs: 30–50

Batch size: 32

Evaluation metrics: Accuracy, AUC, Precision, Recall, and F1-score
