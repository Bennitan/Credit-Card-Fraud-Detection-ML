# 💳 Credit Card Fraud Detection Pipeline

A production-ready **Machine Learning pipeline** developed to accurately classify highly imbalanced credit card transactions as fraudulent or genuine. This project highlights proficiency in feature scaling, handling severe class imbalance, and performance metric prioritization for high-risk applications.

**Status:** Completed | **Model:** Logistic Regression | **Focus Metric:** Recall

---

### 2. 🎯 Key Technical Challenges & Solutions

This section showcases the unique engineering decisions made to solve a real-world problem.

| Challenge | Your Solution (The Engineering Decision) |
| :--- | :--- |
| **Severe Class Imbalance** | Applied **SMOTE (Synthetic Minority Over-sampling Technique)** *exclusively to the training data* to synthetically balance the classes, preventing model bias toward the majority class. |
| **Feature Normalization** | Used **StandardScaler** to normalize the `Time` and `Amount` features, ensuring all inputs were on the same scale for effective model convergence. |
| **Metric Prioritization** | Intentionally shifted the success metric from simple Accuracy (which was misleading) to **Recall** for the Fraud Class (Class 1) to minimize costly **False Negatives** (missed fraud). |

---

### 3. 📈 Final Performance Results

These are the numbers that back up your claims, based on the final model evaluation on the unseen test set.

| Metric | Class 1 (Fraud) | Interpretation |
| :--- | :--- | :--- |
| **Recall** | **0.92** (92%) | The model successfully **caught 92%** of all actual fraud cases on the unseen test data. |
| **Precision** | **0.06** (6%) | Demonstrates the trade-off: Accepting a low precision (many **False Positives** / false alarms) to guarantee high recall and minimize financial risk. |
| **F1-Score** | **0.12** | The harmonic mean confirms the intentional bias towards high recall. |

**Confusion Matrix (Sample from Test Set):**

[[TN (Non-Fraud Correct) FP (False Alarm)]
[FN (Fraud Missed) TP (Fraud Caught)]]
[[55270 1592] [ 8 90]]

### 4. 💻 Technical Stack & Project Structure

Demonstrate that you use professional tools and structure your code logically.

* **Languages & Libraries:** Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE).
* **Project Structure:**
    ```
    FRAUD_DETECTION_PROJECT/
    ├── data/               # Contains the raw creditcard.csv (Ignored by Git due to size)
    ├── src/
    │   └── data_preprocessing.py # Complete ML pipeline script (Preprocessing, SMOTE, Training, Evaluation)
    ├── .gitignore          # Properly excludes large files (>100MB) and virtual environments.
    └── requirements.txt    # Lists all required library dependencies.
    ```

---

### 5. ⚙️ Setup and Execution

Provide clear instructions for running the project.

#### 1. Requirements

* Clone this repository.
* **Data Requirement:** Download `creditcard.csv` (143MB) from [Kaggle](https://www.kaggle.com/datasets/mlg-gza/creditcardfraud) and place it in the local `data/` folder.

#### 2. Install Dependencies

```bash
python3 -m venv venv_fraud
source venv_fraud/bin/activate
pip install -r requirements.txt
