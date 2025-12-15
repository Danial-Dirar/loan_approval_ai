# 🏦 AI-Powered Loan Approval Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📄 Overview
This project is a Machine Learning pipeline designed to automate the loan eligibility process. It analyzes applicant data to predict whether a loan should be **Approved** or **Rejected**. 

The system implements multiple classification algorithms to determine the most accurate model, performing comprehensive data preprocessing, feature scaling, and visualization of performance metrics.

## 📊 Features
* **Automated Preprocessing:** Handles missing values and encodes categorical variables.
* **Exploratory Data Analysis (EDA):** Visualizes class distribution and feature correlations.
* **Multi-Model Comparison:** Trains and evaluates four distinct algorithms:
    * K-Nearest Neighbors (KNN)
    * Logistic Regression
    * Decision Tree Classifier
    * Neural Network (MLPClassifier)
* **Adaptive Scaling:** Applies `MinMaxScaler` for distance-based algorithms (KNN) and `StandardScaler` for others.
* **Performance Visualization:** Generates Confusion Matrices, ROC Curves, and Accuracy comparisons.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (sklearn)

## 📂 Dataset
The project relies on `Loan Approval Dataset.csv`. 
* **Input Features:** Applicant demographics, credit history, income, etc.
* **Target Variable:** `loan_status` (Approved/Rejected).

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/loan-approval-ai.git](https://github.com/your-username/loan-approval-ai.git)
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn
    ```

3.  **Place your dataset:**
    Ensure `Loan Approval Dataset.csv` is in the root directory.

4.  **Run the script:**
    ```bash
    python main.py
    ```

## 🧠 Methodology

### 1. Data Preprocessing
* **Missing Values:** Imputed using the `most_frequent` strategy.
* **Encoding:** `LabelEncoder` is used to convert categorical text data into numerical format.
* **Splitting:** Data is split 70% for training and 30% for testing using **stratified sampling** to maintain class balance.

### 2. Model Training
The system compares four models. Logic is applied to use specific scalers for specific models:
* **KNN:** Uses `MinMaxScaler` (sensitive to data magnitude).
* **Logistic Regression, Decision Tree, Neural Network:** Use `StandardScaler`.

### 3. Evaluation Metrics
The models are ranked based on:
* **Accuracy:** Overall correctness.
* **Precision & Recall:** To understand False Positives vs False Negatives.
* **ROC AUC Score:** To measure the model's ability to distinguish between classes.

## 📈 Results & Visualizations

Upon running the script, the following visualizations are generated:

1.  **Class Distribution:** Checks for dataset imbalance.
2.  **Correlation Heatmap:** Identifies relationships between features.
3.  **Model Comparison Bar Charts:** Visualizes Accuracy, Precision, and Recall side-by-side.
4.  **Confusion Matrices:** Detailed breakdown of correct vs. incorrect predictions for each model.
5.  **ROC Curve:** Compares the True Positive Rate vs. False Positive Rate for all models.

*(Note: You can add screenshots of your plots here after running the code)*

## 🔮 Future Improvements
* Implement Hyperparameter Tuning (GridSearchCV) to optimize model performance.
* Deploy the best model using Streamlit or Flask as a web app.
* Add Feature Importance analysis (SHAP values) to explain *why* a loan was rejected.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
