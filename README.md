# Stroke Prediction using Decision Trees and XGBoost

This project aims to predict the likelihood of a stroke occurring using various health-related features of patients. The dataset used contains patient attributes such as age, gender, hypertension, glucose levels, and more. The goal is to train machine learning models, specifically **Decision Trees** and **XGBoost**, to predict whether a patient is at risk of having a stroke.

### **Table of Contents**
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Steps Involved](#steps-involved)
  - [Data Preprocessing](#data-preprocessing)
  - [Handling Class Imbalance with SMOTE](#handling-class-imbalance-with-smote)
  - [Model Training](#model-training)
    - [Decision Tree Classifier](#decision-tree-classifier)
    - [XGBoost Classifier](#xgboost-classifier)
  - [Model Evaluation](#model-evaluation)
- [How to Run](#how-to-run)
- [License](#license)

---

### **Project Overview**
This project builds predictive models to determine if a patient is at risk of having a stroke based on health-related features. We train two different machine learning models:
- **Decision Trees**: A simple yet interpretable model that splits the data based on feature values.
- **XGBoost**: A powerful gradient boosting machine learning algorithm that is robust to class imbalance and often delivers high performance.

### **Technologies Used**
- Python 3.x
- **XGBoost**: Gradient boosting library for decision trees.
- **Scikit-learn**: For Decision Trees and model evaluation.
- **Imbalanced-learn**: For handling class imbalance using SMOTE.
- **Pandas**: Data manipulation and preprocessing.
- **Matplotlib** (Optional, for visualization).

### **Dataset**
The dataset used in this project is the **Stroke Prediction Dataset**, which contains various health-related attributes like:
- `age`
- `gender`
- `hypertension`
- `heart_disease`
- `ever_married`
- `work_type`
- `Residence_type`
- `avg_glucose_level`
- `bmi`
- `smoking_status`
- `stroke` (target variable)

### **Steps Involved**

#### **Data Preprocessing**
1. **Handling Missing Data**: Missing values in the `bmi` column are filled using the median value.
2. **Label Encoding**: Categorical variables like `gender`, `ever_married`, `work_type`, `Residence_type`, and `smoking_status` are label-encoded into numerical values.
3. **Feature Selection**: The `id` column is removed as it is irrelevant for prediction.

#### **Handling Class Imbalance with SMOTE**
Since the dataset is imbalanced (fewer stroke cases than non-stroke cases), **SMOTE** (Synthetic Minority Over-sampling Technique) is used to generate synthetic samples for the minority class to balance the dataset.

#### **Model Training**
We train two machine learning models on the preprocessed dataset:

##### **Decision Tree Classifier**
1. **Model Configuration**: We use a **Decision Tree Classifier** with parameters like `max_depth` and `min_samples_split` to avoid overfitting and control the complexity of the model.
2. **Class Imbalance Handling**: While Decision Trees handle imbalances reasonably well, SMOTE is applied to ensure the model doesn’t favor the majority class (non-stroke).

##### **XGBoost Classifier**
1. **Model Configuration**: We use **XGBoost**, a gradient boosting algorithm known for its ability to handle class imbalance efficiently. The `scale_pos_weight` parameter is used to adjust the weight of the minority class during training.
2. **Handling Imbalance**: SMOTE is applied before training to further balance the dataset and ensure the model detects stroke cases more effectively.

#### **Model Evaluation**
The performance of both models is evaluated using the following metrics:
- **Accuracy**: The proportion of correct predictions.
- **Confusion Matrix**: To evaluate how well each model is predicting stroke and non-stroke cases.
- **Classification Report**: Includes precision, recall, F1-score, and support for each class (stroke and non-stroke).

### **How to Run**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stroke-prediction.git
   cd stroke-prediction
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Download the **Stroke Prediction Dataset** (CSV) and place it in the root directory.

4. Run the model:
   - For Decision Tree:
     ```bash
     python decision_tree_model.py
     ```
   - For XGBoost:
     ```bash
     python xgboost_model.py
     ```

5. Review the results:
   - The evaluation metrics for each model will be printed in the console, including **accuracy**, **confusion matrix**, and **classification report**.

### **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README now includes both the **Decision Tree** and **XGBoost** classifiers, describing their respective training processes, evaluation, and handling of class imbalance using SMOTE. You can further customize this based on the exact file names and specific configurations you used in your project. Let me know if you'd like any additional details or modifications!
