# Breast Cancer Prediction App

#### 🔬 Medical AI-based tumor classification using digitized FNA samples

This project aims to build a predictive model that classifies breast tumors as benign (B) or malignant (M) using measurements from digitized fine needle aspirate (FNA) images. The model prioritizes high recall for malignant cases, ensuring minimal false negatives—critical in a medical context.

#### 🚀 App Deployment

The model is deployed via Streamlit, allowing users to input tumor measurement values in two ways:

* Manual Entry: Users input 5-10 values per attribute, which are processed into _mean, _se, and _worst metrics.
* CSV Upload: Users can upload a pre-formatted dataset for batch predictions.

#### 🛠️ How to run the app | run the project locally
enter to: [test the app in streamlit](https://breasdiseaseprediction.streamlit.app/)

Clone this repository, and install rependencies *(pip)*:
```
git clone https://github.com/mauricios11/breast_disease_prediction_deploy.git
pip install -r requirements.txt

```

#### 📂 Dataset Overview

The dataset consists of $30$ **numerical columns** extracted from digitalized **FNA samples ob breast masses**
Each feature is computed using three statistical calculations:
* **Average**(`_mean`): average value across the sample
* **Standard Error**(`_se`): variability/error in the measurement.
* **Worst**(`_worst`): Mean of the three highest values per sample

#### 🔢 Features
Includes $10$ base attributes, each with its respective `_mean`, `_worst` and `_se` calculations

| Feature           | Description |
|------------------|-------------|
| **radius**       | Mean distance from the center of the nucleus to its perimeter. |
| **texture**      | Variation in grayscale pixel intensity within the nucleus. |
| **perimeter**    | Length of the nucleus boundary. |
| **area**         | Total area enclosed within the nucleus boundary. |
| **smoothness**   | Local variation in radius length. |

#### ⚖️ Class Distribution & Balancing
* Benign (B)$\rightarrow$`0`: $357$ samples *(original data)*
* Malignant (M)$\rightarrow$`1`: $212$ samples *(original data)*

####  Importance in medical context:
In medical applications *(especially on tumor detection)* it's critical to minimize false negatives. A $95$% recall for the malignant class means that the model fails to detect only $5$% of malignant cases, which is highly desirable.

**Overall (global) vs class comparison**: While overall recall $0.947$ is important, what we're really interested is the specific recall for the "malignant" class.
* If that metric reaches $95$% or higher, our goal is met *(even if the overall is slightly lower)*

**Strategy applied during the training**: Given the high cost of false negatives (FN) in medicine, focusing on maximizing the recall of the "malignant" class is a priority. This has been achieved by tuning class weights and using a custom scorer, which is a good practice in this context.

### 📊 Model Training Strategy
Has been used: Standarization (`StandardScaler`), Data Balancing(`SMOTE`), Dimensionality Reduction (`PCA` $<= 95$% of explained variance), Stacking Classifier combining:
* **Logistic Regression**(*interprebaility*), **Random Forest**(*captures interactions*), **XGBoostC**(*complex relationships*), **SVC**(*different kind of distributions*)

**Class Weights** adjusted to emphasize for malignant cases *(minority class)*:
* `class_weight= {0:1, 1:3}`, `scale_pos_weight=3`(XGBoost)





