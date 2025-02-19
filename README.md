# Breast Cancer Prediction App

#### 🔬 Medical AI-based tumor classification using digitized FNA samples

This project aims to build a predictive model that classifies breast tumors as benign (B) or malignant (M) using measurements from digitized fine needle aspirate (FNA) images. The model prioritizes high recall for malignant cases, ensuring minimal false negatives—critical in a medical context.

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

