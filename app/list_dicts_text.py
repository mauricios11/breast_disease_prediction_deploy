# module made for storing lists and dictionaries used in the deployment
# path:

import streamlit as st

class ListDictText:
    def __init__(self):
        self.BASE_ATTRIBUTES = [
            'radius', 'texture', 'perimeter', 'area','smoothness',
            'compactness', 'concavity', 'concave points',
            'symmetry', 'fractal_dimension'
        ]
        
        self.RANGES = {# range calculation: min/~2, max * ~2
            'radius'   : (1.0, 60),
            'texture'  : (1, 51),
            'perimeter': (9, 400),
            'area'     : (50, 5001),
            'smoothness' : (0.025, 0.32),
            'compactness': (0.00095, 0.68),
            'concavity'  : (0.0000, 0.8),
            'concave points': (0.0000, 0.4),
            'symmetry'      : (0.05, 0.6),
            'fractal_dimension': (0.02, 0.18) 
        }
        
        self.ORIGINAL_COLUMNS = [ # sorted columns (order used in training)
            'radius_mean', 'texture_mean', 'perimeter_mean',      # mean values
            'area_mean', 'smoothness_mean', 'compactness_mean',
            'concavity_mean', 'concave points_mean', 'symmetry_mean',
            'fractal_dimension_mean',     
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', # standard error values
            'smoothness_se',
            'compactness_se', 'concavity_se', 'concave points_se',    
            'symmetry_se', 'fractal_dimension_se', 'radius_worst',# worst values
            'texture_worst', 'perimeter_worst', 'area_worst',         
            'smoothness_worst', 'compactness_worst', 'concavity_worst',
            'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
        
        self.DEFAULT_VALUES = { # to put in the input boxes
            'radius'           : '6.3,  6.5,  6.1,  6.6,  6.2,  6.7,  6.8,  6.9,  6.0',
            'texture'          : '9.1,  12.8,  11.3,  10.8,  9.3,  9.3,  10.3,  11.2,  9.9,  11.7',
            'perimeter'        : '45.2,  36.1,  34.9,  47.0,  50.8,  44.7,  30.5,  48,  38.5,  49',
            'area'             : '200.3,  270.3,  240.2,  280.2,  230.1,  290.1,  200.1,  210.1,  220.1',
            'smoothness'       : '0.027011,  0.027012,  0.02702,  0.02707,  0.25073,  0.02701,  0.02706,  0.02703,  0.02705,  0.02709',
            'compactness'      : '0.14,  0.26,  0.11,  0.25,  0.12,  0.08,  0.55,  0.16,  0.24,  0.1',
            'concavity'        : '0.16,  0.33,  0.12,  0.15,  0.21,  0.19,  0.21,  0.12',
            'concave points'   : '0.24,  0.23,  0.25,  0.26,  0.26,  0.23,  0.24',
            'symmetry'         : '0.16,  0.26,  0.15,  0.22,  0.18,  0.23,  0.12,  0.29,  0.18',
            'fractal_dimension': '0.059,  0.054,  0.058,  0.059,  0.056,  0.054,  0.058,  0.058,  0.055'
        }
        
        self.description_text = """
This app allows you to predict whether a breast tumor is **Benign** or **Malignant**,
based on clinical measurements made on the tumor.
        
##### 🔄 How to enter the data?
You have two options:
* **File Upload:** By uploading a CSV or Excel file with the measurements. 
* **Manual Entry:** By typing the values of each attribute directly into the app.

**🔍 Want to know more about the process?** 
"""  

        self.explanation_text = """____
##### Option 1: ⬆️ upload file
Upload a file with the measurements. Making sure that:
* Each measurement **should be sorted** in the next order:
    * $(1)$ **radius**, $(2)$ **texture**, $(3)$ **perimeter**, $(4)$ **area**, $(5)$ **smoothness**, $(6)$ **compactness**,
    * $(7)$ **concavity**, $(8)$ **concave points**, $(9)$ **symmetry**, $(10)$ **fractal_dimension**
* **Turn on** the *check mark below* if the file **has a header**.

👇 Here's an example of how to add the values (CSV file): 
"""

        self.entry_text = """___        
##### Option 2: 🔢 entry measures manualy
The minimum of values required for each attribute is $5$ **values** and separated by a comma.
* *Example: 9.3, 9.5, 9.1, 9.6, 9.4, 9.2, 9.7, 9.8, 9.9, 9.0*        
"""

        self.insights_text ="""
##### 🧪 How was data analyzed?
The model has used **three types of calculations** to make the prediction: 

* 1️⃣ **Average**: Represents the average value of the measurements.
* 2️⃣ **Standar D. Error**: Indicates the variability in the measurements.  
* 3️⃣ **Greater severity (Worst)**: Calculated as the average of the three highest measurements.

Then whe get a calculation result for each attribute. 
* Example for the attribute 'radius': we obtain `radius_mean`, `radius_se`, `radius_worst`  

✅ These calculations are the result based on the  the measurements you have entered 
*(base attributes)* and finally evaluate the tumor."""
#mejorar texto de explicación y arreglar el problema al cargar el archivo