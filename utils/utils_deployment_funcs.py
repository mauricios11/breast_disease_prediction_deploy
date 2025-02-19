#
#

# libraries
import joblib
import numpy as np
import pandas as pd
#import streamlit as st
from typing import List, Dict, Tuple

#-#-#-#-#-#-#-#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
class DeploymentFuncs:
    def __init__(self)-> None:
        self.model_path = '../models/stacking_model_14.pkl'
        
        self.BASE_ATTRIBUTES = [
            'radius', 'texture', 'perimeter', 'area','smoothness',
            'compactness', 'concavity', 'concave points',
            'symmetry', 'fractal_dimension'
        ]
        
        self.RANGES = {
            # range calculation: min/~2, max * ~2
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
        # sorted columns (order used in training)
        self.ORIGINAL_COLUMNS = [
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
    
    def _process_file_input(self, uploaded_file, header= False)-> pd.DataFrame:
        """Process CSV file uploaded by the user to generate new data for prediction
           - IMPORTANT: The file VALUES (measures) must have -> 
             (1) The same order as self.base_attributes
             (2) If the column names (headers) are not added, indicate it with the check mark ✅
             (3) Is NOT necessary to add the same amount of values for each attribute.
                 NULL -> (without measure in any instance)
                -- (BUT must have at least 5 values and each of them must be separated by commas)
                -- example: if -> 'val,NULL ,val' then -> 'val',, 'val'
            Args:
            - uploaded_file (file): not added this param yet
            - header (bool): if the file contains headers (important for the reading process)"""
        try:
            if uploaded_file.endswith('.csv'):
                df = pd.read_csv(uploaded_file,
                                 skipinitialspace= True,
                                 header= 0 if header else None) # estudiar cómo funciona esto y porqué 0
                
            elif uploaded_file.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file,
                                   header = 0 if header else None)
            
            # validation: number of columns
            if len(df.columns) != len(self.BASE_ATTRIBUTES):
                raise ValueError(f'⚠️ ERROR: CSV file does not contain the required columns\n',
                                 f'- MESSAGE: in adition, make sure they are sorted as needed')
            
            # sort columns with order needed
            df.columns = self.BASE_ATTRIBUTES
            
            # null treatment
            df = df.fillna('nan')
            dict_nulls = df.to_dict('list')
            dict_new_data = {key: [val for val in values if val != 'nan']
                             for key, values in dict_nulls.items()}
            
            return dict_new_data
                
        except Exception as e:
            print(e)
        #     st.error(f'ERROR: can not process the CSV file: {e})
        
        
    #- func 01-#-#-#-#-#-#–#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#   
    def _calculate_stats(self, measurements: List[int|float])-> Dict:
        """Calculate mean, Standard Error & worst(mean of the three largest values)
        Args:
            measurements (List): list of measurements converted to numpy array

        Returns:
            Dict: dictionary with mean, se and worst values
        >>> measurements = [1, 2, 3, 4, 5]
        >>> calculate_mean_se_worst(measurements)"""
        
        measurements_arr = np.array(measurements)
        mean  = np.mean(measurements_arr)
        worst = np.mean(np.sort(measurements_arr)[-3:])
        ## sort in ascending order using only the three largest values -> [-3:]
        
        se    = np.std(measurements_arr, ddof= 1) / np.sqrt(len(measurements_arr))
        # ddof -> "delta degrees of freedom" -> '1' SAMPLE std, '0' -> POPULATION std

        results = {'mean': mean, 'se': se, 'worst': worst}

        return results
    
    #- func 02-#-#-#-#-#-#–#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def _process_values(self, user_data: Dict)-> pd.DataFrame:
        """ """
        features = {}
        for attribute, measurements in user_data.items():
            # validations
            if not all(isinstance(val, (int, float)) for val in measurements):
                raise ValueError(f'⚠️ ERROR: Values in {attribute} must be numerical')
            
            if len(measurements) < 5:
                raise ValueError('⚠️ ERROR: At least 5 measurements are required in each attribute')
            
            # checkout ranges -> (if not found return -> None)
            min_range, max_range = self.RANGES.get(attribute, (None, None))
            if min_range is not None and max_range is not None:
                if not all(min_range <= val <= max_range for val in measurements):
                    raise ValueError(f'⚠️ ERROR: Values in "{attribute.upper()}" are out of range'
                                     f'\n- Value expected between {min_range}|{max_range})')
            
            # calculate metrics
            stats = self._calculate_stats(measurements)
            
            # column names
            features[f'{attribute}_mean'] = stats['mean']
            features[f'{attribute}_se']   = stats['se']
            features[f'{attribute}_worst']= stats['worst']
            
        # df (original column order)
        df_new_data = pd.DataFrame([features], columns= self.ORIGINAL_COLUMNS)
        
        return df_new_data
    
    def prediction(self, uploaded_file= None,
                   frontend_input = None,
                   header= False)-> Tuple[str, float, float]:
        if uploaded_file is not None:
            dict_new_data = self._process_file_input(uploaded_file, header)
            
        else:
             dict_new_data = frontend_input 
        
        df_processed = self._process_values(user_data= dict_new_data)
        
        # prediction & probabilities
        model_path = '../models/stacking_model_14.pkl'
        model = joblib.load(model_path)
        model.set_params(smote= 'passthrough')
        
        pred = model.predict(df_processed)
        pred_result = 'MALIGNANT' if pred == 1 else 'BENIGN'
        
        # remembering the way we mapped target column: {'B': 0, 'M': 1} 
        prob_benign, prob_malignant = model.predict_proba(df_processed)[0] #-> [[B, M]]
        
        print(f'The prediction with the entered values is: {pred_result}')
        print(f'Probabilities:')
        print(f'- benign   : {prob_benign:3%}')
        print(f'- malignant: {prob_malignant:.3%}')
        
        return pred_result, prob_benign, prob_malignant
        
        
        
         
        # 
        
            
        
        
    
    
    
    
                                              
      
        