# functions to be used in the app
# path: 

# libraries
import streamlit as st
import joblib
import numpy  as np
import pandas as pd
from   typing import List, Tuple, Dict

#own modules
from list_dicts_text import ListDictText

#-#-#-#-#–#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
list_dicts = ListDictText()

class DeploymentFuncs:
    def __init__(self):
        self.model_path = './models/stacking_model_14.pkl'

    #- func 01-#-#-#-#-#–#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def process_manual_input(self)->Dict:
        user_inputs = {}
        for attribute, default_vals in list_dicts.DEFAULT_VALUES.items():
            user_input = st.text_input(
                f'{attribute.upper()} threshold accepted: {list_dicts.RANGES[attribute]}',
                value= default_vals)
            # convert input [str] to list[float]
            try:
                values_list = [float(val.strip()) for val in user_input.split(',')
                               if val.strip()]
                if len(values_list) < 5:
                    st.warning(f'⚠️ WARNING: At least 5 values are required in {attribute}')
                    st.stop()
                user_inputs[attribute] = values_list
                        
            except ValueError:
                raise ValueError(
                    f'⚠️ ERROR: Values in all attributes must be numerical\n'
                    f'- There are non-numerical values in {attribute}')

   
        return user_inputs
        
    #- func 02-#-#-#-#-#–#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def _process_file_input(self, uploaded_file, header=False) -> dict:
        """"""
        try:
            # (1) validation: correct file & read it
            file_name = uploaded_file.name.lower()
            if file_name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header= 0 if header else None, skipinitialspace=True)
            elif file_name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file, header=0 if header else None, engine='openpyxl')
            else:
                raise ValueError("⚠️ Format not supported-> (csv, xls, xlsx)")
            
            # (2) add default columns names*
            if not header:
                if len(df.columns) != len(list_dicts.BASE_ATTRIBUTES):
                    raise ValueError('⚠️ The amount of Columns does not match'
                                     'with the required number')
                df.columns = list_dicts.BASE_ATTRIBUTES  #*
            
            # (3) validation: rquired columns
            missing_cols = set(list_dicts.BASE_ATTRIBUTES) - set(df.columns)
            if missing_cols:
                raise ValueError(f'⚠️ Missing columns: {", ".join(missing_cols)}')

            # (4) group by rows
            processed_data = {attr: [] for attr in list_dicts.BASE_ATTRIBUTES}
            
            for _, row in df.iterrows():
                for attr in list_dicts.BASE_ATTRIBUTES:
                    val = row[attr]
                    if pd.notna(val) and isinstance(val, (int, float)):
                        processed_data[attr].append(val)

            return processed_data

        except Exception as e:
            st.error(f'⚠️ ERROR: Could not process the file: {str(e)}')
             
    #- func 03-#-#-#-#-#-#–#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#   
    def _calculate_stats(self, measurements: List[int|float])-> Dict:
        """Calculate mean, Standard Error & worst(mean of the three largest values)
            Args:
            - measurements (List): list of measurements converted to numpy array

            Returns:
            - Dict: dictionary with mean, se and worst values
            >>> measurements = [1, 2, 3, 4, 5]
            >>> calculate_mean_se_worst(measurements)"""    
        measurements_arr = np.array(measurements)
        mean  = np.mean(measurements_arr)
        worst = np.mean(np.sort(measurements_arr)[-3:]) 
        se    = np.std(measurements_arr, ddof= 1) / np.sqrt(len(measurements_arr))
        # ddof -> "delta degrees of freedom" -> '1' SAMPLE std, '0' -> POPULATION std

        results = {'mean': mean, 'se': se, 'worst': worst}
        return results
    
    #- func 04-#-#-#-#-#-#–#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def _process_values(self, user_data: Dict)-> pd.DataFrame:
        """ """
        features = {}
        for attribute, measurements in user_data.items():
            # validations
            if not all(isinstance(val, (int, float)) for val in measurements):
                raise ValueError(f'⚠️ ERROR: Values in {attribute} must be numerical')
                
            if len(measurements) < 5:
                raise ValueError(
                    '⚠️ ERROR: At least 5 measurements are required in each attribute')
                
            # checkout ranges -> (if not found return -> None)
            min_range, max_range = list_dicts.RANGES.get(attribute, (None, None))
            if min_range is not None and max_range is not None:
                if not all(min_range <= val <= max_range for val in measurements):
                    raise ValueError(
                        f'⚠️ ERROR: Values in "{attribute.upper()}" are out of range'
                        f'\n- Value expected between {min_range}|{max_range})')
                
            # calculate metrics
            stats = self._calculate_stats(measurements)
                
            # column names
            features[f'{attribute}_mean'] = stats['mean']
            features[f'{attribute}_se']   = stats['se']
            features[f'{attribute}_worst']= stats['worst']
                
        # df (original column order)
        df_new_data = pd.DataFrame([features], columns= list_dicts.ORIGINAL_COLUMNS) 
        return df_new_data
    
    #- func 05-#-#-#-#-#-#–#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#    
    def prediction(self, uploaded_file= None,
                   user_input= None,
                   header= False
                   )-> Tuple[str, float, float]:
        if uploaded_file:
            dict_new_data = self._process_file_input(uploaded_file, header= header)
        else:
            dict_new_data = user_input
        
        df_processed = self._process_values(user_data= dict_new_data)
            
        # prediction & probabilities
        model = joblib.load(self.model_path)
        model.set_params(smote= 'passthrough')
            
        pred = model.predict(df_processed)
        pred_result = 'MALIGNANT' if pred == 1 else 'BENIGN'
            
        # remembering the way we mapped target column: {'B': 0, 'M': 1} 
        prob_benign, prob_malignant = model.predict_proba(df_processed)[0] #-> [[B, M]]
            
        st.write(f'The prediction with the entered values is: {pred_result}')
        st.write(f'Probabilities:')
        st.write(f'BENIGN: {prob_benign:.3%}')
        st.write(f'MALIGNANT: {prob_malignant:.3%}')
            
        return pred_result, prob_benign, prob_malignant


