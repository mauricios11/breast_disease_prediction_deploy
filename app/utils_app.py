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
                st.warning(f'⚠️ ERROR: Values in all attributes must be numerical\n'
                        f'- There are non-numerical values in {attribute}')
   
        return user_inputs
        
    #- func 01-#-#-#-#-#–#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def _process_file_input(self, uploaded_file, header=False) -> dict:
        """Procesa archivos subidos, tolerando valores faltantes y sin headers."""
        try:
            # 1. Leer archivo según su tipo
            file_name = uploaded_file.name.lower()
            if file_name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header= 0 if header else None, skipinitialspace=True)
            elif file_name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file, header=0 if header else None, engine='openpyxl')
            else:
                raise ValueError("Formato no soportado. Use CSV o Excel.")
            
            # 2. Asignar nombres de columnas si no hay headers
            if not header:
                if len(df.columns) != len(list_dicts.BASE_ATTRIBUTES):
                    raise ValueError("Número de columnas incorrecto. Verifique el archivo.")
                df.columns = list_dicts.BASE_ATTRIBUTES  # Asignar nombres predefinidos
            
            # 3. Validar columnas requeridas
            missing_cols = set(list_dicts.BASE_ATTRIBUTES) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Columnas faltantes: {', '.join(missing_cols)}")

            # 4. Procesar valores (agrupar todas las filas)
            processed_data = {attr: [] for attr in list_dicts.BASE_ATTRIBUTES}
            
            for _, row in df.iterrows():
                for attr in list_dicts.BASE_ATTRIBUTES:
                    val = row[attr]
                    if pd.notna(val) and isinstance(val, (int, float)):
                        processed_data[attr].append(val)

            return processed_data

        except Exception as e:
            st.error(f"🚨 Error al procesar archivo: {str(e)}")

    def _process_file_input(self, file, header= False)-> pd.DataFrame:
        """Process CSV file uploaded by the user to generate new data for prediction
           - IMPORTANT: The file VALUES (measures) must have -> 
             (1) The same order as self.base_attributes
             (2) If the column names (headers) are not added, indicate it with the check mark ✅
             (3) Is NOT necessary to add the same amount of values for each attribute.
                 NULL -> (without measure in any instance)
                -- (BUT must have at least 5 values and each of them must be separated by commas)
                -- example: if -> 'val,NULL ,val' then -> 'val',, 'val'
            Args:
            - file (file): not added this param yet
            - header (bool): if the file contains headers (important for the reading process)"""
        try:
            user_file = file.name.lower()
            if user_file.endswith('csv'):
                df = pd.read_csv(
                    file, skipinitialspace= True, header= 0 if header else None)
                
            elif user_file.endswith('xsl', 'xlsx'):
                df = pd.read_excel(
                    file, header= 0 if header else None, engine='openpyxl')
                
            else:
                raise ValueError('⚠️ ERROR: file extension not recognised,'
                                 'please add a CSV or Excel file')
            
            # validation: number of columns
            if len(df.columns) != len(list_dicts.BASE_ATTRIBUTES):
                raise ValueError(f'⚠️ ERROR: CSV file does not contain the required columns\n',
                                 f'- MESSAGE: in adition, make sure they are sorted as needed')
            
            # sort columns with order needed
            df.columns = list_dicts.BASE_ATTRIBUTES
            
            # null treatment
            df = df.fillna('missing')
            dict_nulls = df.to_dict('list')
            dict_new_data = {key: [val for val in values if val != 'missing']
                             for key, values in dict_nulls.items()}
            
            return dict_new_data
        
        except Exception as e:
            st.error(f'ERROR: file not processed {e}')
                 
    #- func 02-#-#-#-#-#-#–#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#   
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
        # sort in ascending order using only the three largest values -> [-3:]
        se    = np.std(measurements_arr, ddof= 1) / np.sqrt(len(measurements_arr))
        # ddof -> "delta degrees of freedom" -> '1' SAMPLE std, '0' -> POPULATION std

        results = {'mean': mean, 'se': se, 'worst': worst}
        return results
    
    #- func 03-#-#-#-#-#-#–#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
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
            min_range, max_range = list_dicts.RANGES.get(attribute, (None, None))
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
        df_new_data = pd.DataFrame([features], columns= list_dicts.ORIGINAL_COLUMNS) 
        return df_new_data
        
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


