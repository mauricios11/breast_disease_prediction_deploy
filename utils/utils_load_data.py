# module made to load data
# current path: .utils/load_data.py

#libraries
import os
import pandas as pd
from typing import Dict

class Loader:
    def __init__(self) -> None:
        self.df_obj     = '../data'
        self.data_raw   = f'{self.df_obj}/raw'
        self.data_clean = f'{self.df_obj}/clean'
        
    #-func 1-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    
    def load_data(self, file_name: str,
                  dir: str= 'clean',
                  copy= True,
                  header= True):
        """load csv files"""
        if not dir in ['raw', 'clean']:
            raise ValueError('ERROR: The location selected must be either  "raw" | "clean"')
        
        dir_path = os.path.join(self.df_obj, dir) 
        
        files_list = os.listdir(dir_path)
        match_name = [file for file in files_list if file_name in file] # matching file_name with files
        
        if not match_name:
            raise ValueError(f'ERROR: file not found -> {file_name}')
        
        if len(match_name) >1:
            raise ValueError(f'ERROR: More than one file found with {file_name} name')
        
        target_file = match_name[0] # 1º (and only) element in match_name list
        
        if not header:
            df_data = pd.read_csv(os.path.join(dir_path, target_file), header= None)
            
        else:
            df_data = pd.read_csv(os.path.join(dir_path, target_file))
            
        #print(f'dir_path: {dir_path}\ntarget_file: {target_file}\nmatch_name{match_name}')
        return df_data.copy() if copy else df_data
    
    def save_dataframe(self, df: pd.DataFrame,
                       file_name: str,
                       dir: str= 'clean')-> None:
        """Save a dataframe in a csv file
            Args:
            - df (pd.DataFrame): dataframe to be saved
            - file_name (str)..: name of the file
            - dir (str, optional): directory to save the file. Defaults to 'clean'.

            Raises:
            - ValueError:  must be a pandas DataFrame
            - ValueError:  location must be either 'raw' | 'clean' (directory_name)"""
        if not isinstance(df, pd.DataFrame):
            raise ValueError('ERROR: df must be a pandas DF')
        
        if not dir in ['raw', 'clean']:                             # verifying if the location is valid
            raise ValueError('ERROR: The location selected must be either "raw" | "clean"')
        
        df_path = os.path.join(self.df_obj, dir)                    # the path to save the file
        os.makedirs(df_path, exist_ok=True)                         # create the dir if it doesn't exist
        
        df.to_csv(os.path.join(df_path, f'{file_name}.csv'),        # saving the file
                  index= False)
        
        if os.path.exists(os.path.join(df_path,                     # verifying if the file was saved
                                       f'{file_name}.csv')):  
            return print(f'SUCCESS: File {file_name}.csv saved in {df_path}')
        
        else:
            return print(f'ERROR: File {file_name}.csv NOT saved in {df_path}')
        

        
        
        
    
        