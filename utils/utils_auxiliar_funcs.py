# module made for  auxiliary functions used in the project
# use a enviroment with imblearn installed (which is nos compatible with sklearn 0.24)
# current path: .utils/auxiliar_funcs.py

# libraries
import matplotlib.pyplot as plt
import numpy   as np
import pandas  as pd
import seaborn as sns

#from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline      import Pipeline as IMB_pipeline

from sklearn.decomposition                import  PCA
from sklearn.feature_selection            import  mutual_info_classif
from sklearn.linear_model                 import  LogisticRegression
from sklearn.model_selection              import (train_test_split as tts,
                                                  GridSearchCV,
                                                  StratifiedKFold,
                                                  KFold)
from sklearn.preprocessing                import  StandardScaler

from statsmodels.stats.outliers_influence import  variance_inflation_factor
from typing import List, Dict, Union, Tuple


#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

class AuxiliarFuncs:
    # def __init__(self):
    
    #-func 01-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def split_by_attribute(self, df: pd.DataFrame,
                           attributes: List[str])-> Dict[str, pd.DataFrame]:
        """ Takes a df and splits it into Dfs based on the base attributes indicated.
            - Each df will include target col (diagnosis) the cols that start with the
             attribute names. 
            Args:
            - df       : Original df
            - atributes: List with the base attributes to split the df.

        Returns: a dict with the DataFrames separated by the base attributes """
        dict_dfs = {}
        for attribute in attributes:
            filter_cols = [col for col in df.columns 
                           if col.startswith(attribute) or col == 'diagnosis']
            
            df_attribute        = df[filter_cols]
            dict_dfs[attribute] = df_attribute
            
        return dict_dfs
    
    #-func 02-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def find_zero_vals(self, df: pd.DataFrame,
                       return_all: bool= True)-> pd.DataFrame | List[str]:
        """ Identifies the cols with values >= 0.0 on the df.
        - Count them and return a df with the results
            Args:
            - df: df to be analyzed
            Returns:
            - (1) df with the columns and the number of >= 0.0 values
            - (2) df and two lists col names and row index >= 0.000"""
            
        zero_count_list = [] # auxiliar list (containing a dict)
        col_names = []
        row_zero_index = []
        
        for col in df.columns:
            zero_rows = df[df[col] <= 0.000].index.tolist()
                       
            if zero_rows:
                if return_all:
                    col_names.append(col)
                    row_zero_index.append(zero_rows)
                                 
                zero_count_list.append({'col_name'      : col,
                                        'zero_count'    : len(zero_rows),
                                        'zero_rows_index': zero_rows})
                
        df_zero_summary = pd.DataFrame(zero_count_list)
        return_data = df_zero_summary, col_names, row_zero_index
        
        return return_data if return_data else df_zero_summary
    
    #-func 03-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def calc_problematic_props(self, df:pd.DataFrame,
                               cols_list: List[str],
                               target_col: str)-> pd.DataFrame:
        """Calculate the proportion of rows with zero values in the dataset.
           - Considering that: problematic columns are only in benign cases"""
        
        problematic_rows = df[df[cols_list].eq(0.0).any(axis=1)]# any value == 0.0
        total_rows = len(df)                                    # len(df) faster than df.shape[0]
        total_benign = len(df[df[target_col] == 'B'])
        
        # proportions
        total_problematic      = len(problematic_rows)
        total_problematic_prop = round(total_problematic / total_rows * 100, 2)
        
        benign_problematic     = len(problematic_rows[problematic_rows[target_col] == 'B'])
        benign_problematic_prop= round(benign_problematic / total_benign *100, 2)
        
        result =  {'Total rows'                     : total_rows,
                   'Total problematic rows'         : total_problematic,
                   'Proportion (problematic)'       : f'{total_problematic_prop}%',
                   'Total benign'                   : total_benign,
                   'Proportion benign (problematic)': f'{benign_problematic_prop}%'}
        
        df_result = pd.DataFrame(result, index= [0]).T
    
        return df_result
           
    #-func 04-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def calculate_VIF(self, df: pd.DataFrame, cols: List[str])-> pd.DataFrame:
        """ Calculate the VIF for the given columns in a df
        Args:
            - df
            - cols (list): Columns where to calculate the VIF
        Returns: df with the VIF values"""
        X = df[cols]
        vif_data = pd.DataFrame({'Column': cols,
                                 'VIF'   : [variance_inflation_factor(X, i)
                                            for i in range(X.shape[1])]})
        return vif_data
    
     #-func 05-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#  
    def calculate_mutual_information(self, df: pd.DataFrame,
                                     cols: List[str],
                                     target_col: str,
                                     top_n: int = None,
                                     normalize:bool = False)-> pd.DataFrame:
        """Calculate the mutual information between the target column and the given columns"""
        cols = [col for col in cols if col != target_col] # exclude target col
        
        if normalize: 
            data_normalized, scaler = self.normalize_data(df= df, cols_to_scale= cols)
            X = data_normalized[cols]
            y = data_normalized[target_col]
            
        else:
            X = df[cols]
            y = df[target_col]
            scaler = None
        
        mi = mutual_info_classif(X, y, discrete_features= False)
        mi_results = ( pd.DataFrame({'Column': cols, 'mutual_info': mi})
                       .sort_values(by= 'mutual_info', ascending= False) )
        
        if top_n is not None:
            mi_results = mi_results.head(top_n)
    
        print(f'Mutual Information calculated for {len(cols)} columns')
        
        return (mi_results, scaler) if normalize else mi_results

    #-#-#-#-#-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    #- MACHINE LEARNING FUNCTIONS -#
    #-#-#-#-#-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    
    #-func 06-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def normalize_data(self, df, cols_to_scale: List[str]
                       )-> Union[pd.DataFrame, StandardScaler]:
        """Normalizes the specified columns of the DF using StandardScaler
           Returns: 
           - df with normalized columns
           - fitted StandardScaler object"""
    
        # scaled X's
        scaler    = StandardScaler()
        df_scaled = df.copy()
        df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
        
        return df_scaled, scaler
 
    #-func 07-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def plot_explained_variance(self, explained_variance_ratio: np.array,
                                cumulative_variance: np.array,
                                explained_variance_target: float= 0.95,
                                dark_mode: bool= False):
        """Analize & plot the explained variance for a given PCA model
           - (method to be used inside the "perform_pca" method)
           Args:
                - explained_variance_ratio: explained variance for each component
                - cumulative_variance: cumulative explained variance
                - explained_variance_target: target explained variance"""
            
        #plot
        if dark_mode:
            plt.style.use('seaborn-darkgrid')
        fig, axes = plt.subplots(1,2, figsize= (15, 5))
        optimal_k = np.argmax(cumulative_variance >= explained_variance_target)+1
        
        # (1st subplot) individual & cumulative variance 
        axes[0].bar(range(1, len(explained_variance_ratio)+1),# x's
                    explained_variance_ratio,                 # y's
                    alpha= 0.6, color= 'skyblue',
                    label= 'Individual Variance')
        axes[0].plot(range(1, len(cumulative_variance)+1),#x's
                     cumulative_variance,                 #y's
                     marker= 'o', color= 'tomato',
                     label= 'Cumulative Variance')
        axes[0].axhline(y= explained_variance_target, color= 'green', linestyle= ':',
                        label= f'Goal: {explained_variance_target}')
        axes[0].axvline(x= optimal_k,color= 'purple', linestyle= '--', 
                        label= f'K: {optimal_k}')
        axes[0].set_xlabel('Number of Components', fontsize= 16, color= 'gray')
        axes[0].set_ylabel('Explained Variance',   fontsize= 16, color= 'gray')
        axes[0].set_title( 'Explained Variance (Screeplot)',
                           fontsize= 18, fontweight= 'bold', loc= 'left', color= 'gray')
        axes[0].legend()
        
        # (2nd subplot) "Zoom" in the cumulative variance
        axes[1].plot(range(1, len(cumulative_variance)+1),#x's
                     cumulative_variance,                 #y's
                     marker= 'o', color= 'tomato')
        axes[1].axhline(y= explained_variance_target, color= 'green', linestyle= ':')
        axes[1].axvline(x= optimal_k,color= 'purple', linestyle= '--', 
                        label= f'K: {optimal_k}')
        axes[1].set_xlabel('Number of Components', fontsize= 16, color= 'gray')
        axes[1].set_ylabel('Cumulative Explained Variance', fontsize= 16, color= 'gray')
        axes[1].set_title( 'Cumulative Explained Variance',
                           fontsize= 18, fontweight= 'bold', loc= 'left', color= 'gray')
        axes[1].grid()
        
        plt.tight_layout()
        plt.show()
        #params to be added: save_plot=True, filepath='variance_plot.png'
        return None
    
    #-func 08-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def analyze_pca_loadings(self, pca: PCA, feature_names: list,
                             plot: bool = True,
                             plot_method: str= 'heatmap') -> pd.DataFrame:
        """Analyze the PCA loadings to understand which features contribute the most
           to each component.
            Args:
            - pca: Trained PCA model.
            - feature_names: List of original feature names. (exclude target column)
            - plot_method: Method to plot the loadings. Options: 'heatmap', 'clustermap'
            - plot: Whether to generate a heatmap of the loadings.

            Returns:
                DataFrame containing the loadings for each principal component."""
        # Extract PCA loadings (components)
        loadings = pd.DataFrame(pca.components_.T,
                                index=feature_names,
                                columns=[f'PC{i+1}' for i in range(pca.n_components_)] )

        # Sort features by their absolute contribution to each PC
        sorted_loadings = loadings.apply(lambda col: col.abs().sort_values(ascending=False))
        # with absolute values, focusing only on the magnitude of the loadings and not on the direction -1 or 1

        if plot and plot_method in ['heatmap', 'clustermap']:
            # Create a heatmap of the loadings
            plt.figure(figsize=(12, 8))
            if plot_method == 'clustermap':
                sns.clustermap(sorted_loadings, cmap= "coolwarm",
                            annot= True, fmt= ".2f", cbar= True)
                
            if plot_method == 'heatmap':
                sns.heatmap(loadings, annot= True, cmap= "coolwarm",
                            fmt= ".2f", cbar= True, 
                            xticklabels= loadings.columns,
                            yticklabels= loadings.index)
                plt.title("PCA Loadings Heatmap")
                plt.xlabel("Principal Components")
                plt.ylabel("Original Features")
                
            plt.show()
        else:
            raise ValueError
        return sorted_loadings
    
    #-func 09-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def perform_data_balancing(self, X_raw, y_raw,
                               sampler,
                               pca_variance_target: float= 0.95,
                               random_state: int= 42,
                               plot_variance: bool= False,
                               return_preprocesors: bool= False):
        """Build and run a complete data balancing pipeline using the provided sampler
           (for example, SMOTE or BorderlineSMOTE).
           - Make sure that the data does NOT contain nulls / infinite vals
           - This method does NOT handle categorical data
            Process:
            (1) Splitting data into training and test (with stratification)
            (2) Normalizing data
            (3) Balancing (resampling) with the sampler
            (4) Dimensionality reduction using PCA (preserves the indicated variance)
            (5) Training a classifier (LogisticRegression) evaluated by GridSearchCV

            Args:
                - X_raw (DataFrame)---: Raw data (without the target column)
                - y_raw (Series)------: Target variable
                - sampler-------------: Resampling object (SMOTE | BorderlineSMOTE)
                - pca_variance_target : Variance threshold for PCA (default 0.95)
                - plot_variance (bool): If True, display the cumulative variance plot
                - return_preprocesors (bool): If True, return the scaler and PCA objects

            Returns:
                - grid_search------------: Optimized GridSearchCV object
                - Xt_balanced (DataFrame): Data from training transformed and balanced (result of scaler, sampler and PCA)
                - X_test (DataFrame)-----: Original test set
                - yt_balanced (Series)---: Balanced target variable of the training set
                - y_test (Series)--------: Target variable of the test set
                if return_preprocesors=True:
                  - scaler-----------------: Fitted StandardScaler object
                  - pca--------------------: Fitted PCA object
        >>> grid_search, Xt_balanced, X_test, yt_balanced, y_test, scaler, pca = perform_data_balancing(
             ...     X_raw, y_raw, sampler= SMOTE(random_state=42), pca_variance_target=0.95,
             ...     random_state=42, plot_variance=True, return_preprocesors=True)"""
             
        # (1) data split
        X_train, X_test, y_train, y_test = tts(X_raw, y_raw,
                                               stratify= y_raw,
                                               test_size= 0.2,
                                               random_state= random_state)
        
        # (3) pipeline (usin imblearn, NOT sklearn)
        imb_pipeline = IMB_pipeline(steps= [
            
            ('scaler', StandardScaler()),               # data normalization
            ('sampler', sampler),                       # balancing process
            ('pca', PCA(n_components= pca_variance_target)),# dimensionality reduction
            ('classifier', LogisticRegression(random_state=random_state, 
                                              max_iter= 10000))
        ])
        
        # (5) param grid for GridSearch
        params_grid = {'sampler__k_neighbors': [3,5,7],                     # for SMOTE
                       'classifier__C'     : [0.001, 0.01, 0.1, 1, 10, 100],# 
                       'classifier__solver': ['liblinear', 'lbfgs']}        # solver
        
        # (4) cv strategy for data validation: (stratifiedKFold)
        skf = StratifiedKFold(
            # 5 split to avoid overfitting and unbalanced data
            n_splits= 5, shuffle= True, random_state= random_state)
        
        # (6) GridSearch
        grid_search = GridSearchCV(estimator = imb_pipeline,
                                   param_grid= params_grid,
                                   scoring   = 'roc_auc',
                                   cv    = skf,
                                   n_jobs= -1) # all processors
        
        # (7) fit with training data
        grid_search.fit(X_train, y_train)
        
        # (8) balanced data
        best_estimator = grid_search.best_estimator_
        scaler   = best_estimator.named_steps['scaler']
        sampler_ = best_estimator.named_steps['sampler']
        pca      = best_estimator.named_steps['pca'] 
        
        print(f'- ✅ Best validation score (CV) with SMOTE: {grid_search.best_score_}')
        print(f'- 🔍 Best Hyperparameters:\n {grid_search.best_params_}')
        print('-*-' *10)
        
        Xt_scaled      = scaler.transform(X_train) # xt= X_train
        Xt_resampled, yt_resampled = sampler_.fit_resample(Xt_scaled, y_train)
        Xt_transformed = pca.transform(Xt_resampled)

        Xt_balanced = pd.DataFrame(Xt_transformed,
                                   # index compatility issues with the original data
                                   columns= [f'PC_{i+1}' for i in range(Xt_transformed.shape[1])],
                                   index  = range(0, Xt_transformed.shape[0])) # new indices
        
        yt_balanced = pd.Series(yt_resampled, index=range(0, len(yt_resampled)))
        
        if plot_variance:
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            self.plot_explained_variance(
                explained_variance_ratio= pca.explained_variance_ratio_,
                cumulative_variance     = cumulative_variance,
                explained_variance_target= pca_variance_target)
        
        if return_preprocesors:    
            return grid_search, Xt_balanced, X_test, yt_balanced, y_test, scaler, pca
        else:
            return grid_search, Xt_balanced, X_test, yt_balanced, y_test
    
    #-func 10-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
   
        
        
        
        
         
        
        
                               
            