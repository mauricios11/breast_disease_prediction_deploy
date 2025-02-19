# This file contains methods that were discarded of the project.
# path: ./utils/discarted_methods.py

# libraries

import matplotlib.pyplot as plt
import numpy   as np
import pandas  as pd
import seaborn as sns
import streamlit as st
import warnings


from sklearn.decomposition                import  PCA
from sklearn.feature_selection            import (mutual_info_classif,
                                                  SelectFromModel)
from sklearn.linear_model                 import  LogisticRegression
from sklearn.model_selection              import (train_test_split as tts,
                                                  GridSearchCV,
                                                  StratifiedKFold,
                                                  KFold,
                                                  cross_val_score,
                                                  cross_validate)
from sklearn.preprocessing                import  StandardScaler

from statsmodels.stats.outliers_influence import  variance_inflation_factor
from typing import List, Dict, Union, Tuple
from utils.utils_deployment_funcs import DeploymentFuncs

#-#-#-#-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
deploy = DeploymentFuncs()

class DiscartedMethods:
    # def __init__(self):
    
    
    #-func 01-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
    def split_data(self, df, target_col, test_size= 0.3, random_state= 42): 
        """Splits the dataset into training and testing sets
            returns: X_train, X_test, t_train, y_test"""

        X = df.drop(columns= target_col)
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = tts(X, y,
                                                stratify= y,
                                                test_size= test_size,
                                                random_state= random_state,)
        # modificar función en caso de ser necesario
        return X_train, X_test, y_train, y_test
    
    #-#-#-#-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    #- FEATURE SELECTION FUNCS ----------------------------------------------
    #-#-#-#-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    
    #-func 02-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
    # METHOD DISCARTED DUE func 9 (lasso_feature_selection_stability) is more robust
    
    def lasso_feature_selection(self, X_train, y_train, cv= 5):
        """Performs feature selection using Lasso (L1 regularization) and cross-validation
            - it works best for slection features to be used in a Logistic Regression model
            returns:
            - selected features: List of selected column names
            - Lasso model: Trained Lasso model"""
        # Logistic Regression with L1
        lasso = LogisticRegression(penalty= 'l1', solver= 'liblinear', max_iter= 20000)
            
        # grid search to find the best alpha (C in Logistic Regression)
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        grid_search = GridSearchCV(estimator= lasso, param_grid= param_grid, cv= cv, scoring= 'accuracy')
                        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        selector = SelectFromModel(estimator= best_model, prefit= True)
            
        selected_features = X_train.columns[selector.get_support()]
            # selecfrommodel.get_support(): returns a boolean mask of the features selected
            
        print(f'Best alpha(C): {grid_search.best_params_["C"]}')
        print(f'Number of columns: {len(selected_features)}')
        print(f'Selected features: {selected_features.tolist()}')
            
        return selected_features, best_model
    
    #-func 03-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
    def lasso_feature_selection_stability(self, X, y, inner_cv= 5, outer_cv= 5
                                          )-> Union[pd.DataFrame, LogisticRegression]:
        """Repeats Lasso feature selection multiple time to asses stability
        of the selected features.
        - IMPORTANT: (X,y) params can be used with complete data (X=X, y=target)
          OR with train data only -> (X_train, y_train) second case is recommended.
        Args:
            - X, y .........: da features(X) and target(y) 
            - inner/outer_cv: inner and outer cross-validation splits for the Lasso model
            returns:
            - feature_counts: DF with the frequency of selection for each column"""
            
        df_feature_counts = pd.DataFrame(index= X.columns, columns=['selected_count'])
        df_feature_counts['selected_count'] = 0 # initialize counter
        
        #Outer loop: stratify data (StratifiedKFold)
        outer_skf = StratifiedKFold(n_splits= outer_cv, shuffle= True, random_state= 42)
        
        for train_index, test_index in outer_skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            #Inner loop: GridSearch + Lasso
            lasso = LogisticRegression(penalty='l1', solver= 'liblinear', max_iter= 20000)
            param_grid= {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            grid_search = GridSearchCV(estimator= lasso, param_grid= param_grid,
                                       scoring= 'accuracy', cv= inner_cv)
                                       # estimator: model to be trained
                                       # param_grid: hyperparameters to be tested
                                       # cv: cross-validation
                                       # scoring: metric to be optimized
                                       
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            
            # selected columns
            selector = SelectFromModel(estimator= best_model, prefit= True)
                                       # estimator: model to be used for feature selection
                                       # prefit= True: the model has already been trained
                                       
            selected_columns = X_train.columns[selector.get_support()]
                               # selecfrommodel.get_support(): returns a boolean mask of the features selected
                               
            # increase counter
            df_feature_counts.loc[selected_columns, 'selected_count'] += 1
        
        # convert to percentage and sort
        df_feature_counts['frequency%'] = df_feature_counts['selected_count'] / outer_cv * 100
        df_feature_counts = df_feature_counts.sort_values(by= 'frequency%', ascending= False)
        
        return df_feature_counts, best_model
    
    #-func 04-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
    def perform_pca(self, df: pd.DataFrame,
                    target_col: str,
                    explained_variance_target: float= 0.95,
                    plot_explained_variance: bool= True,
                    cv: int= 5,
                    update_pca_with_k: bool= False,
                    only_return_K: bool= False,
                    )-> Union[pd.DataFrame, PCA, Dict[str, float]]:
        """Perform PCA for dimensionality reduction
           - needs normalized data
           Args:
            - df.......................: Df with NORMALIZED data features and target col
            - target_col...............: target column
            - explained_variance_tartet: cumulative explained variance target 
                                         --(default= 0.95 explaining 95% of variance)
            - plot_explained_variance..: Plot the explained variance scree plot
            - updated_pca_with_k.......: If True, the PCA model will be updated with the optimal K
                                         -- use this if you want to use the PCA model to transform
                                         --(default= False)                                    
            - cv.......................: Number of folds to be used in cross validation 
                                         --(model evaluation)
            Returns: 
            - Tuple with:
                -- transformed_X (pd.DataFrame): Data transformed by PCA
                -- pca_model (PCA).............: PCA model
                -- scores (dict | pd.DataFrame): Dict with evaluation scores (accuracy, precision,
                                                 recall, f1)
            >>> df_X_reduced, pca_model, pca_scores = perform_pca(df, target_col)"""
        # (1) X, y
        X = df.drop(columns= target_col)
        y = df[target_col]
        
        # (2) Verification: if data is normalized
        if not np.allclose(X.mean(axis= 0),0 , atol= 1e-3) or not np.allclose(X.std(axis=0), 1, atol = 1e-3): 
        # np.allclose(x.mean(axis= 0),0) OR ",1" -> returns True if all values are 0|1 within a tolerance
            warnings.warn('⚠️ Data seems to be NOT normalized (media≈0, std≈1) ⚠️',
                          UserWarning)
        
        # (3) PCA with all components (to find the optimal K) 
        pca_model = PCA(n_components= None)
        X_pca = pca_model.fit_transform(X)
        
        # (4) search for optimal K components for the explained_variance_target
        cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
            # cumsum: returns the cumulative sum of the elements along a given axis
            # pca.explained_variance_ratio_: returns the variance of each component
                            
        k_optimal = np.argmax(cumulative_variance >= explained_variance_target) +1
                    # argmax: returns the indices of the maximum values along an axis
                    # cumulative_variance >= explained_variance_: returns True or False
                    ### in this case, what argmax is doing is to take the max value in
                    ### cumulative_variance list and compare it with explained_variance
                    # ..)+1 because the index of cumulative_variance list starts at 0
        
        if cumulative_variance[-1] < explained_variance_target:
             warnings.warn(f'⚠️ Maximum variance reached is: {cumulative_variance[-1]:.2f}⚠️',
                           UserWarning)
             k_optimal = len(cumulative_variance)
                    
        print(f'✅ Optimal number of components (K): {k_optimal} over {len(df.columns)-1} training columns')
        
         # (5) only the optimal components (reducing the data to K)
        if update_pca_with_k:
            pca_model           = PCA(n_components= k_optimal)                   # update value
            X_reduced           = pca_model.fit_transform(X)                     # new fit
            cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_) # update value
            if only_return_K:
                return k_optimal

        else:
            X_reduced = X_pca[:, :k_optimal] # all rows, columns from 0 to k_optimal

        # (6) model evaluation (Logistic Regression)
        model_logistic = LogisticRegression(max_iter= 20000,
                                            solver='liblinear',
                                            random_state= 42)
        
        scoring_params = ['accuracy', 'precision', 'recall', 'f1']
        scores = cross_validate(estimator= model_logistic,
                                X= X_reduced,
                                y= y,
                                cv= cv,
                                scoring= scoring_params)
        
        # (7) average scores
        avg_scores = {metric: np.mean(scores[f'test_{metric}'])
                      for metric in scoring_params}
        scores_df = pd.DataFrame(avg_scores, index= [0])
            # index=[0] ->  data is in the first row
            
        # (8) plot explained varianze
        if plot_explained_variance:
            self.plot_explained_variance(
                explained_variance_ratio = pca_model.explained_variance_ratio_,
                cumulative_variance      = cumulative_variance,
                explained_variance_target= explained_variance_target )
        
        df_X_reduced = pd.DataFrame(X_reduced) # df with reduced data   
        
        return df_X_reduced, pca_model, scores_df
    
    #-func 05-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
    def optimal_k(self, X: pd.DataFrame,
                  variance_target: float= 0.95,
                  scale_data: bool= True,
                  plot: bool= False):
        """Find the optimal number of components for PCA 
           based on the target explained variance
           Args:
            - X(DataFrame): array-like, data to be analized.
              --(X must exclude the target column)
              --(X must be normalized, if not, turn scale_data to True)
            - variance_target(float): target explained variance default= 0.95
            - scale_data(bool): if True, the data will be scaled before PCA
            - plot(bool): If True, a scree plot will be displayed
           Returns:
            - k_optimal(int): optimal number of components to explain the target variance"""
        # (1) scale data if needed
        if scale_data:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
        else:
            if not np.allclose(X.mean(axis=0), 0, atol=1e-3) or not np.allclose(X.std(axis=0), 1, atol=1e-3):
                warnings.warn('⚠️ WARNING: Data seems to be NOT normalized (media≈0, stf≈1)⚠️',
                              UserWarning)
                print('- turn "scale_data" to True to normalize the data')
    
            X_scaled = X.copy()
            
        # (2) PCA model to find the optimal K
        pca = PCA()
        pca.fit(X_scaled)
        
        #(3) explaned variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        k_optimal = np.argmax(cumulative_variance >= variance_target) +1
        
        print(f'✅ Optimal number of components (K) to reach {variance_target*100}%: {k_optimal}')
        print(f'over {len(X.columns)} training columns')
        
        if plot:
            self.plot_explained_variance(
                explained_variance_ratio = pca.explained_variance_ratio_,
                cumulative_variance      = cumulative_variance,
                explained_variance_target= variance_target
                )
        return k_optimal
    
    #-#-#–#–#–#–#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
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
            if len(df.columns) != len(deploy.BASE_ATTRIBUTES):
                raise ValueError(f'⚠️ ERROR: CSV file does not contain the required columns\n',
                                 f'- MESSAGE: in adition, make sure they are sorted as needed')
            
            # sort columns with order needed
            df.columns = deploy.BASE_ATTRIBUTES
            
            # null treatment
            df = df.fillna('missing')
            dict_nulls = df.to_dict('list')
            dict_new_data = {key: [val for val in values if val != 'missing']
                             for key, values in dict_nulls.items()}
            
            return dict_new_data
        
        except Exception as e:
            st.error(f'ERROR: file not processed {e}')
            
    
    
    
    