# This file contains the training functions for the model
# path: ./utils/training_funcs.py

# libraries
import numpy as np
import pandas as pd
import os
import joblib

from imblearn.pipeline       import Pipeline as imb_pipeline
from imblearn.over_sampling  import SMOTE

from sklearn.ensemble        import(StackingClassifier,
                                    RandomForestClassifier)
from sklearn.decomposition   import PCA
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import(accuracy_score,
                                    roc_auc_score,
                                    recall_score,
                                    classification_report,
                                    make_scorer)
from sklearn.model_selection import(train_test_split as tts,
                                    StratifiedKFold,
                                    GridSearchCV,
                                    RandomizedSearchCV,
                                    cross_val_score)
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler


#from sklearn.tree           import DecisionTreeClassifier
from sklearn.svm             import SVC # Support Vector Classifier

from xgboost import XGBClassifier
from typing  import List, Dict, Tuple, Union

#-#-#-#-#-#-#-#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

class TrainingFuncs:
    def __init__(self):
      self.random_state_ = 42 # random state (for reproducibility)
      
      self.param_dist_= {
        'stacking__xgb__n_estimators' : [100, 200, 300],
        'stacking__xgb__max_depth'    : [3],
        'stacking__xgb__learning_rate': [0.1, 0.3, 0.5],
        'stacking__rf__n_estimators'  : [50, 100, 200],
        'stacking__rf__max_depth'     : [None, 3, 5, 7],   
        'stacking__lr__C'             : [0.001, 0.01, 0.1, 1, 10],
        'stacking__final_estimator__C': [0.01, 0.1, 1, 10, 150, 250]
        }
    
    #- func 01 -#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def stacking_model_training(self, df: pd.DataFrame,
                                target_col: str,
                                output_file_name: str,
                                cv_n_iters: int= 50,
                                pca_n_components: int|float= 0.95,
                                return_df: bool= False,
                                output_model_dir: str = '../models/',
                                )-> Union[Dict, Dict]:
      """Trains a staking model using a Logistic Regression as meta model
         and XGBoost, Logistic Regression, Random Forest, and Support Vector Classifier
         as base models.
         - The model is trained using a Randomized Search for hyperparameter tuning.
          
         Uses: raw data (not scaled, nor balanced)
         Update: added class weights (for the base models) to improve RECALL

         Args:
          - df..............: pd.DataFrame, processed data
          - target_col......: str, target column name
          - output_file_name: str, name for the output model
          - test_size.......: float, test size for the train_test_split
          - pca_n_components: int|float, number of components for PCA. If float, it is the variance
          - cv_n_iters......: int, number of iterations ONLY for the Randomized Search
          - return_df.......: bool, if True, returns the balanced data (only for analysis)
          - output_model_dir: str, directory where the model will be saved
         Returns:
          - results: Dict, with the best pipeline, parameters, and metrics
          - xy_test: Dict, with X_test and y_test for validation
         >>> stackin_model, xy_test = stacking_model_training( 
         ... df, 'target', 'model_name', test_size= 0.2, cv_n_iters= 50,
         ... pca_n_components= 10, output_model_dir= 'model/dir/path/')"""
      # directory validation
      if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
      
      saving_path = f'{output_model_dir}{output_file_name}.pkl'
      
      # (1) X,y
      X = df.drop(columns= [target_col])
      y = df[target_col]
      
      # (2) data split
      X_train, X_test, y_train, y_test = tts(
        X, y, stratify= y, test_size= 0.2, random_state= self.random_state_)
      
      # (3) meta model (Logistic R)
      meta_model = LogisticRegression(
        random_state= self.random_state_, max_iter= 20000)#, class_weight= 'balanced')
      
      # (4) base models for stacking (XGBoost, LR, RF, SVC)
      weights = {0:1, 1:3} # class weights to improve recall
      base_estimators = [
        ('lr', LogisticRegression(
          random_state= self.random_state_, max_iter= 10000, class_weight= weights)), 
        
        ('xgb', XGBClassifier(
          random_state= self.random_state_, eval_metric= 'logloss', scale_pos_weight= 3) ),
        
        ('rf', RandomForestClassifier(
          random_state= self.random_state_, class_weight= weights)),
                         
        ('svc', SVC(
          random_state= self.random_state_, probability= True, class_weight= weights))
          # class_weight= 'balanced' -> automatic balance
          # class_weight= weights -> manual balance
        ]
      
      # (5) stacking model
      skf = StratifiedKFold(
        n_splits= 5, shuffle= True, random_state= self.random_state_)
      
      stacking_model = StackingClassifier(
        estimators     = base_estimators,
        final_estimator= meta_model,
        passthrough= False, # original columns are not concatenated
        cv = skf,           # uniform split categories
        n_jobs = -1)        # all processors
                                         
      # (6) pipeline: scaler, balance, PCA, stacking
      full_pipeline = imb_pipeline(steps= [
        ('scaler', StandardScaler()),
        ('smote' , SMOTE(random_state= self.random_state_)),
        ('pca'   , PCA(n_components= pca_n_components)), # if already know K, use it
        ('stacking', stacking_model),
      ])
      
      # (7) Cross valitadion: Randomized Search
          
      #-# recall scorer for class 1 (malignant)
      recall_scorer_malign = make_scorer(recall_score, pos_label= 1)
      
      #-# hyperparameter tuning (randomized search CV)
      random_search = (
        RandomizedSearchCV(
          estimator = full_pipeline,
          param_distributions= self.param_dist_,
          random_state = self.random_state_,
          n_iter = cv_n_iters,
          scoring= recall_scorer_malign,
          # change scoring to improve recall, before was ->'roc_auc',
          cv     = skf,
          verbose= 1,
          n_jobs = -1)
        )
      
      # (7) fit & results
      random_search.fit(X_train, y_train)
      best_pipeline = random_search.best_estimator_
      
      score_result = ('acceptable' if random_search.best_score_ > 0.85
                      else 'bad -> improvement needed')
          
      print(f'- Best SCORE:{random_search.best_score_}')
      print(f'-*- the performance score performance is {score_result}-*-')
      print(f'- Best params:\n{random_search.best_params_}, performance {score_result}')
      
      # (8) model evaluation: predictions using test
      y_pred  = best_pipeline.predict(X_test)
      y_proba = best_pipeline.predict_proba(X_test)[:,1] # all rows, column 1
      report  = classification_report(
        y_test, y_pred, target_names= ['benign', 'malignant'])
      
      # (9) metrics
      metrics = {
        'ACCURACY': accuracy_score(y_test, y_pred),
        'ROC AUC' : roc_auc_score(y_test, y_proba),
        'RECALL'  : recall_score(y_test, y_pred),
        'CLASSIFICATION REPORT': f'\n{report}'}     
        
      # (11) save best model
      joblib.dump(best_pipeline, saving_path)
      
      if os.path.exists(saving_path):
        print(f'✅ SUCCESS: model saved in -> {saving_path}')
      else:
        print(f'⚠️ ERROR: model not saved in -> {saving_path}')
      
      # (12) results & df
      results = {'stacking_pipeline': best_pipeline,
                 'params': random_search.best_params_,
                 'metrics': metrics}
      
      xy_test = {'X_test' : X_test, 'y_test' : y_test}
      
      if return_df:
        df = self._return_balanced_data( # helper method
          X_train, y_train, pipeline = best_pipeline)
        
      else:
        df = None
        
      return results, xy_test, df


    #- func 02 -#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def _return_balanced_data(self, X_train: pd.DataFrame,
                              y_train: pd.Series,
                              pipeline: Pipeline|imb_pipeline
                              )-> pd.DataFrame:
      """Returns a balanced DF within the training data.
         - HELPER method: used inside the training pipeline: stacking_model_training()
         - this DF is used for analysis and visualization ONLY
         Args:
          - X_train: pd.DataFrame, training data
          - y_train: pd.Series, target column
          - pipeline: trained pipeline (sklearn Pipeline|imblearn Pipeline)
         Returns:
          - df_balanced: pd.DataFrame, balanced data
         """
      # (1) Extract pipeline objects
      scaler_ = pipeline.named_steps['scaler']
      smote_  = pipeline.named_steps['smote']
      pca_    = pipeline.named_steps['pca']
      
      # (2) transform X_train & y_train
      Xt_scaled = scaler_.transform(X_train)
      Xt_resampled, yt_resampled = smote_.fit_resample(Xt_scaled, y_train)
      Xt_transformed = pca_.transform(Xt_resampled)
      
      # (3) DF for X's
      Xt_balanced = pd.DataFrame(
        Xt_transformed,
        columns= [f'PC_{i+1}' for i in range(Xt_transformed.shape[1])], # column names
        index= range(Xt_transformed.shape[0]) )
      
      # (4) DF for y
      yt_balanced = pd.Series(
        yt_resampled, index= range(len(yt_resampled)) )
      
      # (5) final DF
      df_balanced = pd.DataFrame(Xt_balanced)
      df_balanced['target_column'] = yt_balanced
      
      return df_balanced
      
  
    #- func 02 -#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def validate_pipeline(self, pipeline_model,
                          df: pd.DataFrame,
                          target_col: str,
                          focus_recall_category: str|int|bool,
                          cv= 5,
                          print_results: bool= True)->Dict:
      """Validates the pipeline using the test set.
        Args:
        
         """
      print('Validating the model with CV...')
      # (1) X,y
      X = df.drop(columns= target_col)
      y = df[target_col]
      
      # (2) split
      X_train, X_test, y_train, y_test = tts(
        X, y, stratify= y, test_size= 0.3, random_state= self.random_state_)
      
      # (fit) pipeline.fit(X_train...)
      # not needed if the model is already trained
      
      # (3) predictions
      y_pred = pipeline_model.predict(X_test)
      report = classification_report(
        y_test, y_pred, target_names= ['benign', 'malignant'])
      
      # (4) CV score & recall
      f1_scores = cross_val_score(
        pipeline_model, X, y, cv= cv, scoring= 'f1_macro', n_jobs= -1)
      # X and not X_train, because the model is already trained. estudiar esta parte
      recall_scores= recall_score(
        y_test, y_pred, pos_label= focus_recall_category)
      
      # (5) f1 score average -> to obtain one value
      print(f'F1-score (CV): {np.mean(f1_scores)}:.5f += {np.std(f1_scores):.5f}')
      # estudiar qué hacemos aquí 
      print(f'RECALL score: {recall_scores:.5f}')
      print(f'F1 macro: {np.mean(f1_scores):.5f}')
      
      # other metrics
      results = {
        'F1 macro': np.mean(f1_scores),
        'accuracy (test)': accuracy_score(y_test, y_pred),
        'recall class target (test)': recall_score,
        'classification_report': report
        
      }
      return results

      
    #- func 03 -#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    
      
      
      

      
      
        
        
            
            
            