# main file for the app
# path: ./app/app.py

# libraries
import streamlit as st
import pandas as pd
import joblib
import os

#own modules
from utils_app       import DeploymentFuncs
from list_dicts_text import ListDictText

#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

#NOTE: the model used is in -> './models/stacking_model_12.pkl'
dep        = DeploymentFuncs()
list_dicts = ListDictText()




# cosas que faltan: mejorar la explicación, revisar redacción
# optimizar el código
# subir :D

if __name__ == '__main__':
    st.title('🔬 Breast cancer prediction')

    st.markdown(list_dicts.description_text)
    show_insights = st.checkbox(' Check this box to get more info after'
                                ' the prediction results')
    st.markdown(list_dicts.explanation_text)
    st.image('app/assets/example.png', width= 800)

    # option 1: upload the file
    use_header = st.checkbox(f'Does your file has a header?') 
    file_       = st.file_uploader(f'📂 Upload your file', type= ['csv', 'xls', 'xlsx'])
    ## file is a -> UploadedFile object not a string 'file.csv'
    button_01  = st.button('* prediction')

    if button_01 and file_:
        dep.prediction(uploaded_file= file_, header= use_header)
        if show_insights:
            st.write(list_dicts.insights_text)

    # option 2: manual entry
    st.markdown(list_dicts.entry_text)
    user_input= dep.process_manual_input()
    button_02 = st.button('- prediction')

    if button_02:
        dep.prediction(user_input= user_input)
        if show_insights:
            st.write(list_dicts.insights_text)
    

