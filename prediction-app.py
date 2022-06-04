# подумать над виджетами
# взять обученную модель из пикл файла
import os
# Look to the path of your current working directory
from random import randint

import streamlit as st
import streamlit_option_menu as option
import pandas as pd
import numpy as np
import pickle
from pandas_profiling import ProfileReport
from streamlit.state import get_session_state
from streamlit_pandas_profiling import st_profile_report
import model_fitter
from streamlit import legacy_caching



working_directory = os.getcwd()

selected_page = option.option_menu(
    menu_title=None,
    options=['Make prediction', 'Analyze data', 'Model info'],
    icons=['person-check', 'bar-chart-line-fill'],
    default_index=0,
    orientation='horizontal'
)


# @st.cache(allow_output_mutation=True)
# def get_state():
#     return []


if selected_page == 'Model info':
    st.write("""
        # Here you can see training results of current model
        # Also you can add new training data to improve the model
        """)

    st.write("Current model test results")
    if st.button('Get'):
        model_results = model_fitter.get_current_model_scores()
        st.write(model_fitter.get_current_model_name())
        st.write(model_results)
        st.write("Press get to refresh")
    else:
        st.write('Press get to see results')

    st.write("Training set info")
    if st.button('Info'):
        set_info = model_fitter.get_current_training_set_info()
        st.write('Training set file contains ', set_info, ' rows')
        st.write("Press info to refresh")
    else:
        st.write('Press get to see results')

    uploaded_file = st.sidebar.file_uploader("Upload a data to add it in Training set", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        # updating current
        model_fitter.add_training_data(input_df)
        # training new model on new current
        new_model = model_fitter.create_fitted_model()
        legacy_caching.clear_cache()

if selected_page == 'Make prediction':

    st.write("""
    # Bank Churners prediction
    This app predicts whether the Bank customer leaves or not!
    Data obtained from Kaggle

    'Wont leave': 0, 'Will leave': 1
    """)

    st.sidebar.header('User Input Features')

    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
    else:
        def user_input_features():
            CLIENTNUM = st.sidebar.text_input("Client id", "12345")
            total_trans_amt = st.sidebar.slider('Total_Trans_Amt', 0, 5000, 2500)
            total_amt_chng_q4_q1 = st.sidebar.slider('Total_Amt_Chng_Q4_Q1', 0, 4, 2)
            total_ct_chng_q4_q1 = st.sidebar.slider('Total_Ct_Chng_Q4_Q1', 0, 4, 2)
            total_trans_ct = st.sidebar.slider('Total_Trans_Ct', 0, 100, 50)
            total_revolving_bal = st.sidebar.slider('Total_Revolving_Bal', 0, 3000, 1500)
            data = {'CLIENTNUM': CLIENTNUM,
                    'Total_Trans_Amt': total_trans_amt,
                    'Total_Amt_Chng_Q4_Q1': total_amt_chng_q4_q1,
                    'Total_Ct_Chng_Q4_Q1': total_ct_chng_q4_q1,
                    'Total_Trans_Ct': total_trans_ct,
                    'Total_Revolving_Bal': total_revolving_bal}
            features = pd.DataFrame(data, index=[0])
            return features


        input_df = user_input_features()

    df = input_df  # Selects only the first row (the user input data)

    print(df)

    # Displays the user input features
    st.subheader('User Input features')

    if uploaded_file is not None:
        st.write(df)
    else:
        st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
        st.write(df)

    # Reads in saved classification model
    load_clf = pickle.load(open(working_directory + '\\' + 'attrition_clf.pkl', 'rb'))

    # Apply model to make predictions
    prediction = load_clf.predict(df.drop(['CLIENTNUM'], axis=1))
    prediction_proba = load_clf.predict_proba(df.drop(['CLIENTNUM'], axis=1)).round(4)
    print(pd.DataFrame(prediction_proba, columns=['0', '1']))


    def encode_proba(pp):
        df = pd.DataFrame(pp, columns=['0', '1'])
        df['Probability of leaving'] = 'Undefined'
        df['Probability of leaving'] = np.where((df['1'] >= 0) & (df['1'] < 0.2), 'low', df['Probability of leaving'])
        df['Probability of leaving'] = np.where((df['1'] >= 0.2) & (df['1'] < 0.4), 'average',
                                                df['Probability of leaving'])
        df['Probability of leaving'] = np.where((df['1'] >= 0.4) & (df['1'] < 0.6), 'above average',
                                                df['Probability of '
                                                   'leaving'])
        df['Probability of leaving'] = np.where((df['1'] >= 0.6) & (df['1'] < 0.8), 'high',
                                                df['Probability of leaving'])
        df['Probability of leaving'] = np.where((df['1'] >= 0.8) & (df['1'] <= 1), 'very high',
                                                df['Probability of leaving'])

        return df


    prediction_proba = encode_proba(prediction_proba)
    prediction_proba['CLIENTNUM'] = df['CLIENTNUM']
    prediction_proba.iloc[:, 0:2] = prediction_proba.iloc[:, 0:2].astype(float).round(4)
    st.subheader('Prediction Probability')
    st.write(prediction_proba)

    cat_choice = None
    cat_choice = st.selectbox('Filter predictions by probability of leaving',
                              [None, 'low', 'average', 'above average', 'high', 'very high'])
    if cat_choice is not None:
        choice_result = prediction_proba[prediction_proba['Probability of leaving'] == cat_choice]
        st.write(choice_result)
    else:
        pass

if selected_page == 'Analyze data':
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    """)

    # Pandas Profiling Report
    if uploaded_file is not None:
        @st.cache
        def load_csv():
            csv = pd.read_csv(uploaded_file, sep=';')
            return csv


        df = load_csv()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)
    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Example Dataset'):
            # Example data
            @st.cache
            def load_data():
                a = pd.DataFrame(
                    np.random.rand(100, 5),
                    columns=['a', 'b', 'c', 'd', 'e']
                )
                return a


            df = load_data()
            pr = ProfileReport(df, explorative=True)
            st.header('**Input DataFrame**')
            st.write(df)
            st.write('---')
            st.header('**Pandas Profiling Report**')
            st_profile_report(pr)
