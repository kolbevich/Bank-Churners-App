import json
import os
import pandas as pd
import pickle
from sklearn import metrics
import xgboost as xgb
import shutil
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

working_directory = os.getcwd()

ORIGINAL_TRAINING_DATASET_PATH = r'\Data\BankChurners_cleaned.csv'
CURRENT_TRAINING_DATASET_PATH = r'\Data\Training_set.csv'


class Model:
    def __init__(self, training_set_path=None):
        self.training_set_path = training_set_path
        self.path = self.create_new_model_folder()

    def create_new_model_folder(self, pathname=working_directory + r'/Models/Model', counter=0):
        if not os.path.exists(f'{pathname} {counter}'):
            os.makedirs(f'{pathname} {counter}')
            return f'{pathname} {counter}'
        else:
            return self.create_new_model_folder(f'{pathname}', counter + 1)

    def set_training_set_path(self, path):
        # скопировать файл из current
        df = pd.read_csv(working_directory+path, sep=';',  encoding='utf-8')
        training_path = self.path + r'/Training_set.csv'
        df.to_csv(training_path, sep=';',  encoding='utf-8')
        self.training_set_path = training_path

    def read_training_set(self):
        train_data = pd.read_csv(self.training_set_path, sep=';', encoding='utf-8')
        target = 'Attrition_Flag'

        X = train_data.drop(['CLIENTNUM'], axis=1)
        y = train_data[target]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    def modelfit(self, alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, \
                 early_stopping_rounds=50):
        target = 'Attrition_Flag'
        if useTrainCV:
            xgb_param = alg.get_xgb_params()
            xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target])
            xgtest = xgb.DMatrix(dtest[predictors].values)
            cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                              metrics='auc', early_stopping_rounds=early_stopping_rounds)  # кросс-валидация
            alg.set_params(n_estimators=cvresult.shape[0])

        # Fit the algorithm on the data
        alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')

        # Predict training set:
        dtrain_predictions = alg.predict(dtrain[predictors])
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

        # Save model report:
        self.test_roc_auc = metrics.roc_auc_score(dtrain[target], dtrain_predprob)
        self.test_f1 = metrics.fbeta_score(y_true=dtrain[target], y_pred=dtrain_predictions, beta=0.6)

        # Predict on testing data:
        dtest['pred'] = alg.predict(dtest[predictors])
        dtest['predprob'] = alg.predict_proba(dtest[predictors])[:, 1]

        self.train_roc_auc = metrics.roc_auc_score(dtest[target], dtest['predprob'])
        self.train_f1 = metrics.fbeta_score(y_true=dtest[target], y_pred=dtest['pred'], beta=0.6)

        return alg

    def get_scores(self):
        return {'Train ROC-AUC': self.train_roc_auc,
                'Train F1': self.train_f1,
                'Test ROC-AUC': self.test_roc_auc,
                'Test F1': self.test_f1
                }


def get_current_training_set_info():
    max = 0
    for file in os.listdir(working_directory + r'/Models'):
        if file.startswith("Model"):
            if int(file.split()[1]) > max:
                max = int(file.split()[1])
    training_set = pd.read_csv(f'{working_directory}/Models/Model {str(max)}/training_set.csv', sep=';',  encoding='utf-8')
    return len(training_set)


def get_current_model_name():
    max = 0
    for file in os.listdir(working_directory + r'/Models'):
        if file.startswith("Model"):
            if int(file.split()[1]) > max:
                max = int(file.split()[1])
    return f'Model {str(max)}'


def add_training_data(df):
    combined_csv = pd.concat([pd.read_csv(working_directory+CURRENT_TRAINING_DATASET_PATH, sep=';',  encoding='utf-8'), df])
    combined_csv.to_csv(working_directory+CURRENT_TRAINING_DATASET_PATH, sep=';',   encoding='utf-8', index=False)



def get_current_model_scores():
    max = 0
    for file in os.listdir(working_directory + r'/Models'):
        if file.startswith("Model"):
            if int(file.split()[1]) > max:
                max = int(file.split()[1])
    return pd.read_json(f'{working_directory}/Models/Model {str(max)}/scores.json', typ='series')

# def create_fitted_model() -> object:
#     predictors = ['Total_Trans_Amt',
#                   'Total_Amt_Chng_Q4_Q1',
#                   'Total_Ct_Chng_Q4_Q1',
#                   'Total_Trans_Ct',
#                   'Total_Revolving_Bal']
#
#     xgbc = XGBClassifier(
#         learning_rate=0.1,
#         n_estimators=150,
#         objective='binary:logistic',
#         nthread=-1,
#         scale_pos_weight=1,
#         seed=27)
#
#     model = Model()
#     model.set_training_set_path()
#     model.read_training_set()
#     xgb_fitted = model.modelfit(xgbc, model.x_train, model.x_test, predictors)
#
#     pickle.dump(xgb_fitted, open(model.path + r'/attrition_clf.pkl', 'wb'))
#
#     with open(model.path + r'\scores.json', "w") as fp:
#         json.dump(model.get_scores(), fp)


# здесь мы впервые обучаем модель
#
# data_cleaned = pd.read_csv(working_directory+ORIGINAL_TRAINING_DATASET_PATH, sep=';', encoding='utf-8')
#
# target = 'Attrition_Flag'
#
# X = data_cleaned.drop(['CLIENTNUM'], axis=1)
#
# y = data_cleaned[target]
#
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# predictors = ['Total_Trans_Amt',
#               'Total_Amt_Chng_Q4_Q1',
#               'Total_Ct_Chng_Q4_Q1',
#               'Total_Trans_Ct',
#               'Total_Revolving_Bal']


def create_fitted_model(is_first=False):
    predictors = ['Total_Trans_Amt',
                  'Total_Amt_Chng_Q4_Q1',
                  'Total_Ct_Chng_Q4_Q1',
                  'Total_Trans_Ct',
                  'Total_Revolving_Bal']

    xgbc = XGBClassifier(
        learning_rate=0.1,
        n_estimators=150,
        objective='binary:logistic',
        nthread=-1,
        scale_pos_weight=1,
        seed=27)

    if is_first:
        model = Model()
        model.set_training_set_path(ORIGINAL_TRAINING_DATASET_PATH)
    else:
        model = Model()
        model.set_training_set_path(CURRENT_TRAINING_DATASET_PATH)

    model.read_training_set()
    xgb_fitted = model.modelfit(xgbc, model.x_train, model.x_test, predictors)

    pickle.dump(xgb_fitted, open(model.path + r'/attrition_clf.pkl', 'wb'))

    with open(model.path + r'\scores.json', "w") as fp:
        json.dump(model.get_scores(), fp)


# create_fitted_model(is_first=True)
