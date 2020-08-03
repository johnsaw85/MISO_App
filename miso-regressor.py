# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 21:34:20 2020

@author: John Saw
"""
import os
import matplotlib
matplotlib.use('Agg')

#packages
import numpy as np
from numpy import arange, mean
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix

#sklearn packages
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

#streamlit
import streamlit as st
from streamlit import caching

#Model Finalize
import pickle

#place holder for dummy variables
tmppath = "C:\\Users\\"+os.environ.get('USERNAME')+"\\AppData\\Local\\Temp"
seed = 7
validation_size = 0.2
num_folds = 10
scoring = 'neg_mean_squared_error' # or r2

intro = """
<html>
This is a simple machine learning regression web app called MISO created to help lower entry barrier for every day users to perform regression studies on predicting a single output based on multiple inputs.
</html>
"""

body = """
<html>
Start with View Data for simple data exploration and/or new data frame creation.<br>
Next, use Build Model follow to get recommended regression model.<br>
End off with Predict by entering input values for a prediction.<br><br>
Note: Last Column assumed as Target.
</html>
"""

#@st.cache
def read_file(filename):
    dataset = pd.read_csv(filename)
    dataset.dropna(axis=0,how='any',thresh=None,subset=None,inplace=True)
    #dataset=pd.get_dummies(dataset,drop_first = True)
    X = dataset.iloc[:,0:-1]
    Y = dataset.iloc[:,-1]
    X = pd.get_dummies(X,drop_first=True)
    dataset = pd.concat([X,Y],axis=1)
    return dataset
# =============================================================================
#     allbutlast = dataset.iloc[:,:-1]
#     allbutlast = pd.get_dummies(allbutlast,drop_first = True)
#     dataset = pd.concat([allbutlast,lastcol])
# =============================================================================
    return dataset


def train_test(data):
    array = data.values
    X = array[:,:-1]
    Y = array[:,-1]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=validation_size, random_state=seed)
    return X_train, X_validation, Y_train, Y_validation


#caching.clear_cache()

def root():
    st.title("Machine Learning App")
    st.text("Created by John Saw ©")

    activities = ["About", "1. View Data", "2. Build Model", "3. Predict"]
    choice = st.sidebar.selectbox("Select Page",activities)
     
    dataset = st.sidebar.file_uploader("Upload Dataset",type=["csv","txt","xlsx"])
    
    if choice == 'About':
            st.subheader("**Intro**")
            st.markdown(intro, unsafe_allow_html=True)
            st.write("") 
            st.subheader("**How to use MISO?**")
            st.markdown(body, unsafe_allow_html=True)
    
    if dataset != None:
        df = read_file(dataset)
        
    if st.sidebar.checkbox("Change Data Frame or Subset") and (choice == '1. View Data' or choice == '2. Build Model' or choice == '3. Predict'):
        colList = df.columns.tolist()
        selCols = st.multiselect("Select Columns", colList)
        df = df[selCols]
    else:
        st.write()
    
        
    if 1+1 == 2:
        
        if choice == '1. View Data':
      
            if dataset is not None:
                #df = read_file(dataset)
                st.subheader("Review Data")
                if st.checkbox("View Data Table")and df.shape[1]>1:
                    st.dataframe(df)                    
                if st.checkbox("View Data Type(s)") and df.shape[1]>1:
                    df_types = df.dtypes.to_frame(name='Data Type')
                    df_types
                if st.checkbox("Important Features") and df.shape[1]>1:
                    X = df.iloc[:,0:-1]
                    Y = df.iloc[:,-1]
                    if Y.dtype == "float64":
                        rfr = RandomForestRegressor(random_state=seed)
                        rfr.fit(X,Y)
                        importances = rfr.feature_importances_
                        tbl = pd.Series(importances, index = X.columns)
                        feat_importances = tbl.sort_values(ascending=True)
                        fig = pyplot.figure()
                        fig.suptitle('Rank Features by Importance')
                        feat_plot = feat_importances.plot(kind='barh',figsize=(20,10))
                        st.pyplot()
                        st.dataframe(tbl.to_frame("Rank").sort_values(by='Rank', ascending=False))
                        
            else:
                st.write("Please load data first.")
        
        if choice == '2. Build Model':
            if dataset is not None and df.shape[1]>1:
                st.subheader("Let's Build Your Model")
                X_train, X_validation, Y_train, Y_validation = train_test(df)
                #Baseline Models Evaluation
                models = []
                #models.append(('LR', Pipeline([('Scaler', StandardScaler()), ('Un_LR', LinearRegression())])))
                #models.append(('LASSO', Pipeline([('Scaler', StandardScaler()), ('Un_LASSO', Lasso())])))
                #models.append(('EN', Pipeline([('Scaler', StandardScaler()), ('Un_EN', ElasticNet())])))
                #models.append(('SVR', Pipeline([('Scaler', StandardScaler()), ('Un_SVR', SVR())])))
                
                models.append(('KNN', Pipeline([('Scaler', StandardScaler()), ('Un_KNN', KNeighborsRegressor())])))
                models.append(('RFR', Pipeline([('Scaler', StandardScaler()), ('Un_RFR', RandomForestRegressor(random_state=seed))])))
                models.append(('GBR', Pipeline([('Scaler', StandardScaler()), ('Un_GBR', GradientBoostingRegressor(random_state=seed))])))
                models.append(('ETR', Pipeline([('Scaler', StandardScaler()), ('Un_ETR', ExtraTreesRegressor(random_state=seed))])))
                models.append(('ABR', Pipeline([('Scaler', StandardScaler()), ('Un_ABR', AdaBoostRegressor(random_state=seed))])))
                
                #models.append(('CART', Pipeline([('Scaler', StandardScaler()), ('Un_CART', DecisionTreeRegressor(random_state=seed))])))
                
                results = []
                names = []
                Model_Frame = []
                
                for name, model in models:
                    kfold = KFold(n_splits = num_folds, random_state = seed)
                    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring = scoring)
                    results.append(cv_results*(-1))
                    names.append(name)
                    Model_Frame.append({'Model': name, 'MSE':cv_results.mean(), 'StdDev':cv_results.std() })
                Model_Frame = pd.DataFrame(Model_Frame)
                
                if st.checkbox("Show Models"):
                    fig = pyplot.figure()
                    fig.suptitle('Baseline Models Comparison by MSE')
                    ax = fig.add_subplot(111)
                    pyplot.boxplot(results, showfliers=False)
                    ax.set_xticklabels(names)
                    st.pyplot()
                    Model_Frame.sort_values(['MSE'],ascending=False, inplace=True)
                    Model_Frame['MSE'] = Model_Frame['MSE'].abs()
                    st.dataframe(Model_Frame)
                    
                    if st.checkbox("Tune Best Model"):
                        Best_Model = Model_Frame.iloc[0][0]
                        if Best_Model == "KNN":
                            st.write("KNeighborsRegressor(KNN)")
                            scaler = StandardScaler().fit(X_train)
                            rescaledX_train = scaler.transform(X_train)
                            param_grid_KNN = {'n_neighbors' : list(range(0, 10)), 'p': list(range(0, 10))}
                            best_model_KNN = KNeighborsRegressor()
                            kfold_KNN = KFold(n_splits = num_folds, random_state = seed)
                            grid_KNN = GridSearchCV(estimator = best_model_KNN, param_grid = param_grid_KNN, scoring = scoring, cv=kfold_KNN)
                            grid_KNN_result = grid_KNN.fit(rescaledX_train,Y_train)
                            st.write(abs(round(grid_KNN_result.best_score_,2)))
                            KNN_n_neighbors = list(grid_KNN_result.best_params_.values())[0]
                            KNN_p = list(grid_KNN_result.best_params_.values())[1]
                            
                            #Test Optimized Model
                            MSE = abs(round(grid_KNN_result.best_score_,2))
                            rescaledX_validation = scaler.transform(X_validation)
                            Y_prediction_ori = grid_KNN.predict(rescaledX_validation)
                            Corr = abs(round(r2_score(Y_validation,  Y_prediction_ori),2))
                            st.write("Post hyperparams tuning for KNN has MSE of {} and R^2 of {}.".format(MSE,Corr))
                            
                            filename = 'finalized_model.sav'
                            pickle.dump(grid_KNN, open(os.path.join(tmppath, filename), 'wb'))
                            
                        if Best_Model == "RFR":
                            st.write("RandomForestRegressor(RFR)")
                            scaler = StandardScaler().fit(X_train)
                            rescaledX_train = scaler.transform(X_train)
                            param_grid_RFR = {'n_estimators' : [50, 100, 150], 'min_samples_split': list(range(1, 4))}
                            best_model_RFR = RandomForestRegressor(random_state=seed)
                            kfold_RFR = KFold(n_splits = num_folds, random_state = seed)
                            grid_RFR = GridSearchCV(estimator = best_model_RFR, param_grid = param_grid_RFR, scoring = scoring, cv=kfold_RFR)
                            grid_RFR_result = grid_RFR.fit(rescaledX_train,Y_train)
                            st.write(abs(round(grid_RFR_result.best_score_,2)))
                            RFR_n_estimators = list(grid_RFR_result.best_params_.values())[0]
                            RFR_min_samples_split = list(grid_RFR_result.best_params_.values())[1]
                            
                            #Test Optimized Model
                            MSE = abs(round(grid_RFR_result.best_score_,2))
                            rescaledX_validation = scaler.transform(X_validation)
                            Y_prediction_ori = grid_RFR.predict(rescaledX_validation)
                            Corr = abs(round(r2_score(Y_validation,  Y_prediction_ori),2))
                            st.write("Post hyperparams tuning for RFR has MSE of {} and R^2 of {}.".format(MSE,Corr))
                            
                            filename = 'finalized_model.sav'
                            pickle.dump(grid_RFR, open(os.path.join(tmppath, filename), 'wb'))
        
                       
                        if Best_Model == "GBR":
                            st.write("GradientBoostingRegressor(GBR)")
                            scaler = StandardScaler().fit(X_train)
                            rescaledX_train = scaler.transform(X_train)
                            param_grid_GBR = {'n_estimators' : [50, 100, 150], 'min_samples_split': list(range(1, 4)),'learning_rate': [0.05, 0.1, 0.2]}
                            best_model_GBR = GradientBoostingRegressor(random_state=seed)
                            kfold_GBR = KFold(n_splits = num_folds, random_state = seed)
                            grid_GBR = GridSearchCV(estimator = best_model_GBR, param_grid = param_grid_GBR, scoring = scoring, cv=kfold_GBR)
                            grid_GBR_result = grid_GBR.fit(rescaledX_train,Y_train)
                            st.write(abs(round(grid_GBR_result.best_score_,2)))
                            GBR_n_estimators = list(grid_GBR_result.best_params_.values())[0]
                            GBR_min_samples_split = list(grid_GBR_result.best_params_.values())[1]
                            GBR_learning_rate = list(grid_GBR_result.best_params_.values())[2]
                            
                            
                            #Test Optimized Model
                            MSE = abs(round(grid_GBR_result.best_score_,2))
                            rescaledX_validation = scaler.transform(X_validation)
                            Y_prediction_ori = grid_GBR.predict(rescaledX_validation)
                            Corr = abs(round(r2_score(Y_validation,  Y_prediction_ori),2))
                            st.write("Post hyperparams tuning for GBR has MSE of {} and R^2 of {}.".format(MSE,Corr))
                            
                            filename = 'finalized_model.sav'
                            pickle.dump(grid_GBR, open(os.path.join(tmppath, filename), 'wb'))
                            
                        if Best_Model == "ETR":
                            st.write("ExtraTreesRegressor(ETR)")
                            scaler = StandardScaler().fit(X_train)
                            rescaledX_train = scaler.transform(X_train)
                            param_grid_ETR = {'n_estimators' : [50, 100, 150], 'min_samples_split': list(range(2, 4))}
                            best_model_ETR = ExtraTreesRegressor(random_state=seed)
                            kfold_ETR = KFold(n_splits = num_folds, random_state = seed)
                            grid_ETR = GridSearchCV(estimator = best_model_ETR, param_grid = param_grid_ETR, scoring = scoring, cv=kfold_ETR)
                            grid_ETR_result = grid_ETR.fit(rescaledX_train,Y_train)
                            ETR_n_estimators = list(grid_ETR_result.best_params_.values())[0]
                            ETR_min_samples_split = list(grid_ETR_result.best_params_.values())[1]
                            
                            #Test Optimized Model
                            MSE = abs(round(grid_ETR_result.best_score_,2))
                            rescaledX_validation = scaler.transform(X_validation)
                            Y_prediction_ori = grid_ETR.predict(rescaledX_validation)
                            Corr = abs(round(r2_score(Y_validation,  Y_prediction_ori),2))
                            st.write("Post hyperparams tuning for ETR has MSE of {} and R^2 of {}.".format(MSE,Corr))
                            
                            filename = 'finalized_model.sav'
                            pickle.dump(grid_ETR, open(os.path.join(tmppath, filename), 'wb')) 
        
                        if Best_Model == "ABR":
                            st.write("AdaBoostRegressor(ETR)")
                            scaler = StandardScaler().fit(X_train)
                            rescaledX_train = scaler.transform(X_train)
                            param_grid_ABR = {'n_estimators' : [50, 100, 150],'learning_rate': [0.05, 0.1, 0.2]}
                            best_model_ABR = AdaBoostRegressor(random_state=seed)
                            kfold_ABR = KFold(n_splits = num_folds, random_state = seed)
                            grid_ABR = GridSearchCV(estimator = best_model_ABR, param_grid = param_grid_ABR, scoring = scoring, cv=kfold_ABR)
                            grid_ABR_result = grid_ABR.fit(rescaledX_train,Y_train)
                            st.write(abs(round(grid_ABR_result.best_score_,2)))
                            ABR_n_estimators = list(grid_ABR_result.best_params_.values())[0]
                            ABR_learning_rate = list(grid_ABR_result.best_params_.values())[1]
                            
                            #Test Optimized Model
                            MSE = abs(round(grid_ABR_result.best_score_,2))
                            rescaledX_validation = scaler.transform(X_validation)
                            Y_prediction_ori = grid_ABR.predict(rescaledX_validation)
                            Corr = abs(round(r2_score(Y_validation,  Y_prediction_ori),2))
                            st.write("Post hyperparams tuning for ETR has MSE of {} and R^2 of {}.".format(MSE,Corr))
                            
                            filename = 'finalized_model.sav'
                            pickle.dump(grid_ABR, open(os.path.join(tmppath, filename), 'wb'))
    
            else:
                st.write("Please load data first.")

                
        if choice == '3. Predict':
                if dataset is not None and df.shape[1]>1 and os.path.isfile('finalized_model.sav'):
                    
                    st.subheader("Let's Predict")
                    filename = 'finalized_model.sav'
                    
                    loaded_model = pickle.load(open(os.path.join(tmppath, filename), 'rb'))
                    #Dynamically display input boxes and compile input table
                    oridf = df
                    df = df.iloc[:,:-1]
                    df = df.astype(float)
                    #st.write(oridf)
                    cols = df.columns
                    var = []
                    for col in cols:
                        var.append(st.number_input(col,min_value =df[col].min() , max_value =df[col].max() , key = df.columns.get_loc(col)))
                        #var.append(st.number_input(col,format="%.5f",min_value =df[col].min() , max_value =df[col].max() , key = df.columns.get_loc(col)))
                        #var.append(st.text_input(col, key = df.columns.get_loc(col)))
                        
                    var_table = pd.DataFrame(var)
                    col_table = pd.DataFrame(df.columns.tolist())
                    
                    InputTable = pd.concat([col_table, var_table], axis=1, sort=False)
                    InputTable = InputTable.T
                    InputTableHeader = InputTable.iloc[0]
                    InputTable = InputTable[1:]
                    InputTable.columns = InputTableHeader
                    # InputTable = InputTable.astype(float)
                    
                    scaler = StandardScaler().fit(InputTable)
                    rescaledInputTable = scaler.transform(InputTable)
                    
                    #temp
                    #st.write("Number of row is " + str(rescaledInputTable.shape[0]))
                    #st.write("Number of column is " + str(rescaledInputTable.shape[1]))

                    Y_prediction = ""
                    if st.button("Predict"):
                        Y_prediction = loaded_model.predict(rescaledInputTable)
                        st.write("Predicted value for {} is {}.".format(oridf.columns[-1],round(Y_prediction[0]),4))
                
    
                else:
                    st.write("Please load data and save model first.")
              
                    
if __name__ == '__main__':
    root()