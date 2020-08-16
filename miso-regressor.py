# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 21:34:20 2020

@author: John Saw
"""
import os
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

#wide mode enable
import io
from typing import List, Optional
import markdown

#packages
import numpy as np
from numpy import arange, mean
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix

#sklearn packages
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, cross_validate
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, make_scorer
from sklearn.inspection import permutation_importance

#streamlit
import streamlit as st
from streamlit import caching

#Model Finalize
import pickle

#place holder for dummy variables
seed = 7
validation_size = 0.2
num_folds = 5
scoring = 'neg_mean_squared_error'
COLOR = "black"
BACKGROUND_COLOR = "#fff"


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

Note:<br>
1. Last Column assumed as Target.<br>
2. Please provide dataset with more than 15 rows.

</html>
"""

#wide function start
class Cell:
    """A Cell can hold text, markdown, plots etc."""

    def __init__(
        self,
        class_: str = None,
        grid_column_start: Optional[int] = None,
        grid_column_end: Optional[int] = None,
        grid_row_start: Optional[int] = None,
        grid_row_end: Optional[int] = None,
    ):
        self.class_ = class_
        self.grid_column_start = grid_column_start
        self.grid_column_end = grid_column_end
        self.grid_row_start = grid_row_start
        self.grid_row_end = grid_row_end
        self.inner_html = ""

    def _to_style(self) -> str:
        return f"""
.{self.class_} {{
    grid-column-start: {self.grid_column_start};
    grid-column-end: {self.grid_column_end};
    grid-row-start: {self.grid_row_start};
    grid-row-end: {self.grid_row_end};
}}
"""

    def text(self, text: str = ""):
        self.inner_html = text

    def markdown(self, text):
        self.inner_html = markdown.markdown(text)

    def dataframe(self, dataframe: pd.DataFrame):
        self.inner_html = dataframe.to_html()
    
    def pyplot(self, fig=None, **kwargs):
        string_io = io.StringIO()
        pyplot.savefig(string_io, format="svg", fig=(2, 2))
        svg = string_io.getvalue()[215:]
        pyplot.close(fig)
        self.inner_html = '<div height="200px">' + svg + "</div>"

    def _to_html(self):
        return f"""<div class="box {self.class_}">{self.inner_html}</div>"""


class Grid:
    """A (CSS) Grid"""

    def __init__(
        self,
        template_columns="1 1 1",
        gap="10px",
        background_color=COLOR,
        color=BACKGROUND_COLOR,
    ):
        self.template_columns = template_columns
        self.gap = gap
        self.background_color = background_color
        self.color = color
        self.cells: List[Cell] = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        st.markdown(self._get_grid_style(), unsafe_allow_html=True)
        st.markdown(self._get_cells_style(), unsafe_allow_html=True)
        st.markdown(self._get_cells_html(), unsafe_allow_html=True)

    def _get_grid_style(self):
        return f"""
<style>
    .wrapper {{
    display: grid;
    grid-template-columns: {self.template_columns};
    grid-gap: {self.gap};
    background-color: {self.background_color};
    color: {self.color};
    }}
    .box {{
    background-color: {self.color};
    color: {self.background_color};
    border-radius: 5px;
    padding: 20px;
    font-size: 150%;
    }}
    table {{
        color: {self.color}
    }}
</style>
"""

    def _get_cells_style(self):
        return (
            "<style>"
            + "\n".join([cell._to_style() for cell in self.cells])
            + "</style>"
        )

    def _get_cells_html(self):
        return (
            '<div class="wrapper">'
            + "\n".join([cell._to_html() for cell in self.cells])
            + "</div>"
        )

    def cell(
        self,
        class_: str = None,
        grid_column_start: Optional[int] = None,
        grid_column_end: Optional[int] = None,
        grid_row_start: Optional[int] = None,
        grid_row_end: Optional[int] = None,
    ):
        cell = Cell(
            class_=class_,
            grid_column_start=grid_column_start,
            grid_column_end=grid_column_end,
            grid_row_start=grid_row_start,
            grid_row_end=grid_row_end,
        )
        self.cells.append(cell)
        return cell


def select_block_container_style():
    _set_block_container_style(max_width=1200)

def _set_block_container_style(
    max_width: int = 1200,
    max_width_100_percent: bool = False,
    padding_top: int = 5,
    padding_right: int = 1,
    padding_left: int = 1,
    padding_bottom: int = 10,
):
    if max_width_100_percent:
        max_width_str = f"max-width: 100%;"
    else:
        max_width_str = f"max-width: {max_width}px;"
    st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        {max_width_str}
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
        unsafe_allow_html=True,
    )
        
            
#wide function end


@st.cache(suppress_st_warning=True)
def read_file(filename):
    dataset = pd.read_csv(filename)
    return dataset

def preprocess_file(dataset):
    dataset.dropna(axis=0,how='any',thresh=None,subset=None,inplace=True)
    dataset.iloc[:,-1].astype(float)
    X = dataset.iloc[:,0:-1]
    Y = dataset.iloc[:,-1]
    X = pd.get_dummies(X,drop_first=True)
    dataset = pd.concat([X,Y],axis=1)
    return dataset

@st.cache
def train_test(data):
    array = data.values
    X = array[:,:-1]
    Y = array[:,-1]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=validation_size, random_state=seed)
    return X_train, X_validation, Y_train, Y_validation

def get_param(model, mode):
    if model == "KNN" and mode == 'Normal':
        param_grid = {'n_neighbors' : list(range(1, 10)), 'p': list(range(0, 10))}
        return param_grid
    elif model == 'RFR' and mode == 'Normal':
        param_grid = {'n_estimators' : [50, 100, 150], 'min_samples_split': list(range(1, 4))}
        return param_grid
    elif model == "GBR" and mode == 'Normal':
        param_grid = {'n_estimators' : [50, 100, 150], 'min_samples_split': list(range(1, 4)),'learning_rate': [0.05, 0.1, 0.2]}
        return param_grid
    elif model == 'ABR' and mode == 'Normal':
        param_grid = {'n_estimators' : [50, 100, 150],'learning_rate': [0.05, 0.1, 0.2]}
        return param_grid
    elif model == 'KNN' and mode == 'Abnormal':
        param_grid = {'n_neighbors' : list(range(1, 20)), 'p': list(range(0, 10))}
        return param_grid
    elif model == 'RFR' and mode == 'Abnormal':
        param_grid = {'n_estimators' : [50, 75, 100, 125, 150, 175, 200], 'min_samples_split': list(range(1, 4))}
        return param_grid
    elif model == 'GBR' and mode == 'Abnormal':
        param_grid = {'n_estimators' : [50, 75, 100, 125, 150, 175, 200], 'min_samples_split': list(range(1, 4)),'learning_rate': [0.05, 0.1, 0.2]}
        return param_grid
    elif model == 'ABR' and mode == 'Abnormal':
        param_grid = {'n_estimators' : [50, 75, 100, 125, 150, 175, 200],'learning_rate': [0.05, 0.1, 0.2]}
        return param_grid
    else:
        return None

def root():
    st.title("Machine Learning App")
    st.text("Created by John Saw ©")
    
    #initializers
    mode = 'Normal'
    key = "NG"
    
    #wide mode start
    select_block_container_style()   
    
    #wide mode end
    
    activities = ["About", "1. View Data", "2. Build Model", "3. Predict"]
    choice = st.sidebar.selectbox("Select Page",activities)
     
    dataset = st.sidebar.file_uploader("Upload Dataset",type=["csv","txt","xlsx"])
    
    if choice == 'About':
            st.subheader("**Intro**")
            st.markdown(intro, unsafe_allow_html=True)
            st.write("") 
            st.subheader("**How to use MISO?**")
            st.markdown(body, unsafe_allow_html=True)
            
    if st.sidebar.text_input("Developer Password", type="password") == "BetaPower123":
        key = "OK"            

    if dataset != None:
        df1 = read_file(dataset)

    if st.sidebar.checkbox("Change Data Frame or Subset") and (choice == '1. View Data' or choice == '2. Build Model' or choice == '3. Predict') and dataset != None:
        df = read_file(dataset) 
        colList = df.columns.tolist()
        selCols = st.multiselect("Select Columns", colList)
        df1 = df[selCols]
    
    if dataset != None and df1.empty == False:
        df = preprocess_file(df1)
    
    if 1+1 == 2:
        
        if choice == '1. View Data':
      
            if dataset is not None and df.shape[1]>1 and df.shape[0]>15:
                st.subheader("Review Data")
                if st.checkbox("View Data Table")and df.shape[1]>1:
                    st.dataframe(df)                    
                if st.checkbox("View Data Type(s)") and df.shape[1]>1:
                    df_types = df.dtypes.to_frame(name='Data Type')
                    df_types
                if df.shape[1]<6:
                    if st.checkbox("Scatter Plot Matrix (Max 5 Columns)"):
                        sns.pairplot(df,kind="reg",markers="+",corner=True)
                        st.pyplot() 
                if st.checkbox("Important Features") and df.shape[1]>1:
                    X = df.iloc[:,0:-1]
                    Y = df.iloc[:,-1]
                    if Y.dtype == "float64":
                        rfr = RandomForestRegressor(random_state=seed)
                        rfr.fit(X,Y)
                        importances = rfr.feature_importances_
                        tbl = pd.Series(importances, index = X.columns)
                        feat_importances = tbl.sort_values(ascending=True)
                        #fig = pyplot.figure()
                        #fig.suptitle('Rank Features by Importance')
                        #feat_plot = feat_importances.plot(kind='barh',figsize=(20,10))
                        #st.pyplot()
                        
                        result = permutation_importance(rfr, X, Y, n_repeats = 5, random_state = seed, n_jobs=-1)
                        result_mean = result.importances_mean
                        sorted_idx = result.importances_mean.argsort()
                        
                        r = result_mean.T.reshape(1,X.shape[1])
                        
                        feat_importances1 = pd.DataFrame(r, columns = X.columns)
                        feat_importances1 = feat_importances1.T
                        feat_importances1.rename(columns={ feat_importances1.columns[0]: "Rank" }, inplace = True)
                        feat_importances1 = feat_importances1.sort_values(by="Rank", ascending=False)
                        feat_importances1 = feat_importances1.reset_index()
                        feat_importances1.rename(columns={ feat_importances1.columns[0]: "Features" }, inplace = True)
                        feat_importances1 = feat_importances1.head()

                        pyplot.figure(figsize=(12, 8))
                        sns.barplot(data=feat_importances1,x = feat_importances1["Rank"], y =feat_importances1["Features"],  orient="h").set_title("Rank Importance")
                        st.pyplot()                                                        
                                                                        
            else:
                st.write("Please load data first.")
        
        if choice == '2. Build Model':
            
            if key == "OK":
                mode = st.sidebar.radio("Hyperparameter Tuning Mode",('Normal', 'Abnormal'))
            if dataset is not None and df.shape[1]>1:
                st.subheader("Let's Build Your Model")
                X_train, X_validation, Y_train, Y_validation = train_test(df)

                #Baseline Models Evaluation
                models = []                
                models.append(('KNN', Pipeline([('Scaler', StandardScaler()), ('Un_KNN', KNeighborsRegressor())])))
                models.append(('RFR', Pipeline([('Scaler', StandardScaler()), ('Un_RFR', RandomForestRegressor(random_state=seed))])))
                models.append(('GBR', Pipeline([('Scaler', StandardScaler()), ('Un_GBR', GradientBoostingRegressor(random_state=seed))])))
                models.append(('ABR', Pipeline([('Scaler', StandardScaler()), ('Un_ABR', AdaBoostRegressor(random_state=seed))])))
                            
                results = []
                names = []
                Model_Frame = []
                
                for name, model in models:
                    kfold = KFold(n_splits = num_folds, random_state = seed)
                    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring = scoring)
                    results.append(abs(cv_results))
                    names.append(name)
                    Model_Frame.append({'Model': name, 'MSE(mean)':cv_results.mean(),'MSE(std)':cv_results.std() })              
                
                if st.checkbox("Show Models"):
                    Model_Frame = pd.DataFrame(Model_Frame)
                    Model_Frame['MSE(mean)'] = Model_Frame['MSE(mean)'].abs()
                    Model_Frame = Model_Frame.sort_values(by='MSE(mean)', ascending=True)
                    Model_Frame = Model_Frame.reset_index()
                    Model_Frame.drop(columns="index", inplace = True, axis = 1)
                    st.table(Model_Frame.style.format({"MSE(mean)":"{:.2f}", "MSE(std)":"{:.2f}"}).highlight_min(axis=0))
                
                    if st.checkbox("Tune Model"):
                        Best_Model = st.sidebar.radio("Change Model",(Model_Frame.iloc[0][0], Model_Frame.iloc[1][0],Model_Frame.iloc[2][0],Model_Frame.iloc[3][0],))
                        if Best_Model == "KNN":
                            st.write("KNeighborsRegressor(KNN)")
                            scaler = StandardScaler().fit(X_train)
                            rescaledX_train = scaler.transform(X_train)
                            param_grid = get_param(Best_Model, mode)
                            best_model_KNN = KNeighborsRegressor()
                            kfold_KNN = KFold(n_splits = num_folds, random_state = seed)
                            grid = GridSearchCV(estimator = best_model_KNN, param_grid = param_grid, scoring = scoring, cv=kfold_KNN)
                            grid_result = grid.fit(rescaledX_train,Y_train)
                            MSE_train = abs(round(grid_result.best_score_,2))
                            st.write("Post hyperparams tuning for KNN on training data has MSE(mean) of {}.".format(MSE_train))
                            KNN_n_neighbors = list(grid_result.best_params_.values())[0]
                            KNN_p = list(grid_result.best_params_.values())[1]
                            
                            #Test Optimized Model
                            MSE = abs(round(grid_result.best_score_,2))
                            rescaledX_validation = scaler.transform(X_validation)
                            Y_prediction_ori = grid.predict(rescaledX_validation)
                            Corr = abs(round(r2_score(Y_validation,  Y_prediction_ori),2))
                            st.write("Post hyperparams tuning for KNN on test data has MSE(mean) of {}.".format(MSE))
                            st.write("Post hyperparams tuning for KNN on test data has r^2 of {}.".format(Corr))
                            
                            sns.set_style("whitegrid", {'axes.grid' : True})
                            sns.regplot(Y_validation, Y_prediction_ori).set_title("Predicted Vs Actual")
                            st.pyplot()
                            
                            filename = 'finalized_model.sav'
                            pickle.dump(grid, open(filename, 'wb'))
                            filename1 = 'scaler.pkl'
                            pickle.dump(scaler, open(filename1, 'wb')) 
                                                        
                        if Best_Model == "RFR":
                            st.write("RandomForestRegressor(RFR)")
                            scaler = StandardScaler().fit(X_train)
                            rescaledX_train = scaler.transform(X_train)                            
                            param_grid = get_param(Best_Model, mode)
                            best_model_RFR = RandomForestRegressor(random_state=seed)
                            kfold_RFR = KFold(n_splits = num_folds, random_state = seed)
                            grid = GridSearchCV(estimator = best_model_RFR, param_grid = param_grid, scoring = scoring, cv=kfold_RFR)
                            grid_result = grid.fit(rescaledX_train,Y_train)
                            MSE_train = abs(round(grid_result.best_score_,2))
                            st.write("Post hyperparams tuning for RFR on training data has MSE(mean) of {}.".format(MSE_train))
                            RFR_n_estimators = list(grid_result.best_params_.values())[0]
                            RFR_min_samples_split = list(grid_result.best_params_.values())[1]
                            
                            #Test Optimized Model
                            MSE = abs(round(grid_result.best_score_,2))
                            rescaledX_validation = scaler.transform(X_validation)
                            Y_prediction_ori = grid.predict(rescaledX_validation)
                            Corr = abs(round(r2_score(Y_validation,  Y_prediction_ori),2))
                            st.write("Post hyperparams tuning for RFR on test data has MSE(mean) of {}.".format(MSE))
                            st.write("Post hyperparams tuning for RFR on test data has r^2 of {}.".format(Corr))
                            
                            sns.set_style("whitegrid", {'axes.grid' : True})
                            sns.regplot(Y_validation, Y_prediction_ori).set_title("Predicted Vs Actual")
                            st.pyplot()
                                
                            filename = 'finalized_model.sav'
                            pickle.dump(grid, open(filename, 'wb'))
                            filename1 = 'scaler.pkl'
                            pickle.dump(scaler, open(filename1, 'wb'))      
                                                   
                        if Best_Model == "GBR":
                            st.write("GradientBoostingRegressor(GBR)")
                            scaler = StandardScaler().fit(X_train)
                            rescaledX_train = scaler.transform(X_train)
                            param_grid = get_param(Best_Model, mode) 
                            best_model_GBR = GradientBoostingRegressor(random_state=seed)
                            kfold_GBR = KFold(n_splits = num_folds, random_state = seed)
                            grid = GridSearchCV(estimator = best_model_GBR, param_grid = param_grid, scoring = scoring, cv=kfold_GBR)
                            grid_result = grid.fit(rescaledX_train,Y_train)
                            MSE_train = abs(round(grid_result.best_score_,2))
                            st.write("Post hyperparams tuning for GBR on training data has MSE(mean) of {}.".format(MSE_train))
                            GBR_n_estimators = list(grid_result.best_params_.values())[0]
                            GBR_min_samples_split = list(grid_result.best_params_.values())[1]
                            GBR_learning_rate = list(grid_result.best_params_.values())[2]
                                                                                  
                            #Test Optimized Model
                            MSE = abs(round(grid_result.best_score_,2))
                            rescaledX_validation = scaler.transform(X_validation)
                            Y_prediction_ori = grid.predict(rescaledX_validation)
                            Corr = abs(round(r2_score(Y_validation,  Y_prediction_ori),2))
                            st.write("Post hyperparams tuning for GBR on test data has MSE(mean) of {}.".format(MSE))
                            st.write("Post hyperparams tuning for GBR on test data has r^2 of {}.".format(Corr))
                            
                            sns.set_style("whitegrid", {'axes.grid' : True})
                            sns.regplot(Y_validation, Y_prediction_ori).set_title("Predicted Vs Actual")
                            st.pyplot()
                            
                            filename = 'finalized_model.sav'
                            pickle.dump(grid, open(filename, 'wb'))
                            filename1 = 'scaler.pkl'
                            pickle.dump(scaler, open(filename1, 'wb'))                           
        
                        if Best_Model == "ABR":
                            st.write("AdaBoostRegressor(ABR)")
                            scaler = StandardScaler().fit(X_train)
                            rescaledX_train = scaler.transform(X_train)
                            param_grid = get_param(Best_Model, mode)
                            best_model_ABR = AdaBoostRegressor(random_state=seed)
                            kfold_ABR = KFold(n_splits = num_folds, random_state = seed)
                            grid = GridSearchCV(estimator = best_model_ABR, param_grid = param_grid, scoring = scoring, cv=kfold_ABR)
                            grid_result = grid.fit(rescaledX_train,Y_train)
                            MSE_train = abs(round(grid_result.best_score_,2))
                            st.write("Post hyperparams tuning for ABR on training data has MSE(mean) of {}.".format(MSE_train))
                            ABR_n_estimators = list(grid_result.best_params_.values())[0]
                            ABR_learning_rate = list(grid_result.best_params_.values())[1]
                            
                            #Test Optimized Model
                            MSE = abs(round(grid_result.best_score_,2))
                            rescaledX_validation = scaler.transform(X_validation)
                            Y_prediction_ori = grid.predict(rescaledX_validation)
                            Corr = abs(round(r2_score(Y_validation,  Y_prediction_ori),2))
                            st.write("Post hyperparams tuning for ABR on test data has MSE(mean) of {}.".format(MSE))
                            st.write("Post hyperparams tuning for ABR on test data has r^2 of {}.".format(Corr))
                            
                            sns.set_style("whitegrid", {'axes.grid' : True})
                            sns.regplot(Y_validation, Y_prediction_ori).set_title("Predicted Vs Actual")
                            st.pyplot()
                            
                            filename = 'finalized_model.sav'
                            pickle.dump(grid, open(filename, 'wb'))
                            filename1 = 'scaler.pkl'
                            pickle.dump(scaler, open(filename1, 'wb')) 
                                      
                        
            else:
                st.write("Please load data first.")
             
        if choice == '3. Predict':
            if dataset is not None:
                if df is not None and df.shape[1]>1 and df.shape[0]>15:
                    filename = r'finalized_model.sav'
                    grid = pickle.load(open(filename, 'rb'))
                    
                    filename1 = 'scaler.pkl'
                    scaler = pickle.load(open(filename1, 'rb'))
                    
                    st.subheader("Let's Predict")

                    #Dynamically display input boxes and compile input table
                    oridf = df
                    df = df.iloc[:,:-1]
                    df = df.astype(float)
                    
                    cols = df.columns
                    var = []
                    for col in cols:
                        var.append(st.number_input(col,min_value =df[col].min(),format="%.4f" , max_value =df[col].max() , key = df.columns.get_loc(col)))
                        
                    var_table = pd.DataFrame(var)
                    col_table = pd.DataFrame(df.columns.tolist())
                    
                    InputTable = pd.concat([col_table, var_table], axis=1, sort=False)
                    InputTable = InputTable.T
                    InputTableHeader = InputTable.iloc[0]
                    InputTable = InputTable[1:]
                    InputTable.columns = InputTableHeader
                    
                    rescaledInputTable = scaler.transform(InputTable)

                    Y_prediction = ""
                    if st.button("Predict"):
                        Y_prediction = grid.predict(rescaledInputTable)
                        st.write("Predicted value for **{}** is {}.".format(oridf.columns[-1],round(Y_prediction[0]),4))
                

                else:
                    st.write("Please load data and/or tune best model first.")
            else:
                st.write("Please load data and/or tune best model first.")                
              
                    
if __name__ == '__main__':
    root()