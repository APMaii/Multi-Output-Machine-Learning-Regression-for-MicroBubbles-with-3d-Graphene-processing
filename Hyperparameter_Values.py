# -*- coding: utf-8 -*-
"""
PARALELL_NOTCROSS_MICROBUBBLE

This is hyperparameters 
"""

#=============================================================================
'                              Import                                        '
#=============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from  sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
#from mlxtend.regressor import StackingCVRegressor
#============================================================================
'                              Loading data                                 '
#============================================================================
#data haro kamel check konim
def load_data():
    #====================================DATA1===================================
    global x_data1,y_data1
    x_data1=np.array((0,4,8,16,24,32,40)).reshape(7,1)
    y_data1=np.array([(83.2,10.2,0.122596),(65.8,11.3,0.171733),(49.7,11.9,0.239437),
                    (39.0,12.6,0.323077),(43.2,12.8,0.296296),(47.5,12.5,0.263158),
                    (62.6,11.7,0.186901)])
    #====================================DATA2===================================
    file2_location = 'C:\\Users\\Mnaderi8294\\.spyder-py3\\finalcsv.csv'
    #file2_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\finalcsv.csv'
    f2=open(file2_location,'r')
    global x_data2,data2,data2_0,data2_4,data2_8,data2_16,data2_24,data2_output,data2_outputs,data2_input
    data2=pd.read_csv(f2)
    data2.drop(1010,axis=0,inplace=True)
    x_data2=np.array((0,4,8,16,24)).reshape(5,1)
    data2_0 = data2['ob'] - data2['0f']
    data2_4=data2['4b']-data2['4f']
    data2_8=data2['8b']-data2['8f']
    data2_16=data2['16b']-data2['16f']
    data2_24=data2['24b']-data2['24f']
    data2_output=pd.concat([data2_0,data2_4,data2_8,data2_16,data2_24],axis=1)
    data2_outputs=data2_output.T
    #data2_outputs[::-1]
    data2_input=pd.DataFrame(data=[0,4,8,16,24] ,columns=['%HAGP'])   
    #====================================DATA3===================================
    file3_location = 'C:\\Users\\Mnaderi8294\\.spyder-py3\\section2.csv'
    #file3_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\section2.csv'
    f3=open(file3_location,'r')
    global x_data3,data3,data3_0,data3_4,data3_8,data3_16,data3_24,data3_32,data3_40,y_data3_drain,y_data3_foam
    global half_time
    data3=pd.read_csv(f3)
    x_data3=np.array((0,4,8,16,24,32,40)).reshape(7,1)
    #y_data3=pd.concat([data3_0,data3_4,data3_8,data3_16,data3_24,data3_32,data3_40],axis=1).T
    y_data3_foam=pd.concat([data3['Unnamed: 1'],data3['Unnamed: 3'],data3['Unnamed: 5'],
                       data3['Unnamed: 7'],data3['Unnamed: 9'],data3['Unnamed: 11'],
                       data3['Unnamed: 13']],axis=1).T.reset_index(drop=True)
    y_data3_drain=pd.concat([data3['Unnamed: 0'],data3['Unnamed: 2'],data3['Unnamed: 4'],
                       data3['Unnamed: 6'],data3['Unnamed: 8'],data3['Unnamed: 10'],
                       data3['Unnamed: 12']],axis=1).T.reset_index(drop=True)
    half_time=np.array((58,87.5,200,280,260,210,176)).reshape(7,1)
load_data()


#============================================================================
'                             Select_model                           '
#============================================================================
def refit_strategy(cv_results):
    cv_results_ = pd.DataFrame(cv_results)
    fastest_top_recall_high_precision_index = cv_results_[
            "std_test_score"
        ].idxmin()
    return fastest_top_recall_high_precision_index


def refit_strategy(cv_results):
    cv_results_ = pd.DataFrame(cv_results)
    lenn=len(cv_results_)
    if lenn<10:
        ii=3
    if 21>lenn>9:
        ii=4
    if lenn>20:
        ii=5
    if lenn>50:
        ii=7
    sort_score=cv_results_.sort_values(by='mean_test_score',ascending=False)
    #score=sort_score.reset_index(drop=True)
    #final_score=score[0:3].reset_index(drop=False)
    fastest_top_recall_high_precision_index = sort_score[0:ii][
            "std_test_score"
        ].idxmin()
    return fastest_top_recall_high_precision_index



def single_select_model(x,y,model,cvv,which,sc='neg_mean_absolute_percentage_error'):
    if model=='LR':
        poly=PolynomialFeatures()
        regressior=LinearRegression()        
        params=[{'poly__degree':[1,2]}]
        pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model== 'KNN':
        regressior = KNeighborsRegressor(n_jobs=60)
        params=[{'regressior__n_neighbors':[1,2,3,4,5]}]
        pipe=Pipeline(steps=[('regressior',regressior)]) 
    if model== 'DT':
        poly=PolynomialFeatures()
        regressior=DecisionTreeRegressor(random_state=0)
        params=[{'poly__degree':[1,2],
                'regressior__max_depth':[1,2,5,10],
         'regressior__min_samples_split':[2,4,8]} ]    
        pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model=='RF':
        poly=PolynomialFeatures()
        regressior=RandomForestRegressor(random_state=0,n_jobs=60)
        params=[{'poly__degree':[1,2],
                'regressior__n_estimators':[2,10,40],
                    'regressior__max_depth':[1,2,5,10]}]
        pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model=='SVR':
        poly=PolynomialFeatures()
        #scaler=MinMaxScaler()
        scaler=PowerTransformer()
        regressior = SVR()
        params=[{'poly':[None],
                'regressior__kernel': ['rbf'],
                    'regressior__gamma':np.logspace(-2, 2, 5),
                    'regressior__C':[0.1,1e0, 1e1,1e2, 1e3]},
                    {'poly':[None],
                     'regressior__kernel':['linear'],
                    'regressior__C':[0.1,1e0, 1e1,1e2, 1e3]},
                    {'poly':[None],
                     'regressior__kernel':['poly'],
                     'regressior__C':[0.1,1e0, 1e1,1e2, 1e3],
                     'regressior__degree':[1,2]},
                    {'poly__degree':[1,2],
                'regressior__kernel': ['rbf'],
                    'regressior__gamma':np.logspace(-2, 2, 5),
                    'regressior__C':[0.1,1e0, 1e1,1e2, 1e3]},
                    {'poly__degree':[1,2],
                     'regressior__kernel':['linear'],
                    'regressior__C':[0.1,1e0, 1e1,1e2, 1e3]},
                    {'poly__degree':[1,2],
                     'regressior__kernel':['poly'],
                     'regressior__C':[0.1,1e0, 1e1,1e2, 1e3],
                     'regressior__degree':[1,2]}]            
        pipe=Pipeline(steps=[('poly',poly),('scaler',scaler),('regressior',regressior)])
    if model=='MLP':  
        poly=PolynomialFeatures()
        regressior=MLPRegressor(solver='adam',random_state=40,max_iter=500)
        #scaler=MinMaxScaler()
        scaler=PowerTransformer()
        params=[{'poly__degree':[1,2],
                'regressior__hidden_layer_sizes':[(5,),(10,),(2,),(4,),
(100,),(200,)],
                 'regressior__activation':['identity', 'tanh', 'relu'],
                 'regressior__alpha':[0.0001,0.001,0.1,1],
                'regressior__alpha':[0.000001,0.0001,0.001,0.01,0.01,1]}]
        pipe=Pipeline(steps=[('poly',poly),('scaler',scaler),('regressior',regressior)])
        
    kfold1=KFold(n_splits=cvv,shuffle=False)
    kfold2=KFold(n_splits=cvv+1,shuffle=False)  
    #scoring_list=sc
    scoring_list='neg_mean_absolute_percentage_error'
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list
                                    #,refit=refit_strategy
                                    ,n_jobs=60)
    grid.fit(x,y)
    if which=='gridsearch':
        
        return grid
    if which=='score':
        cross_score=cross_val_score(grid.best_estimator_, x,y,cv=kfold1,scoring=sc,n_jobs=60)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid.best_estimator_, x,y,cv=kfold1,n_jobs=60)
        return cross_pred
    

#without scale ***no edit until now
def select_model(x,y,model,cvv,which,sc='neg_mean_absolute_percentage_error'):
    if model=='LR':
        poly=PolynomialFeatures()
        base_model=LinearRegression()        
        regressior = MultiOutputRegressor(base_model,n_jobs=60) 
        params=[{'poly__degree':[1,2,3,4]}]
        pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model== 'KNN':
        base_model = KNeighborsRegressor()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressior__estimator__n_neighbors':[1,2,3,4,5]}]
        pipe=Pipeline(steps=[('regressior',regressior)]) 
    if model== 'DT':
        base_model=DecisionTreeRegressor(random_state=0)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressior__max_depth':[1,2,5,10],
         'regressior__min_samples_split':[2,4,8]} ]     
        pipe=Pipeline(steps=[('regressior',regressior)]) 
    if model=='RF':
        base_model=RandomForestRegressor(random_state=0,n_jobs=60)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressior__n_estimators':[2,10,40],
                    'regressior__max_depth':[1,2,5,10]}]
        pipe=Pipeline(steps=[('regressior',regressior)])
    if model=='SVR':
        #poly=PolynomialFeatures()
        scaler=MinMaxScaler()
        base_model = SVR()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)        
        params=[{'regressior__estimator__kernel': ['rbf'],
                    'regressior__estimator__gamma':np.logspace(-2, 2, 5),
                    'regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3]},
                    {'regressior__estimator__kernel':['linear'],
                    'regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3]},
                    {'regressior__estimator__kernel':['poly'],
                     'regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3],
                     'regressior__estimator__degree':[1,2,3]}]

        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
    if model=='MLP':
        base_model=MLPRegressor(solver='lbfgs',random_state=0)
        scaler=MinMaxScaler()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressior__estimator__hidden_layer_sizes':[(5,),(10,),
(100,),(200,)],
                 'regressior__estimator__activation':['identity', 'tanh', 'relu'],
                 'regressior__estimator__alpha':[0.0001,0.001,0.1,1]}]
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        #('scaler',scaler),
        
    kfold1=KFold(n_splits=cvv,shuffle=False)
    kfold2=KFold(n_splits=cvv+1,shuffle=False)
    #scoring_list=['neg_mean_absolute_percentage_error',
      #            'neg_mean_absolute_error',
       #           'neg_mean_squared_error',
        #          'neg_root_mean_squared_error']
    scoring_list=sc
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list
                                    #,refit=refit_strategy
                                    ,
                                    
                                    n_jobs=60)
    grid.fit(x,y)
    
    if which=='gridsearch':
        
        return grid
    if which=='score':
        cross_score=cross_val_score(grid.best_estimator_, x,y,cv=kfold1,scoring=sc,n_jobs=60)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid.best_estimator_, x,y,cv=kfold1,n_jobs=60)
        return cross_pred

def select_model2(x,y,model,cvv,which,sc='neg_mean_absolute_percentage_error'):
    if model=='LR':
        poly=PolynomialFeatures()
        base_model=LinearRegression()        
        regressior = MultiOutputRegressor(base_model,n_jobs=60) 
        params=[{'poly__degree':[1,2,3,4]}]
        pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model== 'KNN':
        base_model = KNeighborsRegressor()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressior__estimator__n_neighbors':[1,2,3,4,5]}]
        pipe=Pipeline(steps=[('regressior',regressior)]) 
    if model== 'DT':
        base_model=DecisionTreeRegressor(random_state=0)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressior__max_depth':[1,2,5,10],
         'regressior__min_samples_split':[2,4,8]} ]     
        pipe=Pipeline(steps=[('regressior',regressior)]) 
    if model=='RF':
        base_model=RandomForestRegressor(random_state=0,n_jobs=60)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressior__n_estimators':[2,10,40],
                    'regressior__max_depth':[1,2,5,10]}]
        pipe=Pipeline(steps=[('regressior',regressior)])
    if model=='SVR':
        #poly=PolynomialFeatures()
        scaler=MinMaxScaler()
        base_model = SVR()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)        
        params=[{'regressior__estimator__kernel': ['rbf'],
                    'regressior__estimator__gamma':np.logspace(-2, 2, 5),
                    'regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3]},
                    {'regressior__estimator__kernel':['linear'],
                    'regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3]},
                    {'regressior__estimator__kernel':['poly'],
                     'regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3],
                     'regressior__estimator__degree':[1,2,3]}]

        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
    if model=='MLP':
        base_model=MLPRegressor(solver='lbfgs',random_state=0)
        scaler=MinMaxScaler()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressior__estimator__hidden_layer_sizes':[(5,),(10,),
(100,),(200,)],
                 'regressior__estimator__activation':['identity', 'tanh', 'relu'],
                 'regressior__estimator__alpha':[0.0001,0.001,0.1,1]}]
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        #('scaler',scaler),
        
    kfold1=KFold(n_splits=cvv,shuffle=False)
    kfold2=KFold(n_splits=cvv+1,shuffle=False)
    #scoring_list=['neg_mean_absolute_percentage_error',
      #            'neg_mean_absolute_error',
       #           'neg_mean_squared_error',
        #          'neg_root_mean_squared_error']
    scoring_list=sc
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list
                                    #,refit=refit_strategy
                                    ,
                                    
                                    n_jobs=60)
    grid.fit(x,y)
    
    if which=='gridsearch':
        
        return grid
    if which=='score':
        cross_score=cross_val_score(grid.best_estimator_, x,y,cv=kfold1,scoring=sc,n_jobs=60)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid.best_estimator_, x,y,cv=kfold1,n_jobs=60)
        return cross_pred

#================================================================
#========================================================================
#===========================================================================
#========================================================================
#================================================================
    
#with scale
def select_model(x,y,model,cvv,which,sc='neg_mean_absolute_percentage_error'):
    if model=='LR':
        poly=PolynomialFeatures()
        base_model=LinearRegression()        
        regressior = MultiOutputRegressor(base_model,n_jobs=60) 
        params=[{'regressor__poly':[None],
                'transformer':[None,MinMaxScaler(),StandardScaler()]},
                {'regressor__poly__degree':[1,2],
                'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model== 'KNN':
        base_model = KNeighborsRegressor()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__n_neighbors':[1,2,3,4,5],
                'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model== 'DT':
        poly=PolynomialFeatures()
        base_model=DecisionTreeRegressor(random_state=0)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__poly':[None],
                'regressor__regressior__estimator__max_depth':[1,2,5,10],
         'regressor__regressior__estimator__min_samples_split':[2,4,8],
                 'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
        {'regressor__poly__degree':[1,2],
                'regressor__regressior__estimator__max_depth':[1,2,5,10],
         'regressor__regressior__estimator__min_samples_split':[2,4,8],
                 'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]     
        mini_pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='RF':
        poly=PolynomialFeatures()
        base_model=RandomForestRegressor(random_state=0,n_jobs=60)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__poly':[None],
                'regressor__regressior__estimator__n_estimators':[2,10,40],
                    'regressor__regressior__estimator__max_depth':[1,2,5,10],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                {'regressor__poly__degree':[1,2],
                'regressor__regressior__estimator__n_estimators':[2,10,40],
                    'regressor__regressior__estimator__max_depth':[1,2,5,10],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='SVR':
        poly=PolynomialFeatures()
        #scaler=MinMaxScaler()
        scaler=PowerTransformer()
        base_model = SVR()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)        
        params=[ {'regressor__poly':[None],
          'regressor__regressior__estimator__kernel': ['rbf'],
             'regressor__regressior__estimator__gamma':np.logspace(-2, 2, 5),
             'regressor__regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3],
             'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
             {'regressor__poly':[None],
              'regressor__regressior__estimator__kernel':['linear'],
             'regressor__regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3],
             'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
             {'regressor__poly':[None],
              'regressor__regressior__estimator__kernel':['poly'],
              'regressor__regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3],
              'regressor__regressior__estimator__degree':[1,2],
                      'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                {'regressor__poly__degree':[1,2],
                 'regressor__regressior__estimator__kernel': ['rbf'],
                    'regressor__regressior__estimator__gamma':np.logspace(-2, 2, 5),
                    'regressor__regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3],
                    'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                    {'regressor__poly__degree':[1,2],
                     'regressor__regressior__estimator__kernel':['linear'],
                    'regressor__regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3],
                    'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                    {'regressor__poly__degree':[1,2],
                     'regressor__regressior__estimator__kernel':['poly'],
                     'regressor__regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3],
                     'regressor__regressior__estimator__degree':[1,2],
                             'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]

        mini_pipe=Pipeline(steps=[('poly',poly),('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='MLP':
        poly=PolynomialFeatures()
        base_model=MLPRegressor(solver='adam',random_state=40,max_iter=500)
        #scaler=MinMaxScaler()
        scaler=PowerTransformer()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__poly':[None],
                'regressor__regressior__estimator__hidden_layer_sizes':[(5,),(10,),
                                                              (100,),
                                                              (200,)],
                 'regressor__regressior__estimator__activation':['identity', 'tanh', 'relu'],
                 'regressor__regressior__estimator__alpha':[0.0001,0.001,0.1,1],
                         'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                {'regressor__poly__degree':[1,2],
                'regressor__regressior__estimator__hidden_layer_sizes':[(5,),(10,),
                                                              (100,),
                                                              (200,)],
                 'regressor__regressior__estimator__activation':['identity', 'tanh', 'relu'],
                 'regressor__regressior__estimator__alpha':[0.0001,0.001,0.1,1],
                         'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('poly',poly),('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

        #('scaler',scaler),
        
    kfold1=KFold(n_splits=cvv,shuffle=False)
    kfold2=KFold(n_splits=cvv+1,shuffle=False)
    #scoring_list=['neg_mean_absolute_percentage_error',
      #            'neg_mean_absolute_error',
       #           'neg_mean_squared_error',
        #          'neg_root_mean_squared_error']
    #scoring_list=sc
    scoring_list='neg_mean_absolute_percentage_error'
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list
                                    
                                    #,refit=refit_strategy
                                    ,
                                    
                                    n_jobs=60)
    grid.fit(x,y)
    if which=='gridsearch':  
        
        return grid
    if which=='score':
        cross_score=cross_val_score(grid.best_estimator_, x,y,cv=kfold1,scoring=sc,n_jobs=60)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid.best_estimator_, x,y,cv=kfold1,n_jobs=60)
        return cross_pred


def select_model2(x,y,model,cvv,which,sc='neg_mean_absolute_percentage_error'):
    if model=='LR':
        base_model=LinearRegression()        
        regressior = MultiOutputRegressor(base_model,n_jobs=60) 
        params=[{'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
    if model== 'KNN':
        base_model = KNeighborsRegressor()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__n_neighbors':[1,2,3,4],
                'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model== 'DT':
        base_model=DecisionTreeRegressor(random_state=0)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{
                'regressor__regressior__estimator__max_depth':[1,2,5,10],
         'regressor__regressior__estimator__min_samples_split':[2,4,8],
                 'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]     
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='RF':
        base_model=RandomForestRegressor(random_state=0,n_jobs=60)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__n_estimators':[2,10],
                    'regressor__regressior__estimator__max_depth':[1,2,5,10],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='SVR':
       # scaler=MinMaxScaler()
        #scaler=StandardScaler()
        scaler=PowerTransformer()
        base_model = SVR()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)        
        params=[{'regressor__scaler':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()],
                'regressor__regressior__estimator__kernel': ['rbf'],
                    'regressor__regressior__estimator__gamma':np.logspace(-2, 2, 5),
                    'regressor__regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                    {'regressor__scaler':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()],
                     'regressor__regressior__estimator__kernel':['linear'],
                    'regressor__regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                    
                    {'regressor__scaler':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()],
                     'regressor__regressior__estimator__kernel':['poly'],
                     'regressor__regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3],
                     'regressor__regressior__estimator__degree':[2,3,4],
                             'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]

        mini_pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='MLP':
        base_model=MLPRegressor(solver='adam',random_state=40,max_iter=200,activation='relu',
                                hidden_layer_sizes=(200,),
                                alpha=0.01)
        #scaler=MinMaxScaler()
        #scaler=StandardScaler()
        scaler=PowerTransformer()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)        
        params=[{'regressor__scaler':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()],
                         'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        
        
        mini_pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

        #('scaler',scaler),
    kfold1=KFold(n_splits=cvv,shuffle=False)
    kfold2=KFold(n_splits=cvv+1,shuffle=False)
    #scoring_list=['neg_mean_absolute_percentage_error',
      #            'neg_mean_absolute_error',
       #           'neg_mean_squared_error',
        #          'neg_root_mean_squared_error']
    #scoring_list=sc
    scoring_list='neg_mean_absolute_error'
    #scoring_list='neg_mean_absolute_percentage_error'
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list
                                    
                                    #,refit=refit_strategy
                                    , n_jobs=60)
    grid.fit(x,y)
    if which=='gridsearch':  
        
        return grid
    if which=='score':
        cross_score=cross_val_score(grid.best_estimator_, x,y,cv=kfold1,scoring=sc,n_jobs=60)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid.best_estimator_, x,y,cv=kfold1,n_jobs=60)
        return cross_pred
    

   
def select_model2(x,y,model,cvv,which,sc='neg_mean_absolute_percentage_error'):
    if model=='LR':
        poly=PolynomialFeatures()
        base_model=LinearRegression()        
        regressior = MultiOutputRegressor(base_model,n_jobs=60) 
        params=[{'regressor__poly__degree':[1,2]}]
        mini_pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model== 'KNN':
        base_model = KNeighborsRegressor()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__n_neighbors':[1,2,3,4]},
                {'transformer':[None,MinMaxScaler(),StandardScaler()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model== 'DT':
        base_model=DecisionTreeRegressor(random_state=0)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__max_depth':[1,2,5,10],
         'regressor__regressior__estimator__min_samples_split':[2,4,8]},
                 {'transformer':[None,MinMaxScaler(),StandardScaler()]}]     
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='RF':
        base_model=RandomForestRegressor(random_state=0,n_jobs=60)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__n_estimators':[2,10,40],
                    'regressor__regressior__estimator__max_depth':[1,2,5,10]},
                            {'transformer':[None,MinMaxScaler(),StandardScaler()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='SVR':
        #poly=PolynomialFeatures()
        #scaler=MinMaxScaler()
        scaler=StandardScaler()
        base_model = SVR()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)        
        params=[{'regressor__regressior__estimator__kernel': ['rbf'],
                    'regressor__regressior__estimator__gamma':np.logspace(-2, 2, 5),
                    'regressor__regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3]},
                    {'regressor__regressior__estimator__kernel':['linear'],
                    'regressor__regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3]},
                    {'regressor__regressior__estimator__kernel':['poly'],
                     'regressor__regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3],
                     'regressor__regressior__estimator__degree':[1,2]},
                             {'transformer':[None,MinMaxScaler(),StandardScaler()]}]

        mini_pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='MLP':
        base_model=MLPRegressor(solver='adam',random_state=40,max_iter=500)
        #scaler=MinMaxScaler()
        scaler=StandardScaler()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__hidden_layer_sizes':[(5,),(10,),
                                                              (100,),
                                                              (200,)],
                 'regressor__regressior__estimator__activation':['identity', 'tanh', 'relu'],
                 'regressor__regressior__estimator__alpha':[0.0001,0.001,0.1,1]},
                         {'transformer':[None,MinMaxScaler(),StandardScaler()]}]
        mini_pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

        #('scaler',scaler),
        
    kfold1=KFold(n_splits=cvv,shuffle=False)
    kfold2=KFold(n_splits=cvv+1,shuffle=False)
    #scoring_list=['neg_mean_absolute_percentage_error',
      #            'neg_mean_absolute_error',
       #           'neg_mean_squared_error',
        #          'neg_root_mean_squared_error']
    #scoring_list=sc
    scoring_list='neg_mean_absolute_percentage_error'
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list
                                    
                                    #,refit=refit_strategy
                                    ,
                                    
                                    n_jobs=60)
    grid.fit(x,y)
    if which=='gridsearch':  
        
        return grid
    if which=='score':
        cross_score=cross_val_score(grid.best_estimator_, x,y,cv=kfold1,scoring=sc,n_jobs=60)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid.best_estimator_, x,y,cv=kfold1,n_jobs=60)
        return cross_pred

#============================================================================
'                             Section1 : T/D                                '
#============================================================================
def single_section1(model,which,number=50,sc='neg_mean_absolute_percentage_error'):
    x=x_data1
    #y=y_data1[:,2].reshape(-1,1)
    y=y_data1[:,2].reshape(-1,)
    #grid=single_select_model(x,y,model,7,'gridsearch',sc=sc)
    if which=='p':
        grid=single_select_model(x,y,model,7,'gridsearch',sc=sc)
        y_pred=grid.predict(np.array(range(0,number)).reshape(-1,1))
        return y_pred
    if which=='s':
        pred=single_select_model(x,y,model,7,'prediction')
        score1=mean_absolute_percentage_error(y_data1[:,2],pred)
        
        score2=mean_absolute_error(y_data1[:,2],pred)
        
        score3=mean_squared_error(y_data1[:,2],pred)
        
        score44=mean_squared_error(y_data1[:,2],pred)
        score4=np.sqrt(score44)
        
        score=pd.DataFrame(data=((score1),(score2),(score3),(score4))) 
        #score=grid.best_score_
        return score


def multi_output_section1(model,which,number=50):
    x=x_data1
    y=y_data1
    if which=='p':
        grid=select_model(x,y,model,7,'gridsearch')
        y_pred=grid.predict(np.array(range(0,number)).reshape(-1,1))
        return y_pred
    #for only T/D
    if which=='s1':
        
        pred=select_model(x,y,model,7,'prediction')
        
        
        score1=mean_absolute_percentage_error(y_data1[:,2],pred[:,2])
        
        score2=mean_absolute_error(y_data1[:,2],pred[:,2])
        
        score3=mean_squared_error(y_data1[:,2],pred[:,2])
        
        score44=mean_squared_error(y_data1[:,2],pred[:,2])
        score4=np.sqrt(score44)
        score=pd.DataFrame(data=((score1),(score2),(score3),(score4)))                  
        
        return score
    #for T pred/ D pred
    if which=='s2':
        pred=select_model(x,y,model,7,'prediction')
        predd=pred[:,1]/pred[:,0]       
        
        
        score1=mean_absolute_percentage_error(y_data1[:,2],predd)
        
        score2=mean_absolute_error(y_data1[:,2],predd)
        
        score3=mean_squared_error(y_data1[:,2],predd)
        
        score44=mean_squared_error(y_data1[:,2],predd)
        score4=np.sqrt(score44)
        score=pd.DataFrame(data=((score1),(score2),(score3),(score4)))
        return score

def base_model(x,y,model,cv,which,number=100):
    grid=select_model(x,y,model,cv,'gridsearch') 
    if which=='p':  
        y_pred=grid.predict(x)
        return y_pred,grid
    if which=='s':
        xf=x_data1[number].reshape(-1,1)
        y_pred=grid.predict(xf)
        return y_pred
#metaa_model('LR','MLP','s')

def metaa_model(model,meta_model,which):
    if which=='p':
        pre_x=x_data1
        y=y_data1
        x1=x_data1.reshape(-1)
        x1=pd.Series(x1)
        y_pred=base_model(pre_x,y,model,7,'p')[0]
        x2=pd.DataFrame(y_pred)
        x=pd.concat([x1,x2],axis=1)
        x=np.array(x).reshape(-1,4)
        meta_grid=select_model(x,y,meta_model,7,'gridsearch')
        return meta_grid
    if which=='s':

        x1=x_data1.reshape(-1)
        x1=pd.Series(x1) 
        y_pred=base_model(x_data1.reshape(-1,1),y_data1,model,7,'p')[0]
        x2=pd.DataFrame(y_pred)  
        y=y_data1  
        x=pd.concat([x1,x2],axis=1)
        x=np.array(x).reshape(-1,4)
        
        meta_pred=select_model(x,y_data1,meta_model,7,'prediction')   
        
        score1=mean_absolute_percentage_error(y_data1[:,2],meta_pred[:,2])
        score2=mean_absolute_error(y_data1[:,2],meta_pred[:,2])
        score3=mean_squared_error(y_data1[:,2],meta_pred[:,2])
        score44=mean_squared_error(y_data1[:,2],meta_pred[:,2])
        score4=np.sqrt(score44)
       
            
        score=pd.DataFrame(data=(score1,score2,
                                     score3,score4))
        return score
    
def base_model_predict (n,model):
    pre_x=x_data1
    y=y_data1
    grid=base_model(pre_x,y,model,7,'p')[1]
    base_predict=grid.predict(np.array(range(0,n)).reshape(-1,1))
    return base_predict

j=90
def meta_model_predict(j,model,meta_model):
    n=j 
    meta_grid=metaa_model(model,meta_model,'p')
    x1_list=pd.Series(list(range(0,n)))
    xy_list=base_model_predict(n,model)
    x2_list=pd.DataFrame(xy_list)
    x_final=pd.concat([x1_list,x2_list],axis=1)
    x_final=np.array(x_final).reshape(-1,4)                
    meta_pred=meta_grid.predict(x_final)
    return meta_pred
model='SVR'
meta_model='SVR'
def score_section1(which,sc='neg_mean_absolute_percentage_error'):
    
    if sc=='neg_mean_absolute_percentage_error':
        titsc='MAPE'
    if sc=='neg_mean_absolute_error':
        titsc='MAE'
    if sc=='neg_mean_squared_error':
        titsc='MSE'
    if sc=='neg_root_mean_squared_error':
        titsc='RMSE'
    models=['LR','KNN','DT','RF','SVR','MLP']
    if which=='MTRS':
        score_list1=[]
        score_list2=[]
        score_list3=[]
        score_list4=[]
        #**jabeja krdm meta modelo ba model
        for i in range(0,6):
            meta_model=models[i]
            
            for i in range(0,6):
                model=models[i]
                scoree=metaa_model(model, meta_model, 's')
                #score=np.array(scoree).mean()
                print('this Loop passed===========================================================')
                score_list1.append(scoree[0][0])
                score_list2.append(scoree[0][1])
                score_list3.append(scoree[0][2])
                score_list4.append(scoree[0][3])
        
        final1=pd.DataFrame(data=((score_list1[0:6]),(score_list1[6:12]),(score_list1[12:18]),
                           (score_list1[18:24]),(score_list1[24:30]),(score_list1[30:36])),
                           index=models,
                           columns=models)
        name1='1_MTRS_MAPE.csv'
        final1.to_csv(name1)
        print('The file is saved')
        
        final2=pd.DataFrame(data=((score_list2[0:6]),(score_list2[6:12]),(score_list2[12:18]),
                           (score_list2[18:24]),(score_list2[24:30]),(score_list2[30:36])),
                           index=models,
                           columns=models)
        name2='1_MTRS_MAE.csv'
        final2.to_csv(name2)
        print('The file is saved')
        
        final3=pd.DataFrame(data=((score_list3[0:6]),(score_list3[6:12]),(score_list3[12:18]),
                           (score_list3[18:24]),(score_list3[24:30]),(score_list3[30:36])),
                           index=models,
                           columns=models)
        name3='1_MTRS_MSE.csv'
        final3.to_csv(name3)
        print('The file is saved')
        
        final4=pd.DataFrame(data=((score_list4[0:6]),(score_list4[6:12]),(score_list4[12:18]),
                           (score_list4[18:24]),(score_list4[24:30]),(score_list4[30:36])),
                           index=models,
                           columns=models)
        name4='1_MTRS_RMSE.csv'
        final4.to_csv(name4)
        print('The file is saved')
        
      #  return final
    
    if which=='single':
        score_list1=[]
        score_list2=[]
        score_list3=[]
        score_list4=[]
        for i in range(0,6):
            model=models[i]
            #score=single_section1(model,'s',sc=sc)
            score=single_section1(model,'s')
            score_list1.append(score[0][0])
            score_list2.append(score[0][1])
            score_list3.append(score[0][2])
            score_list4.append(score[0][3])
            #score_list.append(-1*(score.mean()))
        
        final1=pd.DataFrame(data=np.array(score_list1).reshape(1,-1),columns=models)
        name1='1_Single_MAPE.csv'
        final1.to_csv(name1)
        print('The file is saved')
        #return final
        final2=pd.DataFrame(data=np.array(score_list2).reshape(1,-1),columns=models)
        name2='1_Single_MAE.csv'
        final2.to_csv(name2)
        print('The file is saved')
        
        final3=pd.DataFrame(data=np.array(score_list3).reshape(1,-1),columns=models)
        name3='1_Single_MSE.csv'
        final3.to_csv(name3)
        print('The file is saved')
        
        final4=pd.DataFrame(data=np.array(score_list4).reshape(1,-1),columns=models)
        name4='1_Single_RMSE.csv'
        final4.to_csv(name4)
        print('The file is saved')
        
    if which=='multi':
        score_list11=[]
        score_list12=[]
        score_list13=[]
        score_list14=[]
        for i in range(0,6):
            model=models[i]
            scoree=multi_output_section1(model,'s1')
            score_list11.append(scoree[0][0])
            score_list12.append(scoree[0][1])
            score_list13.append(scoree[0][2])
            score_list14.append(scoree[0][3])
        score_list21=[]
        score_list22=[]
        score_list23=[]
        score_list24=[]
        for i in range(0,6):
            model=models[i]
            scoree=multi_output_section1(model,'s2')
  
            score_list21.append(scoree[0][0])
            score_list22.append(scoree[0][1])
            score_list23.append(scoree[0][2])
            score_list24.append(scoree[0][3])
            
        score_final1=pd.concat([pd.Series(score_list11),pd.Series(score_list21)],axis=1).reset_index(drop=True).T
        final1=pd.DataFrame(score_final1)
        final1=final1.set_axis(models,axis='columns')
        final1=final1.set_axis(['Only T/D','Tp / Dp'],axis='index')
        name1='1_Multi_MAPE.csv'
        final1.to_csv(name1)
        print('The file is saved')
        
        score_final2=pd.concat([pd.Series(score_list12),pd.Series(score_list22)],axis=1).reset_index(drop=True).T
        final2=pd.DataFrame(score_final2)
        final2=final2.set_axis(models,axis='columns')
        final2=final2.set_axis(['Only T/D','Tp / Dp'],axis='index')
        name2='1_Multi_MAE.csv'
        final2.to_csv(name2)
        print('The file is saved')
        
        score_final3=pd.concat([pd.Series(score_list13),pd.Series(score_list23)],axis=1).reset_index(drop=True).T
        final3=pd.DataFrame(score_final3)
        final3=final3.set_axis(models,axis='columns')
        final3=final3.set_axis(['Only T/D','Tp / Dp'],axis='index')
        name3='1_Multi_MSE.csv'
        final3.to_csv(name3)
        print('The file is saved')
        
        score_final4=pd.concat([pd.Series(score_list14),pd.Series(score_list24)],axis=1).reset_index(drop=True).T
        final4=pd.DataFrame(score_final4)
        final4=final4.set_axis(models,axis='columns')
        final4=final4.set_axis(['Only T/D','Tp / Dp'],axis='index')
        name4='1_Multi_RMSE.csv'
        final4.to_csv(name4)
        print('The file is saved')
        
        
singlemodel='LR'
multimodel='SVR'
basemodel='SVR'
metamodel='SVR'
j=55
def plot_prediction_section1(singlemodel,multimodel,basemodel,metamodel,range_number):
    j=range_number
   #x=x_data1/100
    x_count=np.arange(j)/100
    x_count=x_count.reshape(-1,1)

    y_single=single_section1(singlemodel,'p',j).reshape(1,-1)
    y_single_csv=pd.DataFrame(data=(y_single))
    y_single_csv.to_csv('y_single_1')
    
    plot_list=meta_model_predict(j,basemodel,metamodel)
    plot_list_csv=pd.DataFrame(data=(plot_list))
    plot_list_csv.to_csv('plot_list_1')
    
    plot_two=plot_list[:,1]/plot_list[:,0]

    y_mo=multi_output_section1(multimodel,'p', j)
    y_mo_csv=pd.DataFrame(data=(y_mo))
    y_mo_csv.to_csv('y_mo_1')
    
    
    #plott----------------------------------------------------------------
    plt.figure(figsize=(10, 6))  # Set the figure size (width: 10 inches, height: 6 inches)
    plt.rcParams["figure.dpi"] = 600 

    #plt.scatter(x_count,plot_list[:,2],label='MTRS')
    #plt.scatter(x_count,plot_two,label='MTRS')
    plt.scatter(x_count,y_mo[:,2],label='ST multi outputs')
    plt.scatter(x_count,plot_list[:,2],label='MTRS')

    plt.scatter(x_count,y_single,label='classical machine leaning')
    #plt.scatter(x_count,y_mo[:,2],label='ST multi outputs')
    x_countt=np.array((0,0.04,0.08,0.16,0.24,0.32,0.40)).reshape(7,1)
    plt.scatter(x_countt,y_data1[:,2].reshape(-1,1),c='k',label='Experimental')
    plt.xlabel('HAG-MP %') # X-Label
    plt.ylabel('T/D') # Y-Label
    plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.25))
    ax=plt.gca()
    ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=False,left=True,right=False)

    plt.show()


#============================================================================
'                             Section2 : P   az microbubble omde           '
#============================================================================
#hala scale ha hamechi chon 5 ta sample drim


def single_select_model(x,y,model,cvv,which,sc='neg_mean_absolute_percentage_error'):
    if model=='LR':
        poly=PolynomialFeatures()
        regressior=LinearRegression()        
        params=[{'poly':[None]},
                 {'poly__degree':[2,3,4]}]
        pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model== 'KNN':
        regressior = KNeighborsRegressor(n_jobs=60)
        params=[{'regressior__n_neighbors':[1,2,3,4]}]
        pipe=Pipeline(steps=[('regressior',regressior)]) 
    if model== 'DT':
        poly=PolynomialFeatures()
        regressior=DecisionTreeRegressor(random_state=0)
        params=[{'poly':[None],
                 'regressior__max_depth':[1,2,5,10],
         'regressior__min_samples_split':[2,4,8]},
                {'poly__degree':[2,3,4],
                 'regressior__max_depth':[1,2,5,10],
         'regressior__min_samples_split':[2,4,8]} ]    
        pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model=='RF':
        poly=PolynomialFeatures()
        regressior=RandomForestRegressor(random_state=0,n_jobs=60)
        params=[{'poly':[None],
                'regressior__n_estimators':[2,10,40],
                    'regressior__max_depth':[1,2,5,10]},
                {'poly__degree':[2,3,4],
                'regressior__n_estimators':[2,10,40],
                    'regressior__max_depth':[1,2,5,10]}]
        pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model=='SVR':
        poly=PolynomialFeatures()
        #scaler=MinMaxScaler()
        scaler=StandardScaler()

        regressior = SVR()
       # params=[{'poly__degree':[1,2]},
        #        {'regressior__kernel': ['rbf'],
         #           'regressior__gamma':np.logspace(-2, 2, 5),
          #          'regressior__C':[0.1,1e0, 1e1,1e2, 1e3]},
           #         {'regressior__kernel':['linear'],
            #        'regressior__C':[0.1,1e0, 1e1,1e2, 1e3]},
             #       {'regressior__kernel':['poly'],
              #       'regressior__C':[0.1,1e0, 1e1,1e2, 1e3],
               #      'regressior__degree':[1,2]}]   
        params=[{'poly':[None],'regressior__kernel': ['rbf'],
                    'regressior__gamma':np.logspace(-2, 2, 5),
                    'regressior__C':[00.1,0.1,1e0, 1e3]},
                    {'poly':[None],'regressior__kernel':['linear'],
                    'regressior__C':[00.1,0.1,1e0, 1e3]},
                    {'poly':[None],'regressior__kernel':['poly'],
                     'regressior__C':[00.1,0.1,1e0, 1e3],
                     'regressior__degree':[2,3,4]},
                    {'poly__degree':[2,3,4],'regressior__kernel': ['rbf'],
                    'regressior__gamma':np.logspace(-2, 2, 5),
                    'regressior__C':[00.1,0.1,1e0,  1e3]},
                    {'poly__degree':[2,3,4],'regressior__kernel':['linear'],
                    'regressior__C':[00.1,0.1,1e0,  1e3]},
                    {'poly__degree':[2,3,4],'regressior__kernel':['poly'],
                     'regressior__C':[00.1,0.1,1e0,  1e3],
                     'regressior__degree':[2,3,4]}]  
          
         
        pipe=Pipeline(steps=[('poly',poly),('scaler',scaler),('regressior',regressior)])
    if model=='MLP':  
        poly=PolynomialFeatures()
        regressior=MLPRegressor(solver='lbfgs',random_state=40,max_iter=200,activation='tanh')
        #scaler=MinMaxScaler()
        scaler=StandardScaler()
        
        params=[{'poly':[None],
                 'regressior__hidden_layer_sizes':[(2,),(3,),(4,),(5,),(20,),(100,),(200,)],
                'regressior__alpha':[0.001,0.0001,0.01,1]},
                {'poly__degree':[2,3,4],
                 'regressior__hidden_layer_sizes':[(2,),(3,),(4,),(5,),(20,),(100,),(200,)],
                'regressior__alpha':[0.001,0.0001,0.01,1]}]
        pipe=Pipeline(steps=[('poly',poly),('scaler',scaler),('regressior',regressior)])
    kfold1=KFold(n_splits=cvv,shuffle=False)
    kfold2=KFold(n_splits=cvv+1,shuffle=False)  
    #scoring_list=sc
    scoring_list='neg_mean_absolute_error'
    #scoring_list='neg_mean_absolute_percentage_error'
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list
                                    #,refit=refit_strategy
                                    ,n_jobs=60)
    grid.fit(x,y)
    if which=='gridsearch':
        
        return grid
    if which=='score':
        cross_score=cross_val_score(grid.best_estimator_, x,y,cv=kfold1,scoring=sc,n_jobs=60)
        print(cross_score.mean())

        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid.best_estimator_, x,y,cv=kfold1,n_jobs=60)
        return cross_pred

print(cross_score.mean())

#grid.best_score_
grid.best_params_

#cv = pd.DataFrame(grid.cv_results_)
#scores = np.array(cv.mean_test_score).reshape(6, 6)
#sns.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'],
# ylabel='C', yticklabels=param_grid['C'], cmap="viridis")

def select_model(x,y,model,cvv,which,sc='neg_mean_absolute_percentage_error'):
    if model=='LR':
        poly=PolynomialFeatures()
        base_model=LinearRegression()        
        regressior = MultiOutputRegressor(base_model,n_jobs=60) 
        params=[{'regressor__poly__degree':[2],
                'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
       # {'regressor__poly':[None],
        #        'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}
        mini_pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model== 'KNN':
        base_model = KNeighborsRegressor()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__n_neighbors':[1,2],
                'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model== 'DT':
        base_model=DecisionTreeRegressor(random_state=0)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__max_depth':[1,2],
         'regressor__regressior__estimator__min_samples_split':[2,4],
                 'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]     
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='RF':
        base_model=RandomForestRegressor(random_state=0,n_jobs=60)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__n_estimators':[2,10],
                    'regressor__regressior__estimator__max_depth':[1,2],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='SVR':
        poly=PolynomialFeatures(degree=2)
       # scaler=MinMaxScaler()
        #scaler=StandardScaler()
        scaler=PowerTransformer()
        base_model = SVR()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)        
        params=[{'regressor__regressior__estimator__kernel': ['rbf'],
                    'regressor__regressior__estimator__gamma':[0.1,1e0,1e2],
                    'regressor__regressior__estimator__C':[0.1,1e0, 1e1],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                    {'regressor__regressior__estimator__kernel':['linear'],
                    'regressor__regressior__estimator__C':[0.1,1e0, 1e1],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                    {'regressor__regressior__estimator__kernel':['poly'],
                     'regressor__regressior__estimator__C':[0.1,1e0, 1e1],
                     'regressor__regressior__estimator__degree':[2,4],
                             'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]

        mini_pipe=Pipeline(steps=[('poly',poly),('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='MLP':
        poly=PolynomialFeatures()
        base_model=MLPRegressor(solver='lbfgs',random_state=40,max_iter=200,activation='tanh',
                                hidden_layer_sizes=(4,),
                                alpha=0.01)
        #scaler=MinMaxScaler()
        scaler=StandardScaler()
       # scaler=PowerTransformer()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)        
        params=[{'regressor__poly__degree':[4],
                         'transformer':[None,StandardScaler()]}]
        
        #{'regressor__poly':[None],
         #                'transformer':[None,PowerTransformer()]},
                
        mini_pipe=Pipeline(steps=[('poly',poly),('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

        #('scaler',scaler),
    kfold1=KFold(n_splits=cvv,shuffle=False)
    kfold2=KFold(n_splits=cvv+1,shuffle=False)
    #scoring_list=['neg_mean_absolute_percentage_error',
      #            'neg_mean_absolute_error',
       #           'neg_mean_squared_error',
        #          'neg_root_mean_squared_error']
    #scoring_list=sc
    scoring_list='neg_mean_absolute_error'
    #scoring_list='neg_mean_absolute_percentage_error'
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list
                                    
                                    #,refit=refit_strategy
                                    , n_jobs=60)
    grid.fit(x,y)
    if which=='gridsearch':  
        
        return grid
    if which=='score':
        cross_score=cross_val_score(grid.best_estimator_, x,y,cv=kfold1,scoring=sc,n_jobs=60)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid.best_estimator_, x,y,cv=kfold1,n_jobs=60)
        return cross_pred





def select_model2(x,y,model,cvv,which,sc='neg_mean_absolute_percentage_error'):
    if model=='LR':
        base_model=LinearRegression()        
        regressior = MultiOutputRegressor(base_model,n_jobs=60) 
        params=[{'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
       # {'regressor__poly':[None],
        #        'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}
        mini_pipe=Pipeline(steps=[('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model== 'KNN':
        base_model = KNeighborsRegressor()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__n_neighbors':[1,2,3,4],
                'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model== 'DT':
        base_model=DecisionTreeRegressor(random_state=0)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{
                'regressor__regressior__estimator__max_depth':[1,2,3,5],
         'regressor__regressior__estimator__min_samples_split':[2,4,8],
                 'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]     
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='RF':
        base_model=RandomForestRegressor(random_state=0,n_jobs=60)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__n_estimators':[2],
                    'regressor__regressior__estimator__max_depth':[1,2,3,4],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='SVR':
       # scaler=MinMaxScaler()
        #scaler=StandardScaler()
        scaler=PowerTransformer()
        base_model = SVR()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)        
        params=[{'regressor__scaler':[None,PowerTransformer()],
                'regressor__regressior__estimator__kernel': ['rbf'],
                    'regressor__regressior__estimator__gamma':[0.1,1e0,1e2],
                    'regressor__regressior__estimator__C':[0.1,1e0, 1e1],
                            'transformer':[None,PowerTransformer()]},
                    {'regressor__scaler':[None,PowerTransformer()],
                     'regressor__regressior__estimator__kernel':['linear'],
                    'regressor__regressior__estimator__C':[0.1,1e0, 1e1],
                            'transformer':[None,PowerTransformer()]}]

        mini_pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='MLP':
        base_model=MLPRegressor(solver='lbfgs',random_state=40,max_iter=200,activation='tanh',
                                hidden_layer_sizes=(4,),
                                alpha=0.01)
        #scaler=MinMaxScaler()
        scaler=StandardScaler()
       # scaler=PowerTransformer()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)        
        params=[{
                 'regressor__scaler':[None,StandardScaler()],
                         'transformer':[None,PowerTransformer()]}]
        
        
        mini_pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

        #('scaler',scaler),
    kfold1=KFold(n_splits=cvv,shuffle=False)
    kfold2=KFold(n_splits=cvv+1,shuffle=False)
    #scoring_list=['neg_mean_absolute_percentage_error',
      #            'neg_mean_absolute_error',
       #           'neg_mean_squared_error',
        #          'neg_root_mean_squared_error']
    #scoring_list=sc
    scoring_list='neg_mean_absolute_error'
    #scoring_list='neg_mean_absolute_percentage_error'
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list
                                    
                                    #,refit=refit_strategy
                                    , n_jobs=60)
    grid.fit(x,y)
    if which=='gridsearch':  
        
        return grid
    if which=='score':
        cross_score=cross_val_score(grid.best_estimator_, x,y,cv=kfold1,scoring=sc,n_jobs=60)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid.best_estimator_, x,y,cv=kfold1,n_jobs=60)
        return cross_pred



s=metaa_model2('','LR','s')

























#================paeeni ghadimie==================================
#with scale
def select_model(x,y,model,cvv,which,sc='neg_mean_absolute_percentage_error'):
    if model=='LR':
        poly=PolynomialFeatures(degree=2)
        base_model=LinearRegression()        
        regressior = MultiOutputRegressor(base_model,n_jobs=60) 
        params=[{'transformer':[None,PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe,transformer= PowerTransformer())

    if model== 'KNN':
        base_model = KNeighborsRegressor()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__n_neighbors':[1,2,3,4],
                'transformer':[None,PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=PowerTransformer())

    if model== 'DT':
        base_model=DecisionTreeRegressor(random_state=0)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{
                'regressor__regressior__estimator__max_depth':[1,2,5],
         'regressor__regressior__estimator__min_samples_split':[2,4],
                 'transformer':[None,PowerTransformer()]}]     
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=PowerTransformer())

    if model=='RF':
        base_model=RandomForestRegressor(n_estimators=2,random_state=0,n_jobs=60)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__max_depth':[1,2,5,10],
                            'transformer':[None,PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=PowerTransformer())

    if model=='SVR':
       # scaler=MinMaxScaler()
        #scaler=StandardScaler()
        scaler=PowerTransformer()
        base_model = SVR(kernel='rbf')
        regressior = MultiOutputRegressor(base_model,n_jobs=60)        
        params=[{ 'regressor__regressior__estimator__gamma':[0.1,1e0],
                    'regressor__regressior__estimator__C':[1e0, 1e1],
                            'transformer':[None,PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=PowerTransformer())

    if model=='MLP':
        poly=PolynomialFeatures(degree=3)
        base_model=MLPRegressor(solver='adam',random_state=40,max_iter=100,activation='relu',
                                hidden_layer_sizes=(100,),
                                alpha=0.01)
        scaler=MinMaxScaler()
        #scaler=StandardScaler()
       #scaler=PowerTransformer()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)        
        params=[{'transformer':[PowerTransformer()]}]

        mini_pipe=Pipeline(steps=[('poly',poly),('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

        #('scaler',scaler),
    kfold1=KFold(n_splits=cvv,shuffle=False)
    kfold2=KFold(n_splits=cvv+1,shuffle=False)
    #scoring_list=['neg_mean_absolute_percentage_error',
      #            'neg_mean_absolute_error',
       #           'neg_mean_squared_error',
        #          'neg_root_mean_squared_error']
    #scoring_list=sc
    scoring_list='neg_mean_absolute_error'
    #scoring_list='neg_mean_absolute_percentage_error'
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list
                                    
                                    #,refit=refit_strategy
                                    , n_jobs=60)
    grid.fit(x,y)
    if which=='gridsearch':  
        
        return grid
    if which=='score':
        cross_score=cross_val_score(grid.best_estimator_, x,y,cv=kfold1,scoring=sc,n_jobs=60)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid.best_estimator_, x,y,cv=kfold1,n_jobs=60)
        return cross_pred

def select_model2(x,y,model,cvv,which,sc='neg_mean_absolute_percentage_error'):
    if model=='LR':
        base_model=LinearRegression()        
        regressior = MultiOutputRegressor(base_model,n_jobs=60) 
        params=[{'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
    if model== 'KNN':
        base_model = KNeighborsRegressor()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__n_neighbors':[1,2,3,4],
                'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model== 'DT':
        base_model=DecisionTreeRegressor(random_state=0)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{
                'regressor__regressior__estimator__max_depth':[1,2,5,10],
         'regressor__regressior__estimator__min_samples_split':[2,4,8],
                 'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]     
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='RF':
        base_model=RandomForestRegressor(random_state=0,n_jobs=60)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__n_estimators':[2,10],
                    'regressor__regressior__estimator__max_depth':[1,2,5,10],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='SVR':
       # scaler=MinMaxScaler()
        #scaler=StandardScaler()
        scaler=PowerTransformer()
        base_model = SVR()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)        
        params=[{'regressor__scaler':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()],
                'regressor__regressior__estimator__kernel': ['rbf'],
                    'regressor__regressior__estimator__gamma':np.logspace(-2, 2, 5),
                    'regressor__regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                    {'regressor__scaler':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()],
                     'regressor__regressior__estimator__kernel':['linear'],
                    'regressor__regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                    
                    {'regressor__scaler':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()],
                     'regressor__regressior__estimator__kernel':['poly'],
                     'regressor__regressior__estimator__C':[0.1,1e0, 1e1,1e2, 1e3],
                     'regressor__regressior__estimator__degree':[2,3,4],
                             'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]

        mini_pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='MLP':
        base_model=MLPRegressor(solver='adam',random_state=40,max_iter=100,activation='relu',
                                hidden_layer_sizes=(100,),
                                alpha=0.01)
        #scaler=MinMaxScaler()
        #scaler=StandardScaler()
        scaler=PowerTransformer()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)        
        params=[{'regressor__scaler':[PowerTransformer()],
                         'transformer':[None,PowerTransformer()]}]
        
        
        mini_pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

        #('scaler',scaler),
    kfold1=KFold(n_splits=cvv,shuffle=False)
    kfold2=KFold(n_splits=cvv+1,shuffle=False)
    #scoring_list=['neg_mean_absolute_percentage_error',
      #            'neg_mean_absolute_error',
       #           'neg_mean_squared_error',
        #          'neg_root_mean_squared_error']
    #scoring_list=sc
    scoring_list='neg_mean_absolute_error'
    #scoring_list='neg_mean_absolute_percentage_error'
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list
                                    
                                    #,refit=refit_strategy
                                    , n_jobs=60)
    grid.fit(x,y)
    if which=='gridsearch':  
        
        return grid
    if which=='score':
        cross_score=cross_val_score(grid.best_estimator_, x,y,cv=kfold1,scoring=sc,n_jobs=60)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid.best_estimator_, x,y,cv=kfold1,n_jobs=60)
        return cross_pred


grid.best_score_
grid.best_params_

pred=cross_pred
print(score1)

def single_section2(model,which,n=50):
    x=x_data2
    #y=np.array(data2_outputs[0]).reshape(-1,1)
    y=np.array(data2_outputs[0]).reshape(-1,)
    #grid=single_select_model(x, y, model, 5, 'gridsearch')
    if which=='p':
        grid=single_select_model(x, y, model, 5, 'gridsearch')
        #grid=single_select_model(x, y, model, 5, 'gridsearch')
        x_list=list(range(0,n))
        pred=grid.predict(np.array(x_list).reshape(-1,1))
        return pred
    if which=='s':
        #score=grid.best_score_
        pred=single_select_model(x, y, model, 5, 'prediction')
        score1=mean_absolute_percentage_error(np.array(data2_outputs[0]).reshape(-1,1),pred)
        
        score2=mean_absolute_error(np.array(data2_outputs[0]).reshape(-1,1),pred)
        
        score3=mean_squared_error(np.array(data2_outputs[0]).reshape(-1,1),pred)
        
        score44=mean_squared_error(np.array(data2_outputs[0]).reshape(-1,1),pred)
        score4=np.sqrt(score44)
        
        score=pd.DataFrame(data=((score1),(score2),(score3),(score4))) 
        
        return score

#s=multi_section2('LR','s')
def multi_section2(model,which,n=50):
    x=x_data2
    y=data2_outputs
    
    if which=='p':
        grid=select_model(x, y, model, 5, 'gridsearch')
        x_list=list(range(0,n))
        pred=grid.predict(np.array(x_list).reshape(-1,1))
        return pred
    if which=='s':    
        cross_pred=select_model(x, y, model, 5, 'prediction')
        
        #cross_pred=grid.predict(x_data2)
        #cross_pred=select_model(x,y,model,5,'prediction')
        pred=cross_pred[:,0]
        
        true=data2_outputs[:][0]
        
        score1=mean_absolute_percentage_error(true,pred)
        
        score2=mean_absolute_error(true,pred)
        
        score3=mean_squared_error(true,pred)
        
        score44=mean_squared_error(true,pred)
        score4=np.sqrt(score44)
        
        score=pd.DataFrame(data=((score1),(score2),(score3),(score4)))
        return score
           


def base_model2(x,y,model,cv,which,number=0):
    grid=select_model(x,y,model,cv,'gridsearch') 
    if which=='p':  
        y_pred=grid.predict(x)
        return y_pred,grid
    if which=='s':
        xf=x_data2[number].reshape(-1,1)
        y_pred=grid.predict(xf)
        return y_pred



def metaa_model2(model,meta_model,which):
    if which=='p':
        pre_x=x_data2
        y=data2_outputs
        x1=x_data2.reshape(-1)
        x1=pd.Series(x1)
        y_pred=base_model2(pre_x,y,model,5,'p')[0]
        x2=pd.DataFrame(y_pred)
        x=pd.concat([x1,x2],axis=1)
        x=np.array(x).reshape(-1,1011)
        meta_grid=select_model2(x,y,meta_model,5,'gridsearch')
        return meta_grid
    if which=='s':
 
        y_data2=np.array(data2_outputs)
        
        x1=x_data2.reshape(-1)
        x1=pd.Series(x1)
        y_pred=base_model2(x_data2.reshape(-1,1),y_data2,model,5,'p')[0]
        x2=pd.DataFrame(y_pred)  
        y=y_data2 
        x=pd.concat([x1,x2],axis=1)
        x=np.array(x).reshape(-1,1011)
        
        meta_pred=select_model2(x,y_data2,meta_model,5,'prediction')  
       # meta_grid=select_model(x,y_data2,meta_model,5,'gridsearch') 
        
        #x11=x_data2.reshape(-1)
       # x11=pd.Series(x11)
        ##y_predd=base_model2(x_data2.reshape(-1,1),y_data2,model,5,'p')[0]
        #x22=pd.DataFrame(y_pred)
        #x=pd.concat([x11,x22],axis=1)
        #x=np.array(x).reshape(-1,1011)
       # meta_pred=meta_grid.predict(x)
        score1=mean_absolute_percentage_error(y_data2[:,0],meta_pred[:,0])
        score2=mean_absolute_error(y_data2[:,0],meta_pred[:,0])
        score3=mean_squared_error(y_data2[:,0],meta_pred[:,0])
        score44=mean_squared_error(y_data2[:,0],meta_pred[:,0])
        score4=np.sqrt(score44)

        score=pd.DataFrame(data=((score1),(score2),(score3),(score4)))
        
        return score



def base_model_predict2(n,model):
    pre_x=x_data2
    y=data2_outputs
    grid=base_model2(pre_x,y,model,5,'p')[1]
    base_predict=grid.predict(np.array(range(0,n)).reshape(-1,1))
    return base_predict

def meta_model_predict2(j,model,meta_model):
    n=j 
    meta_grid=metaa_model2(model,meta_model,'p')
    x1_list=pd.Series(list(range(0,j)))
    xy_list=base_model_predict2(n,model)
    x2_list=pd.DataFrame(xy_list)
    x_final=pd.concat([x1_list,x2_list],axis=1)
    x_final=np.array(x_final).reshape(-1,1011)                
    meta_pred=meta_grid.predict(x_final)
    return meta_pred



def score_section2(which,sc='neg_mean_absolute_percentage_error'):
    
    if sc=='neg_mean_absolute_percentage_error':
        titsc='MAPE'
    if sc=='neg_mean_absolute_error':
        titsc='MAE'
    if sc=='neg_mean_squared_error':
        titsc='MSE'
    if sc=='neg_root_mean_squared_error':
        titsc='RMSE'
    models=['LR','KNN','DT','RF','SVR','MLP']
    if which=='MTRS':
        global score_list1,score_list2,score_list3,score_list4
        score_list1=[]
        score_list2=[]
        score_list3=[]
        score_list4=[]
        #**jabeja krdm meta modelo ba model
        for i in range(0,6):
            meta_model=models[i]
        
            for i in range(0,6):
                model=models[i]
                scoree=metaa_model2(model, meta_model, 's')
                #score=np.array(scoree).mean()
                print('this Loop passed===========================================================')
                score_list1.append(scoree[0][0])
                score_list2.append(scoree[0][1])
                score_list3.append(scoree[0][2])
                score_list4.append(scoree[0][3])
        
        final1=pd.DataFrame(data=((score_list1[0:6]),(score_list1[6:12]),(score_list1[12:18]),
                           (score_list1[18:24]),(score_list1[24:30]),(score_list1[30:36])),
                           index=models,
                           columns=models)
        name1='2_MTRS_MAPE.csv'
        final1.to_csv(name1)
        print('The file is saved')
        
        final2=pd.DataFrame(data=((score_list2[0:6]),(score_list2[6:12]),(score_list2[12:18]),
                           (score_list2[18:24]),(score_list2[24:30]),(score_list2[30:36])),
                           index=models,
                           columns=models)
        name2='2_MTRS_MAE.csv'
        final2.to_csv(name2)
        print('The file is saved')
        
        final3=pd.DataFrame(data=((score_list3[0:6]),(score_list3[6:12]),(score_list3[12:18]),
                           (score_list3[18:24]),(score_list3[24:30]),(score_list3[30:36])),
                           index=models,
                           columns=models)
        name3='2_MTRS_MSE.csv'
        final3.to_csv(name3)
        print('The file is saved')
        
        final4=pd.DataFrame(data=((score_list4[0:6]),(score_list4[6:12]),(score_list4[12:18]),
                           (score_list4[18:24]),(score_list4[24:30]),(score_list4[30:36])),
                           index=models,
                           columns=models)
        name4='2_MTRS_RMSE.csv'
        final4.to_csv(name4)
        print('The file is saved')
        
      #  return final
    
    if which=='single':
        score_list1=[]
        score_list2=[]
        score_list3=[]
        score_list4=[]
        for i in range(0,6):
            
            model=models[i]
            #score=single_section2(model,'s',sc=sc)
           # score_list.append(-1*(score.mean()))
            score=single_section2(model,'s')
            score_list1.append(score[0][0])
            score_list2.append(score[0][1])
            score_list3.append(score[0][2])
            score_list4.append(score[0][3])
        final1=pd.DataFrame(data=np.array(score_list1).reshape(1,-1),columns=models)
        name1='2_Single_MAPE.csv'
        final1.to_csv(name1)
        print('The file is saved')
        #return final
        final2=pd.DataFrame(data=np.array(score_list2).reshape(1,-1),columns=models)
        name2='2_Single_MAE.csv'
        final2.to_csv(name2)
        print('The file is saved')
        
        final3=pd.DataFrame(data=np.array(score_list3).reshape(1,-1),columns=models)
        name3='2_Single_MSE.csv'
        final3.to_csv(name3)
        print('The file is saved')
        
        final4=pd.DataFrame(data=np.array(score_list4).reshape(1,-1),columns=models)
        name4='2_Single_RMSE.csv'
        final4.to_csv(name4)
        print('The file is saved')
        
        #final=pd.DataFrame(data=np.array(score_list).reshape(1,-1),columns=models)
       # name='2_Single_'+titsc+'.csv'
       # final.to_csv(name)
       # print('The file is saved')
        #return final
        
    if which=='multi':
        score_list1=[]
        score_list2=[]
        score_list3=[]
        score_list4=[]
        for i in range(0,6):
            model=models[i]
            #model='LR'
            score=multi_section2(model,'s')
            #score_list.append(np.array(score).mean())
            score_list1.append(score[0][0])
            score_list2.append(score[0][1])
            score_list3.append(score[0][2])
            score_list4.append(score[0][3])
        
        final1=pd.DataFrame(data=np.array(score_list1))
        name1='2_Multi_MAPE.csv'
        final1.to_csv(name1)
        print('The file is saved')
        
        final2=pd.DataFrame(data=np.array(score_list2))
        name2='2_Multi_MAE.csv'
        final2.to_csv(name2)
        print('The file is saved')
        
        final3=pd.DataFrame(data=np.array(score_list3))
        name3='2_Multi_MSE.csv'
        final3.to_csv(name3)
        print('The file is saved')
        
        final4=pd.DataFrame(data=np.array(score_list4))
        name4='2_Multi_RMSE.csv'
        final4.to_csv(name4)
        print('The file is saved')


       
singlemodel='MLP'
multimodel='MLP'
basemodel='SVR'
metamodel='LR'
j=50
def plot_prediction_section2(singlemodel,multimodel,basemodel,metamodel,range_number=50):
    j=range_number
    plot_list=meta_model_predict2(j, basemodel, metamodel)
    global plot_all
    plot_all=plot_list
    #ezaf
    plot_list_csv=pd.DataFrame(data=(plot_list))
    plot_list_csv.to_csv('plot_list_2')
    #single and other
    #x=x_data2/100
    #y=data2_outputs
    x_count=np.arange(j)/100
    x_count=x_count.reshape(-1,1)
    global yyy
    yyy=single_section2(singlemodel,'p',j).reshape(1,-1)
    #ezaf
    yyy_csv=pd.DataFrame(data=(yyy))
    yyy_csv.to_csv('yyy_2')
        
        
    plot_list=plot_all[:,0]
    
        
    #global y_mo
    y_mo=multi_section2(multimodel,'p',j)[:,0]
    #ezaf
    y_mo_csv=pd.DataFrame(data=(y_mo))
    y_mo_csv.to_csv('y_mo_2')
        
    
    plt.figure(figsize=(10, 6))  # Set the figure size (width: 10 inches, height: 6 inches)
    plt.rcParams["figure.dpi"] = 600 
    
    plt.scatter(x_count,plot_list,label='MTRS')
    plt.scatter(x_count,yyy,label='classical machine learning')
    plt.scatter(x_count,y_mo,label='ST multi outputs')
    #dota balae shabihe hame
    #plt.scatter(x_count,y_mo_two,label='two per multi output')
    
    
    x_countt=np.array((0,0.04,0.08,0.16,0.24)).reshape(5,1)
    y5=np.array((0.419601,0.352938,0.241008,0.150044,0.196248)).reshape(5,1)
    plt.scatter(x_countt,y5,c='k',label='Experimental')
    plt.xlabel('HAG-MP %') # X-Label
    plt.ylabel('Density difference at P=0 bar') # Y-Label
    plt.legend(loc='upper center',bbox_to_anchor=(0,1.2))
    ax=plt.gca()
    ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=False,left=True,right=False)

    plt.show()

file_location = 'C:\\Users\\sunhouse\\plot_list_2'
f1=open(file_location,'r')
data1=pd.read_csv(f1)
data1=data1.drop('Unnamed: 0',axis=1)
plot_all=np.array(data1)



percentage=16
ii=3

percentage=24
ii=4
def plot_all_2(percentage):
    plt.figure(figsize=(10, 6))  # Set the figure size (width: 10 inches, height: 6 inches)
    plt.rcParams["figure.dpi"] = 600 
    #inja msihe for gozasht va hame ro hesab krd
    y=plot_all[percentage]
    x = np.arange(1010)
    # plot() is used for plotting a line plot
    plt.plot(x,y)
    # Adding title, xlabel and ylabel
    a='Dencity difference per pressure steps for '
    b=str(percentage/100)
    c=" HAG-MP %"
    titles=a+b+c
    plt.title(titles) # Title of the plot
    plt.xlabel('Pressure step(from 1 to 400 BAR)') # X-Label
    plt.ylabel('Density difference') # Y-Label
    ax=plt.gca()
    #ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
    ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=False,left=True,right=False)
    plt.ylim(0,0.5)

    plt.show()
    
    
def plot_experiment_2(percentage):
    plt.figure(figsize=(10, 6))  # Set the figure size (width: 10 inches, height: 6 inches)
    plt.rcParams["figure.dpi"] = 600 
    #inja msihe for gozasht va hame ro hesab krd
    y=np.array(data2_outputs)
    y=y[ii][:]
    x = np.arange(1010)
    # plot() is used for plotting a line plot
    plt.plot(y)
    # Adding title, xlabel and ylabel
    a='Dencity difference per pressure steps for '
    b=str(percentage/100)
    c=" HAG-MP %"
    titles=a+b+c
    plt.title(titles) # Title of the plot
    plt.xlabel('Pressure step(from 1 to 400 BAR)') # X-Label
    plt.ylabel('Density difference') # Y-Label
    ax=plt.gca()
    #ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
    ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=False,left=True,right=False)
    plt.ylim(0,0.5)

    plt.show()
    
        






#============================================================================
'                             Section3 : T 1/2                                '
#============================================================================


def single_select_model(x,y,model,cvv,which,sc='neg_mean_absolute_percentage_error'):
    if model=='LR':
        poly=PolynomialFeatures()
        regressior=LinearRegression()        
        params=[{'poly':[None]},
                 {'poly__degree':[2,3,4]}]
        pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model== 'KNN':
        regressior = KNeighborsRegressor(n_jobs=60)
        params=[{'regressior__n_neighbors':[1,2,3,4,5,6]}]
        pipe=Pipeline(steps=[('regressior',regressior)]) 
    if model== 'DT':
        poly=PolynomialFeatures()
        regressior=DecisionTreeRegressor(random_state=0)
        params=[{'poly':[None],
                 'regressior__max_depth':[1,2,5,10],
         'regressior__min_samples_split':[2,4,8]},
                {'poly__degree':[2,3,4],
                 'regressior__max_depth':[1,2,5,10],
         'regressior__min_samples_split':[2,4,8]} ]    
        pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model=='RF':
        poly=PolynomialFeatures()
        regressior=RandomForestRegressor(random_state=0,n_jobs=60)
        params=[{'poly':[None],
                'regressior__n_estimators':[2,10,40],
                    'regressior__max_depth':[1,2,5,10]},
                {'poly__degree':[2,3,4],
                'regressior__n_estimators':[2,10,40],
                    'regressior__max_depth':[1,2,5,10]}]
        pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model=='SVR':
        poly=PolynomialFeatures()
        #scaler=MinMaxScaler()
        #scaler=StandardScaler()
        scaler=PowerTransformer()
        regressior = SVR()
       # params=[{'poly__degree':[1,2]},
        #        {'regressior__kernel': ['rbf'],
         #           'regressior__gamma':np.logspace(-2, 2, 5),
          #          'regressior__C':[0.1,1e0, 1e1,1e2, 1e3]},
           #         {'regressior__kernel':['linear'],
            #        'regressior__C':[0.1,1e0, 1e1,1e2, 1e3]},
             #       {'regressior__kernel':['poly'],
              #       'regressior__C':[0.1,1e0, 1e1,1e2, 1e3],
               #      'regressior__degree':[1,2]}]   
        params=[{'poly':[None],'regressior__kernel': ['rbf'],
                    'regressior__gamma':np.logspace(-2, 2, 5),
                    'regressior__C':[00.1,0.1,1e0, 1e1,1e2,1e3]},
                    {'poly':[None],'regressior__kernel':['linear'],
                    'regressior__C':[00.1,0.1,1e0, 1e1,1e2,1e3]},
                    {'poly':[None],'regressior__kernel':['poly'],
                     'regressior__C':[00.1,0.1,1e0, 1e1,1e2,1e3],
                     'regressior__degree':[2,3,4]},
                    {'poly__degree':[2,3,4],'regressior__kernel': ['rbf'],
                    'regressior__gamma':np.logspace(-2, 2, 5),
                    'regressior__C':[00.1,0.1,1e0, 1e1,1e2,1e3]},
                    {'poly__degree':[2,3,4],'regressior__kernel':['linear'],
                    'regressior__C':[00.1,0.1,1e0, 1e1,1e2,1e3]},
                    {'poly__degree':[2,3,4],'regressior__kernel':['poly'],
                     'regressior__C':[00.1,0.1,1e0, 1e1,1e2,1e3],
                     'regressior__degree':[2,3,4]}]  
          
         
        pipe=Pipeline(steps=[('poly',poly),('scaler',scaler),('regressior',regressior)])
    if model=='MLP':  
        poly=PolynomialFeatures()
        regressior=MLPRegressor(solver='adam',random_state=40,max_iter=500)
        #scaler=MinMaxScaler()
        #scaler=StandardScaler()
        scaler=PowerTransformer()
        params=[{'poly':[None],
                 'regressior__hidden_layer_sizes':[(5,),(10,),(200,)],
                 'regressior__activation':['identity','relu','tanh'],
                'regressior__alpha':[0.001,0.01,1]},
                {'poly__degree':[2,4],
                 'regressior__hidden_layer_sizes':[(5,),(10,),(200,)],
                 'regressior__activation':['identity','relu','tanh'],
                'regressior__alpha':[0.001,0.01,1]}]
        pipe=Pipeline(steps=[('poly',poly),('scaler',scaler),('regressior',regressior)])
    kfold1=KFold(n_splits=cvv,shuffle=False)
    kfold2=KFold(n_splits=cvv+1,shuffle=False)  
    #scoring_list=sc
    scoring_list='neg_mean_absolute_error'
    #scoring_list='neg_mean_absolute_percentage_error'
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list
                                    #,refit=refit_strategy
                                    ,n_jobs=60)
    grid.fit(x,y)
    if which=='gridsearch':
        
        return grid
    if which=='score':
        cross_score=cross_val_score(grid.best_estimator_, x,y,cv=kfold1,scoring=sc,n_jobs=60)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid.best_estimator_, x,y,cv=kfold1,n_jobs=60)
        return cross_pred

#grid.best_score_
#grid.best_params_


#cv = pd.DataFrame(grid.cv_results_)
#scores = np.array(cv.mean_test_score).reshape(6, 6)
#sns.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'],
# ylabel='C', yticklabels=param_grid['C'], cmap="viridis")


#with scale
def select_model(x,y,model,cvv,which,sc='neg_mean_absolute_percentage_error'):
    if model=='LR':
        poly=PolynomialFeatures()
        base_model=LinearRegression()        
        regressior = MultiOutputRegressor(base_model,n_jobs=60) 
        params=[{'regressor__poly':[None],
                'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                {'regressor__poly__degree':[2,3,4],
                'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model== 'KNN':
        base_model = KNeighborsRegressor()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__n_neighbors':[1,2,3,4,5,6],
                'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model== 'DT':
        poly=PolynomialFeatures()
        base_model=DecisionTreeRegressor(random_state=0)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__poly':[None],
                'regressor__regressior__estimator__max_depth':[1,2,5,10],
         'regressor__regressior__estimator__min_samples_split':[2,4,8],
                 'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                {'regressor__poly__degree':[2,3,4],
                'regressor__regressior__estimator__max_depth':[1,2,5,10],
         'regressor__regressior__estimator__min_samples_split':[2,4,8],
                 'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]     
        mini_pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='RF':
        poly=PolynomialFeatures()
        base_model=RandomForestRegressor(random_state=0,n_jobs=60)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__poly':[None],
                'regressor__regressior__estimator__n_estimators':[2,10,40],
                    'regressor__regressior__estimator__max_depth':[1,2,5,10],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                {'regressor__poly__degree':[2,3,4],
                'regressor__regressior__estimator__n_estimators':[2,10,40],
                    'regressor__regressior__estimator__max_depth':[1,2,5,10],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='SVR':
        poly=PolynomialFeatures()
        #scaler=MinMaxScaler()
        scaler=PowerTransformer()
        base_model = SVR()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)        
        params=[{'regressor__poly':[None],
                'regressor__regressior__estimator__kernel': ['rbf'],
                    'regressor__regressior__estimator__gamma':np.logspace(-2, 2, 5),
                    'regressor__regressior__estimator__C':[00.1,0.1,1e0, 1e1,1e2,1e3],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                    {'regressor__poly':[None],
                     'regressor__regressior__estimator__kernel':['linear'],
                    'regressor__regressior__estimator__C':[00.1,0.1,1e0, 1e1,1e2,1e3],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                    {'regressor__poly':[None],
                     'regressor__regressior__estimator__kernel':['poly'],
                     'regressor__regressior__estimator__C':[00.1,0.1,1e0, 1e1,1e2,1e3],
                     'regressor__regressior__estimator__degree':[2,3,4],
                             'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                    {'regressor__poly__degree':[2,3,4],
                'regressor__regressior__estimator__kernel': ['rbf'],
                    'regressor__regressior__estimator__gamma':np.logspace(-2, 2, 5),
                    'regressor__regressior__estimator__C':[00.1,0.1,1e0, 1e1,1e2,1e3],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                    {'regressor__poly__degree':[2,3,4],
                     'regressor__regressior__estimator__kernel':['linear'],
                    'regressor__regressior__estimator__C':[00.1,0.1,1e0, 1e1,1e2,1e3],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                    {'regressor__poly__degree':[2,3,4],
                     'regressor__regressior__estimator__kernel':['poly'],
                     'regressor__regressior__estimator__C':[00.1,0.1,1e0, 1e1,1e2,1e3],
                     'regressor__regressior__estimator__degree':[2,3,4],
                             'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]

        mini_pipe=Pipeline(steps=[('poly',poly),('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='MLP':
        poly=PolynomialFeatures(degree=4)
        
        base_model=MLPRegressor(solver='adam',random_state=40,max_iter=500,hidden_layer_sizes=(200,),
                                activation='relu',alpha=0.01)
        #scaler=MinMaxScaler()
        scaler=PowerTransformer()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'transformer':[None,StandardScaler()]}]
        
        mini_pipe=Pipeline(steps=[('poly',poly),('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=StandardScaler())

        #('scaler',scaler),
        
    kfold1=KFold(n_splits=cvv,shuffle=False)
    kfold2=KFold(n_splits=cvv+1,shuffle=False)
    #scoring_list=['neg_mean_absolute_percentage_error',
      #            'neg_mean_absolute_error',
       #           'neg_mean_squared_error',
        #          'neg_root_mean_squared_error']
    #scoring_list=sc
    scoring_list='neg_mean_absolute_error'
    #scoring_list='neg_mean_absolute_percentage_error'
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list
                                    
                                    #,refit=refit_strategy
                                    ,
                                    
                                    n_jobs=60)
    grid.fit(x,y)
    if which=='gridsearch':  
        
        return grid
    if which=='score':
        cross_score=cross_val_score(grid.best_estimator_, x,y,cv=kfold1,scoring=sc,n_jobs=60)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid.best_estimator_, x,y,cv=kfold1,n_jobs=60)
        return cross_pred
    
    
def select_model2(x,y,model,cvv,which,sc='neg_mean_absolute_percentage_error'):
    if model=='LR':
        poly=PolynomialFeatures()
        base_model=LinearRegression()        
        regressior = MultiOutputRegressor(base_model,n_jobs=60) 
        params=[{'regressor__poly':[None],
                'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                {'regressor__poly__degree':[2,3,4],
                'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model== 'KNN':
        base_model = KNeighborsRegressor()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__regressior__estimator__n_neighbors':[1,2,3,4,5,6],
                'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model== 'DT':
        poly=PolynomialFeatures()
        base_model=DecisionTreeRegressor(random_state=0)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__poly':[None],
                'regressor__regressior__estimator__max_depth':[1,2,5,10],
         'regressor__regressior__estimator__min_samples_split':[2,4,8],
                 'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                {'regressor__poly__degree':[2,3,4],
                'regressor__regressior__estimator__max_depth':[1,2,5,10],
         'regressor__regressior__estimator__min_samples_split':[2,4,8],
                 'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]     
        mini_pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='RF':
        poly=PolynomialFeatures()
        base_model=RandomForestRegressor(random_state=0,n_jobs=60)
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__poly':[None],
                'regressor__regressior__estimator__n_estimators':[2,10,40],
                    'regressor__regressior__estimator__max_depth':[1,2,5,10],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                {'regressor__poly__degree':[2,3,4],
                'regressor__regressior__estimator__n_estimators':[2,10,40],
                    'regressor__regressior__estimator__max_depth':[1,2,5,10],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='SVR':
    
        #scaler=MinMaxScaler()
        scaler=PowerTransformer()
        base_model = SVR()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)        
        params=[{'regressor__scaler':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()],
                'regressor__regressior__estimator__kernel': ['rbf'],
                    'regressor__regressior__estimator__gamma':np.logspace(-2, 2, 5),
                    'regressor__regressior__estimator__C':[00.1,0.1,1e0, 1e1,1e2,1e3],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                    {'regressor__scaler':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()],
                     'regressor__regressior__estimator__kernel':['linear'],
                    'regressor__regressior__estimator__C':[00.1,0.1,1e0, 1e1,1e2,1e3],
                            'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]},
                    {'regressor__scaler':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()],
                      
                     'regressor__regressior__estimator__kernel':['poly'],
                     'regressor__regressior__estimator__C':[00.1,0.1,1e0, 1e1,1e2,1e3],
                     'regressor__regressior__estimator__degree':[2,3,4],
                             'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]

        mini_pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

    if model=='MLP':
        base_model=MLPRegressor(solver='adam',random_state=40,max_iter=500,hidden_layer_sizes=(200,))
        #scaler=MinMaxScaler()
        scaler=PowerTransformer()
        regressior = MultiOutputRegressor(base_model,n_jobs=60)
        params=[{'regressor__scaler':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()],
                 'regressor__regressior__estimator__activation':['relu','tanh'],
                 'regressor__regressior__estimator__alpha':[0.01,0.1,1],
                         'transformer':[None,MinMaxScaler(),StandardScaler(),PowerTransformer()]}]
        mini_pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())

        #('scaler',scaler),
        
    kfold1=KFold(n_splits=cvv,shuffle=False)
    kfold2=KFold(n_splits=cvv+1,shuffle=False)
    #scoring_list=['neg_mean_absolute_percentage_error',
      #            'neg_mean_absolute_error',
       #           'neg_mean_squared_error',
        #          'neg_root_mean_squared_error']
    #scoring_list=sc
    scoring_list='neg_mean_absolute_error'
    #scoring_list='neg_mean_absolute_percentage_error'
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list
                                    
                                    #,refit=refit_strategy
                                    ,
                                    
                                    n_jobs=60)
    grid.fit(x,y)
    if which=='gridsearch':  
        
        return grid
    if which=='score':
        cross_score=cross_val_score(grid.best_estimator_, x,y,cv=kfold1,scoring=sc,n_jobs=60)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid.best_estimator_, x,y,cv=kfold1,n_jobs=60)
        return cross_pred
    
    
    

    
def half_timee(a):
    a=np.array(a)
    ii_list=[]
    for i in range(0,7):
        b=a[i]
        b_new=[]
        for j in range(0,21):
            b_new.append(int(b[j]))
            
        if 175 in b_new:
            for k in range(0,21):
                if b_new[k]==175:
                    ii_list.append(k*20)
                    break
                    
        else:
            diff=[]
            diff_list=[]
            
            for z in range(0,21):
                
                diff=b_new[z]-175
                diff_list.append(diff)
            diff_pos=[]
            diff_neg=[]
            cneg=0
            cpos=0
            for h in range(0,21):
                if diff_list[h]>0:
                    diff_pos.append(diff_list[h])
                    cpos=cpos+1
                if diff_list[h]<0:
                    diff_neg.append(diff_list[h])
                    cneg=cneg+1
            if cpos==0 or cneg==0:
                if cpos==0:
                    ii_list.append(400)
                    
                if cneg==0:
                    ii=0
                    ii_list.append(0)
            else:
                diff_pos_sort=sorted(diff_pos)
                diff_neg_sort=sorted(diff_neg)
                i2=diff_list.index(diff_pos_sort[0])
                i1=diff_list.index(diff_neg_sort[cneg-1])
                x1=i1*20
                x2=i2*20
                y1=b[i1]
                y2=b[i2]
                K=(y2-y1)/(x2-x1)
                B=((y1*x2)-(y2*x1))/(x2-x1)
                #Y=K*X+B
                ii=(175-B)/(K)
                ii_list.append(ii)
   
                        
    return ii_list

def half_timee3(a):
    
    b=a
    b_new=[]
    for j in range(0,21):
        b_new.append(int(b[0][j]))
            
    if 175 in b_new:
        for k in range(0,21):
            if b_new[k]==175:
                return k*20
                    
    else:
        diff=[]
        diff_list=[]
            
        for z in range(0,21):
                
            diff=b_new[z]-175
            diff_list.append(diff)
        diff_pos=[]
        diff_neg=[]
        cneg=0
        cpos=0
        for h in range(0,21):
            if diff_list[h]>0:
                diff_pos.append(diff_list[h])
                cpos=cpos+1
            if diff_list[h]<0:
                diff_neg.append(diff_list[h])
                cneg=cneg+1
        if cpos==0 or cneg==0:
            if cpos==0:
                return 400
                    
            if cneg==0:
                return 0
                    
        else:
            diff_pos_sort=sorted(diff_pos)
            diff_neg_sort=sorted(diff_neg)
            i2=diff_list.index(diff_pos_sort[0])
            i1=diff_list.index(diff_neg_sort[cneg-1])
            x1=i1*20
            x2=i2*20
            y1=b[0][i1]
            y2=b[0][i2]
            K=(y2-y1)/(x2-x1)
            B=((y1*x2)-(y2*x1))/(x2-x1)
            #Y=K*X+B
            ii=(175-B)/(K)
            return ii


model='SVR'
def single_section3(model,which,n=50,sc='neg_mean_absolute_percentage_error'):
    x=x_data3
    y=half_time
    if which=='p':
        grid=single_select_model(x, half_time, model, 7,'gridsearch',sc=sc)
        pred=grid.predict(np.array(list(range(0,n))).reshape(-1,1))
        return pred
    if which=='s':
        #score=grid.best_score_
        pred=single_select_model(x,half_time,model,7,'prediction')
        score1=mean_absolute_percentage_error(half_time,pred)
        
        score2=mean_absolute_error(half_time,pred)
        
        score3=mean_squared_error(half_time,pred)
        
        score44=mean_squared_error(half_time,pred)
        score4=np.sqrt(score44)
        
        score=pd.DataFrame(data=((score1),(score2),(score3),(score4))) 
        
        return score

def multi_section3(model,which,n=50):
    x=x_data3
    y1=y_data3_foam
    y2=y_data3_drain
    if which=='p':
        grid1=select_model(x, y2, model, 7, 'gridsearch')
        pred1=grid1.predict(np.array(list(range(0,n))).reshape(-1,1))
        return pred1

    if which=='s':
        
        grid=select_model(x,y2,model,7,'prediction')
        #pred2=grid.predict(x_data3)
        pred=half_timee(grid)
        
        score1=mean_absolute_percentage_error(half_time,np.array(pred).reshape(-1,1))
        score2=mean_absolute_error(half_time,np.array(pred).reshape(-1,1))
        score3=mean_squared_error(half_time,np.array(pred).reshape(-1,1))
        score44=mean_squared_error(half_time,np.array(pred).reshape(-1,1))
        score4=np.sqrt(score44)
        
        score=pd.DataFrame(data=((score1),(score2),(score3),(score4)))
               
        return score
    



def base_model3(x,y,model,cv,which,number=0):
    grid=select_model(x,y,model,cv,'gridsearch') 
    if which=='p':  
        y_pred=grid.predict(x)
        return y_pred,grid
    if which=='s':
        xf=x_data3[number].reshape(-1,1)
        y_pred=grid.predict(xf)
        return y_pred



def metaa_model3(model,meta_model,which):
    if which=='p_d':
        pre_x=x_data3
        y=y_data3_drain
        x1=x_data3.reshape(-1)
        x1=pd.Series(x1)
        y_pred=base_model3(pre_x,y,model,7,'p')[0]
        x2=pd.DataFrame(y_pred)
        x=pd.concat([x1,x2],axis=1)
        x=np.array(x).reshape(-1,22)
        meta_grid=select_model2(x,y,meta_model,7,'gridsearch')
        return meta_grid
    if which=='p_f':
        pre_x=x_data3
        y=y_data3_foam
        x1=x_data3.reshape(-1)
        x1=pd.Series(x1)
        y_pred=base_model3(pre_x,y,model,7,'p')[0]
        x2=pd.DataFrame(y_pred)
        x=pd.concat([x1,x2],axis=1)
        x=np.array(x).reshape(-1,22)
        meta_grid=select_model2(x,y,meta_model,7,'gridsearch')
        return meta_grid
    if which=='s':

        y_data3_drainn=np.array(y_data3_drain)

        x1=x_data3.reshape(-1)
        x1=pd.Series(x1) 
        y_pred=base_model3(x_data3.reshape(-1,1),y_data3_drainn,model,7,'p')[0]
        x2=pd.DataFrame(y_pred)  
        #y=y_data3_drainn
        x=pd.concat([x1,x2],axis=1)
        x=np.array(x).reshape(-1,22)
        meta_pred=select_model2(x,y_data3_drainn,meta_model,7,'prediction')   
        meta_t=half_timee(meta_pred)
        
        
        score1=mean_absolute_percentage_error(half_time,meta_t)
        score2=mean_absolute_error(half_time,meta_t)
        score3=mean_squared_error(half_time,meta_t)
        score44=mean_squared_error(half_time,meta_t)
        score4=np.sqrt(score44)
        score=pd.DataFrame(data=((score1),(score2),(score3),(score4)))
       
        return score

def base_model_predict3(n,model,which):
    if which=='p_d':
        y=y_data3_drain
        
    if which=='p_f':
        y=y_data3_foam
        
    pre_x=x_data3
    grid=base_model3(pre_x,y,model,7,'p')[1]
    base_predict=grid.predict(np.array(range(0,n)).reshape(-1,1))
    return base_predict

def meta_model_predict3(j,model,meta_model,which):
    n=j 
    if which=='p_d':
        meta_grid=metaa_model3(model,meta_model,'p_d')
        x1_list=pd.Series(list(range(0,j)))
        xy_list=base_model_predict3(n,model,'p_d')
        x2_list=pd.DataFrame(xy_list)
        x_final=pd.concat([x1_list,x2_list],axis=1)
        x_final=np.array(x_final).reshape(-1,22)                
        meta_pred=meta_grid.predict(x_final)
        return meta_pred
        
    if which=='p_f':
        meta_grid=metaa_model3(model,meta_model,'p_f')
        x1_list=pd.Series(list(range(0,j)))
        xy_list=base_model_predict3(n,model,'p_f')
        x2_list=pd.DataFrame(xy_list)
        x_final=pd.concat([x1_list,x2_list],axis=1)
        x_final=np.array(x_final).reshape(-1,22)                
        meta_pred=meta_grid.predict(x_final)
        return meta_pred



def score_section3(which,sc='neg_mean_absolute_percentage_error'):
    
    if sc=='neg_mean_absolute_percentage_error':
        titsc='MAPE'
    if sc=='neg_mean_absolute_error':
        titsc='MAE'
    if sc=='neg_mean_squared_error':
        titsc='MSE'
    if sc=='neg_root_mean_squared_error':
        titsc='RMSE'
    models=['LR','KNN','DT','RF','SVR','MLP']
    global final1,final2,final3,final4
    if which=='MTRS':
        score_list1=[]
        score_list2=[]
        score_list3=[]
        score_list4=[]
        #**jabeja krdm meta modelo ba model
        for i in range(0,6):
            meta_model=models[i]
            
            for i in range(0,6):
                model=models[i]
                scoree=metaa_model3(model, meta_model, 's')
                #score=np.array(scoree).mean()
                print('this Loop passed===========================================================')
                score_list1.append(scoree[0][0])
                score_list2.append(scoree[0][1])
                score_list3.append(scoree[0][2])
                score_list4.append(scoree[0][3])
        
        final1=pd.DataFrame(data=((score_list1[0:6]),(score_list1[6:12]),(score_list1[12:18]),
                           (score_list1[18:24]),(score_list1[24:30]),(score_list1[30:36])),
                           index=models,
                           columns=models)
        name1='3_MTRS_MAPE.csv'
        final1.to_csv(name1)
        print('The file is saved')
        
        final2=pd.DataFrame(data=((score_list2[0:6]),(score_list2[6:12]),(score_list2[12:18]),
                           (score_list2[18:24]),(score_list2[24:30]),(score_list2[30:36])),
                           index=models,
                           columns=models)
        name2='3_MTRS_MAE.csv'
        final2.to_csv(name2)
        print('The file is saved')
        
        final3=pd.DataFrame(data=((score_list3[0:6]),(score_list3[6:12]),(score_list3[12:18]),
                           (score_list3[18:24]),(score_list3[24:30]),(score_list3[30:36])),
                           index=models,
                           columns=models)
        name3='3_MTRS_MSE.csv'
        final3.to_csv(name3)
        print('The file is saved')
        
        final4=pd.DataFrame(data=((score_list4[0:6]),(score_list4[6:12]),(score_list4[12:18]),
                           (score_list4[18:24]),(score_list4[24:30]),(score_list4[30:36])),
                           index=models,
                           columns=models)
        name4='3_MTRS_RMSE.csv'
        final4.to_csv(name4)
        print('The file is saved')
        
      #  return final
    
    if which=='single':
        score_list1=[]
        score_list2=[]
        score_list3=[]
        score_list4=[]
        for i in range(0,6):
            model=models[i]
            #score=single_section3(model,'s',sc=sc)
            #score_list.append(-1*(score.mean()))
            score=single_section3(model,'s')
            score_list1.append(score[0][0])
            score_list2.append(score[0][1])
            score_list3.append(score[0][2])
            score_list4.append(score[0][3])
        
        #final=pd.DataFrame(data=np.array(score_list).reshape(1,-1),columns=models)
        #name='3_Single_'+titsc+'.csv'
        #final.to_csv(name)
       # print('The file is saved')
        #return final
        
        final1=pd.DataFrame(data=np.array(score_list1).reshape(1,-1),columns=models)
        name1='3_Single_MAPE.csv'
        final1.to_csv(name1)
        print('The file is saved')
        #return final
        final2=pd.DataFrame(data=np.array(score_list2).reshape(1,-1),columns=models)
        name2='3_Single_MAE.csv'
        final2.to_csv(name2)
        print('The file is saved')
        
        final3=pd.DataFrame(data=np.array(score_list3).reshape(1,-1),columns=models)
        name3='3_Single_MSE.csv'
        final3.to_csv(name3)
        print('The file is saved')
        
        final4=pd.DataFrame(data=np.array(score_list4).reshape(1,-1),columns=models)
        name4='3_Single_RMSE.csv'
        final4.to_csv(name4)
        print('The file is saved')
        
    if which=='multi':
        score_list1=[]
        score_list2=[]
        score_list3=[]
        score_list4=[]
        for i in range(0,6):
            model=models[i]
            #model='LR'
            score=multi_section3(model,'s')
            #score_list.append(np.array(score).mean())
            score_list1.append(score[0][0])
            score_list2.append(score[0][1])
            score_list3.append(score[0][2])
            score_list4.append(score[0][3])
        
        final1=pd.DataFrame(data=np.array(score_list1))
        name1='3_Multi_MAPE.csv'
        final1.to_csv(name1)
        print('The file is saved')
        
        final2=pd.DataFrame(data=np.array(score_list2))
        name2='3_Multi_MAE.csv'
        final2.to_csv(name2)
        print('The file is saved')
        
        final3=pd.DataFrame(data=np.array(score_list3))
        name3='3_Multi_MSE.csv'
        final3.to_csv(name3)
        print('The file is saved')
        
        final4=pd.DataFrame(data=np.array(score_list4))
        name4='3_Multi_RMSE.csv'
        final4.to_csv(name4)
        print('The file is saved')


singlemodel='SVR'
multimodel='SVR'
basemodel='SVR'
metamodel='MLP'
J=50
#it is not true
def plot_prediction_section3(singlemodel,multimodel,basemodel,metamodel,which,range_number=50,percentage=0):
    j=range_number
    if which=='T1/2':

      #  x=x_data3/100
       # y=half_time
        x_count=np.arange(j)/100
        x_count=x_count.reshape(-1,1) 
        y_single=single_section3(singlemodel,'p',j).reshape(1,-1)
        #ezaf
        y_single_csv=pd.DataFrame(data=(y_single))
        y_single_csv.to_csv('y_single_3')
        
        
        pre_y_multi=multi_section3(multimodel,'p',j)
        y_multi=half_timee(pre_y_multi)
        #ezaf
        y_multi_csv=pd.DataFrame(data=(y_multi))
        y_multi_csv.to_csv('y_multi_3')
        
        pre_y_mtrs=meta_model_predict3(j,basemodel,metamodel,'p_d')
        y_mtrs=half_timee(pre_y_mtrs)
        #ezaf
        y_mtrs_csv=pd.DataFrame(data=(y_mtrs))
        y_mtrs_csv.to_csv('y_mtrs_3')

        
    if which=='drain_foam':
        foam=meta_model_predict3(j,basemodel,metamodel,'p_f')
        drain=meta_model_predict3(j,basemodel,metamodel,'p_d')

        t=half_timee(drain)
        #n=np.array(range(0,401,20))
        
        t_csv=pd.DataFrame(data=(t))
        t_csv.to_csv('t_'+str(percentage))
        
        y_foam_csv=pd.DataFrame(data=(foam))
        y_foam_csv.to_csv('yfoam')
     
        y_drain_csv=pd.DataFrame(data=(drain))
        y_drain_csv.to_csv('yfoam')
        
def half_timee4(a,n):
    a=np.array(a)
    ii_list=[]
    for i in range(0,n):
        b=a[i]
        b_new=[]
        for j in range(0,21):
            b_new.append(int(b[j]))
            
        if 175 in b_new:
            for k in range(0,21):
                if b_new[k]==175:
                    ii_list.append(k*20)
                    
        else:
            diff=[]
            diff_list=[]
            
            for z in range(0,21):
                
                diff=b_new[z]-175
                diff_list.append(diff)
            diff_pos=[]
            diff_neg=[]
            cneg=0
            cpos=0
            for h in range(0,21):
                if diff_list[h]>0:
                    diff_pos.append(diff_list[h])
                    cpos=cpos+1
                if diff_list[h]<0:
                    diff_neg.append(diff_list[h])
                    cneg=cneg+1
            if cpos==0 or cneg==0:
                if cpos==0:
                    ii_list.append(400)
                    
                if cneg==0:
                    ii=0
                    ii_list.append(0)
            else:
                diff_pos_sort=sorted(diff_pos)
                diff_neg_sort=sorted(diff_neg)
                i2=diff_list.index(diff_pos_sort[0])
                i1=diff_list.index(diff_neg_sort[cneg-1])
                x1=i1*20
                x2=i2*20
                y1=b[i1]
                y2=b[i2]
                K=(y2-y1)/(x2-x1)
                B=((y1*x2)-(y2*x1))/(x2-x1)
                #Y=K*X+B
                ii=(175-B)/(K)
                ii_list.append(ii)
   
                        
    return ii_list

'''
singlemodel='SVR'
multimodel='SVR'
basemodel='SVR'
metamodel='MLP'
j=90
'''
def plot_prediction_section3(singlemodel,multimodel,basemodel,metamodel,which,range_number=50,percentage=0):
    j=range_number
    if which=='T1/2':
      #  x=x_data3/100
       # y=half_time
        x_count=np.arange(j)/100
        x_count=x_count.reshape(-1,1) 
        global y_single
        global y_multi
        global y_mtrs
        global pre_y_mtrs
        global yy_mtrs
        y_single=single_section3(singlemodel,'p',j).reshape(1,-1)
        #ezaf
        y_single_csv=pd.DataFrame(data=(y_single))
        y_single_csv.to_csv('y_single_3')
        
        
        pre_y_multi=multi_section3(multimodel,'p',j)
        y_multi=half_timee4(pre_y_multi,j)
        #ezaf
        y_multi_csv=pd.DataFrame(data=(y_multi))
        y_multi_csv.to_csv('y_multi_3')
        
        pre_y_mtrs=meta_model_predict3(j, basemodel, metamodel, 'p_d')
        y_mtrs=half_timee4(pre_y_mtrs,j)
        #ezaf
        y_mtrs_csv=pd.DataFrame(data=(y_mtrs))
        y_mtrs_csv.to_csv('y_mtrs_3')
        #================================================================
        
        pre_y_mtrs_csv=pd.DataFrame(data=(pre_y_mtrs))
        pre_y_mtrs_csv.to_csv('ydrain')
        
        
        yy_mtrs=meta_model_predict3(j,basemodel,metamodel,'p_f')
        yy_mtrs_csv=pd.DataFrame(data=(yy_mtrs))
        yy_mtrs_csv.to_csv('yfoam')
        #================================================================
        
        
        
        plt.figure(figsize=(10, 6))  # Set the figure size (width: 10 inches, height: 6 inches)
        plt.rcParams["figure.dpi"] = 600 
        
        
        
        #plt.scatter(x_count,np.array(y_mtrs).reshape(-1,1),label='MTRS')
        plt.scatter(x_count,np.array(y_multi).reshape(-1,1),label='ST multi outputs',linewidths=0.01)        
        plt.scatter(x_count,y_single.reshape(-1,1),label='Classical Machine learning',linewidths=0.01)
        plt.scatter(x_count,np.array(y_mtrs).reshape(-1,1),label='MTRS',linewidths=0.01)

        x_countt=np.array((0,0.04,0.08,0.16,0.24,0.32,0.40)).reshape(7,1)
        plt.scatter(x_countt,half_time,c='k',label='Experimental')
        plt.xlabel('HAG-MP %') # X-Label
        plt.ylabel('Hlaf Time (min)') # Y-Label
        plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.25))
        plt.ylim(0,300)
        ax=plt.gca()
        ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=False,left=True,right=False)

        plt.show()
        
'''     
file_location = 'C:\\Users\\sunhouse\\ydrain'
f1=open(file_location,'r')
data1=pd.read_csv(f1)
data1=data1.drop('Unnamed: 0',axis=1)
y1=np.array(data1)

file_location = 'C:\\Users\\sunhouse\\yfoam'
f1=open(file_location,'r')
data1=pd.read_csv(f1)
data1=data1.drop('Unnamed: 0',axis=1)
y2=np.array(data1)
'''
      




pp=40
ii=6
   
drain=y1[pp][:]
foam=y2[pp][:]
x = np.array((0,20,40,60,80,100,120,140,160,180,200,
              220,240,260,280,300,320,340,360,380,400))
plt.figure(figsize=(10, 6))  # Set the figure size (width: 10 inches, height: 6 inches)
plt.rcParams["figure.dpi"] = 600 
plt.scatter(x,drain,label='hdrainage')
plt.scatter(x,foam,label='hfoam')
plt.title('Half time') # Title of the plot
plt.xlabel('Time (Min)') # X-Label
plt.ylabel('Volume (cc)') # Y-Label
ax=plt.gca()
ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=False,left=True,right=False)
plt.ylim(0,1000)
plt.legend()
plt.show()


------------------------------------------------

drainn=np.array(y_data3_drain)
foamm=np.array(y_data3_foam)
f=foamm[ii]
d=drainn[ii]

x = np.array((0,20,40,60,80,100,120,140,160,180,200,
              220,240,260,280,300,320,340,360,380,400))
plt.figure(figsize=(10, 6))  # Set the figure size (width: 10 inches, height: 6 inches)
plt.rcParams["figure.dpi"] = 600 
plt.scatter(x,d,label='hdrainage')
plt.scatter(x,f,label='hfoam')
plt.title('Half time') # Title of the plot
plt.xlabel('Time (Min)') # X-Label
plt.ylabel('Volume (cc)') # Y-Label
ax=plt.gca()
ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=False,left=True,right=False)
plt.ylim(0,1000)
plt.legend()
plt.show()










