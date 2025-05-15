"""
NEW TRY AND ERROR
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
#import seaborn
#from numpy import absolute
#from numpy import mean
#from numpy import std
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from  sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
#from sklearn.svm import LinearSVR
from sklearn.multioutput import RegressorChain
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
#from keras.models import Sequential
#from keras.layers import Dense
from sklearn.pipeline import Pipeline
#from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
#from sklearn.model_selection import RepeatedKFold
#from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
#finall_data= pd.concat([data_input, data_outputs], axis=1)
#final_data=finall_data.rename(index={0:'0',1:'4',2:'8',3:'16',4:'24'})
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from mlxtend.regressor import StackingCVRegressor


def load_data():
    #explain about data
    #====================================DATA1===================================

    global x_data1,y_data1
   # global y_data1_scaled
    #x_data1=np.array((0.00,0.04,0.08,0.16,0.24,0.32,0.40)).reshape(7,1)
    x_data1=np.array((0,4,8,16,24,32,40)).reshape(7,1)
    y_data1=np.array([(83.2,10.2,0.122596),(65.8,11.3,0.171733),(49.7,11.9,0.239437),
                    (39.0,12.6,0.323077),(43.2,12.8,0.296296),(47.5,12.5,0.263158),
                    (62.6,11.7,0.186901)])
    
    #scaler.fit(y_data1)
    #y_data1_scaled=scaler.transform(y_data1)
    
    #====================================DATA2===================================
    file2_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\finalcsv.csv'
    f2=open(file2_location,'r')
    global x_data2,data2,data2_0,data2_4,data2_8,data2_16,data2_24,data2_output,data2_outputs,data2_input
   # global data2_outputs_scaled
    data2=pd.read_csv(f2)
    data2.drop(1010,axis=0,inplace=True)
    #x_data3=np.array((0.00,0.04,0.08,0.16,0.24)).reshape(5,1)
    x_data2=np.array((0,4,8,16,24)).reshape(5,1)
    data2_0 = data2['ob'] - data2['0f']
    data2_4=data2['4b']-data2['4f']
    data2_8=data2['8b']-data2['8f']
    data2_16=data2['16b']-data2['16f']
    data2_24=data2['24b']-data2['24f']
    data2_output=pd.concat([data2_0,data2_4,data2_8,data2_16,data2_24],axis=1)
    data2_outputs=data2_output.T
    data2_input=pd.DataFrame(data=[0,4,8,16,24] ,columns=['%HAGP'])   
   # scaler=MinMaxScaler()
    #scaler=Normalizer()
    #scaler=StandardScaler()
   # scaler.fit(data2_outputs)
   # data2_outputs_scaled=scaler.transform(data2_outputs)
    
    #====================================DATA3===================================
    file3_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\section2.csv'
    f3=open(file3_location,'r')
    global x_data3,data3,data3_0,data3_4,data3_8,data3_16,data3_24,data3_32,data3_40,data3_drain,data3_foam
   # global data3_foam_scaled
    data3=pd.read_csv(f3)
    #x_data2=np.array((0.00,0.04,0.08,0.16,0.24,0.32,0.40)).reshape(7,1)
    x_data3=np.array((0,4,8,16,24,32,40)).reshape(7,1)
    data3_0=pd.concat([data3['Unnamed: 0'],data3['Unnamed: 1']],axis=1).reset_index(drop=True)
    data3_4=pd.concat([data3['Unnamed: 2'],data3['Unnamed: 3']],axis=1).reset_index(drop=True)
    data3_8=pd.concat([data3['Unnamed: 4'],data3['Unnamed: 5']],axis=1).reset_index(drop=True)
    data3_16=pd.concat([data3['Unnamed: 6'],data3['Unnamed: 7']],axis=1).reset_index(drop=True)
    data3_24=pd.concat([data3['Unnamed: 8'],data3['Unnamed: 9']],axis=1).reset_index(drop=True)
    data3_32=pd.concat([data3['Unnamed: 10'],data3['Unnamed: 11']],axis=1).reset_index(drop=True)
    data3_40=pd.concat([data3['Unnamed: 12'],data3['Unnamed: 13']],axis=1).reset_index(drop=True)
    data3_drain=pd.concat([data3['Unnamed: 0'],data3['Unnamed: 2'],data3['Unnamed: 4'],
                           data3['Unnamed: 6'],data3['Unnamed: 8'],data3['Unnamed: 10']
                           ,data3['Unnamed: 12']],axis=1).T
    data3_foam=pd.concat([data3['Unnamed: 1'],data3['Unnamed: 3'],data3['Unnamed: 5'],
                           data3['Unnamed: 7'],data3['Unnamed: 9'],data3['Unnamed: 11']
                           ,data3['Unnamed: 13']],axis=1).T

load_data()
def single_select_model(x,y,model,cvv,which):

    if model=='LR':
        poly=PolynomialFeatures()
        regressior=LinearRegression()        
        params=[{'poly__degree':[1,2]}]
        pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model== 'KNN':
        regressior = KNeighborsRegressor()
        params=[{'regressior__n_neighbors':[2,3,4,5,6,7,8,9,10,15,20]}]
        pipe=Pipeline(steps=[('regressior',regressior)]) 
    if model=='RF':
        regressior=RandomForestRegressor(random_state=0)
        params={'regressior__n_estimators':[100,200],
                    'regressior__max_depth':[None,1,2,3,4,5,6]}
        pipe=Pipeline(steps=[('regressior',regressior)])
    if model=='SVR':
        scaler=MinMaxScaler()
       # scaler=None
        poly=PolynomialFeatures()
        regressior = SVR(max_iter=10000)
        params=[{'poly__degree':[1,2,3,4],
                     'regressior__kernel': ['rbf'],
                    'regressior__gamma':[0.0001,0.1,1,10,100],
                    'regressior__C':[0.0001,0.1,1,10,100]},
                    {'poly__degree':[1,2,3,4],
                     'regressior__kernel':['linear'],
                    'regressior__C':[0.0001,0.1,1,10,100]}]       
        pipe=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
    if model=='MLP':   
        regressior=MLPRegressor(solver='lbfgs',activation='tanh',alpha=0.001,max_iter=10000,
                                random_state=40)
        scaler=MinMaxScaler()
        poly=PolynomialFeatures() 
        params={'regressior__hidden_layer_sizes':[[10,10]],
                    'poly__degree':[1,2]}
        pipe=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
        
    kfold1=KFold(n_splits=cvv,shuffle=False)
    if cvv<7:
        kfold2=KFold(n_splits=cvv+1,shuffle=False)
        
    
    scoring_list=['neg_mean_absolute_error','neg_mean_squared_error',
                       'neg_mean_absolute_percentage_error']
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list,
                                    refit='neg_mean_absolute_error',n_jobs=-1)
    if which=='gridsearch':
        
        grid.fit(x,y)
        return grid
    if which=='score':
        cross_score=cross_val_score(grid, x,y,cv=kfold2,scoring='neg_mean_absolute_error',
                                n_jobs=-1)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid, x,y,cv=kfold2,
                                n_jobs=-1)
        return cross_pred

def select_model(x,y,model,cvv,which):

    if model=='LR':
        poly=PolynomialFeatures()
        base_model=LinearRegression()        
        regressior = MultiOutputRegressor(base_model) 
        params=[{'regressor__poly__degree':[1,2]}]
        mini_pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
    if model== 'KNN':
        base_model = KNeighborsRegressor()
        regressior = MultiOutputRegressor(base_model)
        params=[{'regressor__regressior__estimator__n_neighbors':[2,3,4,5,6,7,8,9,10,15,20]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
    if model=='RF':
        base_model=RandomForestRegressor(random_state=0)
        regressior = MultiOutputRegressor(base_model)
        params={'regressor__regressior__estimator__n_estimators':[100,200],
                    'regressor__regressior__estimator__max_depth':[None,1,2,3,4,5,6]}
        mini_pipe=Pipeline(steps=[('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
    if model=='SVR':
        scaler=MinMaxScaler()
       # scaler=None
        poly=PolynomialFeatures()
        base_model = SVR(max_iter=10000)
        regressior = MultiOutputRegressor(base_model)        
        params=[{'regressor__poly__degree':[1,2,3,4],
                     'regressor__regressior__estimator__kernel': ['rbf'],
                    'regressor__regressior__estimator__gamma':[0.0001,0.1,1,10,100],
                    'regressor__regressior__estimator__C':[0.0001,0.1,1,10,100]},
                    {'regressor__poly__degree':[1,2,3,4],
                     'regressor__regressior__estimator__kernel':['linear'],
                    'regressor__regressior__estimator__C':[0.0001,0.1,1,10,100]}]       
        mini_pipe=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
    if model=='MLP':   
        base_model=MLPRegressor(solver='lbfgs',activation='tanh',alpha=0.001,max_iter=10000,
                                random_state=40)
        scaler=MinMaxScaler()
        poly=PolynomialFeatures() 
        regressior = MultiOutputRegressor(base_model)
        params={'regressor__regressior__estimator__hidden_layer_sizes':[[10,10]],
                    'regressor__poly__degree':[1,2]}
        mini_pipe=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
        
    kfold1=KFold(n_splits=cvv,shuffle=False)
    if cvv<7:
        kfold2=KFold(n_splits=cvv+1,shuffle=False)
        
    
    scoring_list=['neg_mean_absolute_error','neg_mean_squared_error',
                       'neg_mean_absolute_percentage_error']
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list,
                                    refit='neg_mean_absolute_error',n_jobs=-1)
    if which=='gridsearch':
        
        grid.fit(x,y)
        return grid
    if which=='score':
        cross_score=cross_val_score(grid, x,y,cv=kfold2,scoring='neg_mean_absolute_error',
                                n_jobs=-1)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid, x,y,cv=kfold2,
                                n_jobs=-1)
        return cross_pred



       
#============================================================================
'                             Section1 : T/D                                '
#============================================================================

def single_section1(model,number):
    x=x_data1
    y=y_data1[:,2].reshape(-1,1)
    grid=single_select_model(x,y,model,7,'gridsearch')
    x_list=[]
    for i in range(0,number):
        x_list.append(i)
    x_list=np.array(x_list).reshape(-1,1)
    y_pred=grid.predict(x_list)
    return y_pred
    

def multi_output_section1(model,number):
    x=x_data1
    y=y_data1
    grid=single_select_model(x,y,model,7,'gridsearch')
    x_list=[]
    for i in range(0,number):
        x_list.append(i)
    x_list=np.array(x_list).reshape(-1,1)
    y_pred=grid.predict(x_list)
    return y_pred
    
    



def base_model(x,y,model):
    global y_pred
    grid=select_model(x,y,model,7,'gridsearch')
    y_pred=grid.predict(x)
    
    return y_pred,grid

def metaa_model(model,meta_model,which):

    pre_x=x_data1
    y=y_data1
    if which=='p':
        x1=x_data1.reshape(-1)
        x1=pd.Series(x1)
        y_pred=base_model(pre_x,y,model)[0]
        x2=pd.DataFrame(y_pred)
        x=pd.concat([x1,x2],axis=1)
        x=np.array(x).reshape(-1,4)
        meta_grid=select_model(x,y,meta_model,7,'gridsearch')
        
        return meta_grid
        #meta_list=[]
        #for i in range(0,j):
        #    meta_list.append(i)
        #meta_pred=grid.predict(meta_list)

    if which=='s':
        print('recently is not ok :)')
        

def base_model_predict (n,model):
    pre_x=x_data1
    y=y_data1
    grid=base_model(pre_x,y,model)[1]
    x_listt=[]
    for i in range(0,n):
        x_listt.append(i)
    x_listt=np.array(x_listt).reshape(-1,1)
    base_predict=grid.predict(x_listt)
    return base_predict


def meta_model_predict(j,model,meta_model):
    n=j
    
    meta_grid=metaa_model(model,meta_model,'p')
    meta_list=[]
    x_list=[]
    for i in range(0,j):
        x_list.append(i)
    x1_list=pd.Series(x_list)
    xy_list=base_model_predict(n,model)
    x2_list=pd.DataFrame(xy_list)
    x_final=pd.concat([x1_list,x2_list],axis=1)
    x_final=np.array(x_final).reshape(-1,4)    
        
        
    meta_pred=meta_grid.predict(x_final)
    return meta_pred
    
 









    
    
def plot_section1(single_model,model,meta_model,range_number):
    j=range_number
    plot_list=meta_model_predict(j,model,meta_model)
    x=x_data1
    y=y_data1
    x_count=np.arange(j)
    x_count=x_count.reshape(-1,1)

    plot_two=plot_list[:,1]/plot_list[:,0]
    yyy=single_section1(single_model,j).reshape(1,-1)
    y_mo=multi_output_section1(single_model, j)
    y_mo_two=y_mo[:,1]/y_mo[:,0]
    
    plt.scatter(x_count,plot_two)
    #plt.scatter(x_count,plot_list[:,2])
    plt.scatter(x_count,yyy)
    plt.scatter(x_count,y_mo[:,2])
    #dota balae shabihe hame
    plt.scatter(x_count,y_mo_two)
    
    
    
    plt.scatter(x_data1,y_data1[:,2].reshape(-1,1),c='k')
    plt.xlabel('GPA %') # X-Label
    plt.ylabel('T/D') # Y-Label
    plt.show()
plot_section1('LR','SVR','LR',50)

#avalk abi bad narenji bad asabz bad ghermez bad banafsj


    
def score_section1():
    

#============================================================================
'                             Section2 : P                                 '
#============================================================================

def prediction_section2():
    
    
def score_section2():
    
    
def plot_section2():
    
    
#============================================================================
'                             Section3 : T 1/2                                '
#============================================================================

def prediction_section3():
    
    
def score_section3():
    
    
def plot_section3():


#aval train roo alll















