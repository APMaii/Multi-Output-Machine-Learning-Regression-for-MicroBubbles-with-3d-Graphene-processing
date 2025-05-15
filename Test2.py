"""
FINAL_for_me_without description
"""
'''
FINAL NOTE :distribution change distplot to histplot ( because of new future version) in

#after that we can talk about the changing scale in outputs

'''
#=============================================================================
'                              Import                                        '
#=============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
#from numpy import absolute
#from numpy import mean
#from numpy import std
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from  sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import RobustScaler
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import Normalizer
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
#from sklearn.model_selection import cross_val_predict
#finall_data= pd.concat([data_input, data_outputs], axis=1)
#final_data=finall_data.rename(index={0:'0',1:'4',2:'8',3:'16',4:'24'})


#============================================================================
'                              Loading data                                 '
#============================================================================
#bayad data haro mostaghim vared konim ya bznim on bala too description ke 
def load_data():
    #explain about data
    #====================================DATA2===================================

    global x_data1,y_data1,y_data1_scaled
    x_data1=np.array((0.00,0.04,0.08,0.16,0.24,0.32,0.40)).reshape(7,1)
    y_data1=np.array([(83.2,10.2,0.122596),(65.8,11.3,0.171733),(49.7,11.9,0.239437),
                    (39.0,12.6,0.323077),(43.2,12.8,0.296296),(47.5,12.5,0.263158),
                    (62.6,11.7,0.186901)])
    scaler=MinMaxScaler()
    scaler.fit(y_data1)
    y_data1_scaled=scaler.transform(y_data1)
    #====================================DATA2===================================
    file2_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\section2.csv'
    f2=open(file2_location,'r')
    global data2,data2_0,data2_4,data2_8,data2_16,data2_24,data2_32,data2_40,data2_drain,data2_foam
    global data2_foam_scaled
    data2=pd.read_csv(f2)
    x_data2=np.array((0.00,0.04,0.08,0.16,0.24,0.32,0.40)).reshape(7,1)
    data2_0=pd.concat([data2['Unnamed: 0'],data2['Unnamed: 1']],axis=1).reset_index(drop=True)
    data2_4=pd.concat([data2['Unnamed: 2'],data2['Unnamed: 3']],axis=1).reset_index(drop=True)
    data2_8=pd.concat([data2['Unnamed: 4'],data2['Unnamed: 5']],axis=1).reset_index(drop=True)
    data2_16=pd.concat([data2['Unnamed: 6'],data2['Unnamed: 7']],axis=1).reset_index(drop=True)
    data2_24=pd.concat([data2['Unnamed: 8'],data2['Unnamed: 9']],axis=1).reset_index(drop=True)
    data2_32=pd.concat([data2['Unnamed: 10'],data2['Unnamed: 11']],axis=1).reset_index(drop=True)
    data2_40=pd.concat([data2['Unnamed: 12'],data2['Unnamed: 13']],axis=1).reset_index(drop=True)
    data2_drain=pd.concat([data2['Unnamed: 0'],data2['Unnamed: 2'],data2['Unnamed: 4'],
                           data2['Unnamed: 6'],data2['Unnamed: 8'],data2['Unnamed: 10']
                           ,data2['Unnamed: 12']],axis=1).T
    data2_foam=pd.concat([data2['Unnamed: 1'],data2['Unnamed: 3'],data2['Unnamed: 5'],
                           data2['Unnamed: 7'],data2['Unnamed: 9'],data2['Unnamed: 11']
                           ,data2['Unnamed: 13']],axis=1).T
    scaler=MinMaxScaler()
    scaler.fit(data2_foam)
    data2_foam_scaled=scaler.transform(data2_foam)
    
    #====================================DATA3===================================
    file3_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\finalcsv.csv'
    f3=open(file3_location,'r')
    global data3,data3_0,data3_4,data3_8,data3_16,data3_24,data3_output,data3_outputs,data3_input
    global data3_outputs_scaled
    data3=pd.read_csv(f3)
    data3.drop(1010,axis=0,inplace=True)
    x_data3=np.array((0.00,0.04,0.08,0.16,0.24)).reshape(5,1)
    data3_0 = data3['ob'] - data3['0f']
    data3_4=data3['4b']-data3['4f']
    data3_8=data3['8b']-data3['8f']
    data3_16=data3['16b']-data3['16f']
    data3_24=data3['24b']-data3['24f']
    data3_output=pd.concat([data3_0,data3_4,data3_8,data3_16,data3_24],axis=1)
    data3_outputs=data3_output.T
    data3_input=pd.DataFrame(data=[0,4,8,16,24] ,columns=['%HAGP'])   
    scaler=MinMaxScaler()
    scaler.fit(data3_outputs)
    data3_outputs_scaled=scaler.transform(data3_outputs)
    
load_data()
#============================================================================
'                              DATA distribution                            '
#============================================================================
def distribution(data,scale):
    if data=='1':
        if scale=='on':
            for i in range(0,3):
                output = y_data1_scaled[:,i]
                sns.distplot(output, label={'Output {}'.format(i+1)})
        if scale=='off':
            for i in range(0,3):
                output = y_data1[:,i]
                sns.distplot(output, label={'Output {}'.format(i+1)})
            
    if data=='2':
        if scale=='on':
            np_data2=np.array(data2_foam_scaled)
            for i in range(0,3):
                output=np_data2[:,i]
                sns.distplot(output)
        if scale=='off':
            np_data2=np.array(data2_foam)
            for i in range(0,3):
                output=np_data2[:,i]
                sns.distplot(output)
        
    if data=='3':
        if scale=='on':
            data33=np.array(data3_outputs_scaled)
            for i in range(0,1000):
                output = data33[:,i]
                sns.distplot(output)
        if scale=='off':
            data33=np.array(data3_outputs)
            for i in range(0,1000):
                output = data33[:,i]
                sns.distplot(output)

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution of Outputs')
    
    if data=='1':
        plt.legend()

    plt.show()
distribution('3','on')
#============================================================================
'                             Sect_model                           '
#============================================================================      
def select_model(model,chain,n_m):
    if model=='LR':
        poly=PolynomialFeatures()
        if chain=='on':
            mini_model= LinearRegression()
            regressior = RegressorChain(mini_model)
        if chain=='off':
            regressior = LinearRegression()
        params=[{'poly__degree':[1,2]}]
        pipe1=Pipeline(steps=[('poly',poly),('regressior',regressior)])
        pipe2=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model== 'KNN':
        if chain=='on':
            mini_model=KNeighborsRegressor()
            regressior = RegressorChain(mini_model)
            params=[{'regressior__base_estimator__n_neighbors':[2,3,4,5,6,7,8,9,10,15,20]}]
        if chain=='off':
            regressior = KNeighborsRegressor()
            params=[{'regressior__n_neighbors':[2,3,4,5,6,7,8,9,10,15,20]}]
        pipe1=Pipeline(steps=[('regressior',regressior)]) 
        pipe2=Pipeline(steps=[('regressior',regressior)]) 
    if model== 'DT':
        if chain=='on':
            mini_model=DecisionTreeRegressor()
            regressior = RegressorChain(mini_model)
            params={'regressior__base_estimator__max_depth':[1,2,3,4,5,6,7,8,9,10],
            'regressior__base_estimator__max_leaf_nodes':[2,3,4,5,6,7,8,10]}       
        if chain=='off':
            regressior = DecisionTreeRegressor()
            params={'regressior__max_depth':[1,2,3,4,5,6,7,8,9,10],
            'regressior__max_leaf_nodes':[2,3,4,5,6,7,8,10]}
        pipe1=Pipeline(steps=[('regressior',regressior)])     
        pipe2=Pipeline(steps=[('regressior',regressior)])     
    if model=='RF':
        if chain=='on':
            mini_model=RandomForestRegressor(random_state=0)
            regressior = RegressorChain(mini_model)
            params={'regressior__base_estimator__n_estimators':[100,200],
                    'regressior__base_estimator__max_depth':[None,1,2,3,4,5,6]}
        if chain=='off':
            regressior=RandomForestRegressor(random_state=0)
            params={'regressior__n_estimators':[100,200],
                    'regressior__max_depth':[None,1,2,3,4,5,6]}
        pipe1=Pipeline(steps=[('regressior',regressior)])
        pipe2=Pipeline(steps=[('regressior',regressior)])
    if model=='SVR':
        scaler=MinMaxScaler()
        poly=PolynomialFeatures()
        mini_model = SVR(max_iter=10000)
        if chain=='on':
            regressior = RegressorChain(mini_model)
            params=[{'poly__degree':[1,2,3,4],
                     'regressior__base_estimator__kernel': ['rbf'],
                    'regressior__base_estimator__gamma':[0.0001,0.1,1,10,100],
                    'regressior__base_estimator__C':[0.0001,0.1,1,10,100]},
                    {'poly__degree':[1,2,3,4],
                     'regressior__base_estimator__kernel':['linear'],
                    'regressior__base_estimator__C':[0.0001,0.1,1,10,100]}]
        if chain=='off':
            params=[{'poly__degree':[1,2,3,4],
                     'regressior__estimator__kernel': ['rbf'],
                    'regressior__estimator__gamma':[0.0001,0.1,1,10,100],
                    'regressior__estimator__C':[0.0001,0.1,1,10,100]},
                    {'poly__degree':[1,2,3,4],
                     'regressior__estimator__kernel':['linear'],
                    'regressior__estimator__C':[0.0001,0.1,1,10,100]}]       
            regressior = MultiOutputRegressor(mini_model)   
        pipe1=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
        pipe2=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
    if model=='ANN':   
        mini_model=MLPRegressor(solver='lbfgs',activation='tanh',alpha=0.001,max_iter=10000,
                                random_state=40)
        scaler=MinMaxScaler()
        poly=PolynomialFeatures() 
        if chain=='on':
            regressior = RegressorChain(mini_model)
            params={'regressior__base_estimator__hidden_layer_sizes':[[10,10],[100,100],
                                                                 [100,200]],
                    'poly__degree':[1,2,3,4]}  
        if chain=='off':
            regressior = MultiOutputRegressor(mini_model)
            params={'regressior__estimator__hidden_layer_sizes':[[10,10],[100,100],
                                                                 [100,200]],
                    'poly__degree':[1,2,3,4]}
        pipe1=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
        pipe2=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
    kfold=KFold(n_splits=5,shuffle=False)
    scoring_list=['neg_mean_absolute_error','neg_mean_squared_error',
                      'neg_mean_absolute_percentage_error']
    pipe_gc1 = GridSearchCV(pipe1,
                                   param_grid=params,
                                   cv=kfold,scoring=scoring_list,
                                   refit='neg_mean_absolute_error',n_jobs=-1)
    kfold2=KFold(n_splits=7,shuffle=True,random_state=42)
    pipe_gc2 = GridSearchCV(pipe2,
                                   param_grid=params,
                                   cv=kfold2,scoring=scoring_list,
                                   refit='neg_mean_absolute_error',n_jobs=-1)
    if n_m == '1':
        return pipe_gc1
    if n_m == '2':
        return pipe_gc2
    else:
        return pipe_gc1
#============================================================================
'                             Section1 : T/D                                '
#============================================================================
def predictor1(percent,model1,model2):
    x=np.array((0,4,8,16,24)).reshape(-1,1)
    #y1=pd.concat((data0_1,data4_1,data8_1,data16_1,data24_1),axis=1).T
    y2=pd.concat((data0_2,data4_2,data8_2,data16_2,data24_2),axis=1).T     
    pipe_gc1=model_select(model1, 'on','1') 
    pipe_gc1.fit(x, y2)
    row=[percent]
    yhat2 = pipe_gc1.predict([row]).reshape(-1,1)
    x1=np.array(pd.concat((data0_1,data4_1,data8_1,data16_1,data24_1),axis=0).reset_index(drop=True)).reshape(-1,1)
    x2=np.array(pd.concat((data0_2,data4_2,data8_2,data16_2,data24_2),axis=0).reset_index(drop=True)).reshape(-1,1)
    pipe_gc2=model_select(model2, 'off','2')
    pipe_gc2.fit(x2,x1)
    y_final=pipe_gc2.predict(yhat2).reshape(-1,1)
    summ=y_final+yhat2
    xx1=y_final/summ
    diff=list(abs(xx1- 0.5) )
    minn=min(diff)
    ii=diff.index(minn)
    c=ii*20
    print('our final T 1/2 ',c)
    return c
def score1():       
def plot1(n,model1,model2):
    for i in range(0,n+1):
        if i==0:
            liss=[]
        pre=predictor(i,model1,model2)
        liss.append(pre)
        if i==n:
            axx=np.arange(n+1).reshape(-1,1)
            lisst=np.array(liss).reshape(-1,1)
            axx_real=np.array((0,4,8,16,24)).reshape(-1,1)
            lisst_real=np.array((60,87.5,200,280,260)).reshape(-1,1)
            plt.scatter(axx,lisst)
            plt.scatter(axx_real,lisst_real,c='r')
            plt.xlabel('GPA %') # X-Label
            plt.ylabel('T/12') # Y-Label
            plt.show()    
plot1(40,'LR','LR')
#============================================================================
'                              Section2: T 1/2                              '
#============================================================================
def predictor2(percent,model1,model2):
    x=np.array((0,4,8,16,24)).reshape(-1,1)
    #y1=pd.concat((data0_1,data4_1,data8_1,data16_1,data24_1),axis=1).T
    y2=pd.concat((data0_2,data4_2,data8_2,data16_2,data24_2),axis=1).T     
    pipe_gc1=model_select(model1, 'on','1') 
    pipe_gc1.fit(x, y2)
    row=[percent]
    yhat2 = pipe_gc1.predict([row]).reshape(-1,1)
    x1=np.array(pd.concat((data0_1,data4_1,data8_1,data16_1,data24_1),axis=0).reset_index(drop=True)).reshape(-1,1)
    x2=np.array(pd.concat((data0_2,data4_2,data8_2,data16_2,data24_2),axis=0).reset_index(drop=True)).reshape(-1,1)
    pipe_gc2=model_select(model2, 'off','2')
    pipe_gc2.fit(x2,x1)
    y_final=pipe_gc2.predict(yhat2).reshape(-1,1)
    summ=y_final+yhat2
    xx1=y_final/summ
    diff=list(abs(xx1- 0.5) )
    minn=min(diff)
    ii=diff.index(minn)
    c=ii*20
    print('our final T 1/2 ',c)
    return c
def score2():
def plot2(n,model1,model2):
    for i in range(0,n+1):
        if i==0:
            liss=[]
        pre=predictor(i,model1,model2)
        liss.append(pre)
        if i==n:
            axx=np.arange(n+1).reshape(-1,1)
            lisst=np.array(liss).reshape(-1,1)
            axx_real=np.array((0,4,8,16,24)).reshape(-1,1)
            lisst_real=np.array((60,87.5,200,280,260)).reshape(-1,1)
            plt.scatter(axx,lisst)
            plt.scatter(axx_real,lisst_real,c='r')
            plt.xlabel('GPA %') # X-Label
            plt.ylabel('T/12') # Y-Label
            plt.show()    
plot2(40,'LR','LR')
#============================================================================
'                        Section3: DENSITY/PRESSURE                         '
#============================================================================
    if model=='LR':
        poly=PolynomialFeatures()
        if chain=='on':
            mini_model= LinearRegression()
            regressior = RegressorChain(mini_model)
        if chain=='off':
            regressior = LinearRegression()
        params=[{'poly__degree':[1,2]}]
        pipe1=Pipeline(steps=[('poly',poly),('regressior',regressior)])
        pipe2=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model== 'KNN':
        if chain=='on':
            mini_model=KNeighborsRegressor()
            regressior = RegressorChain(mini_model)
            params=[{'regressior__base_estimator__n_neighbors':[2,3,4,5,6,7,8,9,10,15,20]}]
        if chain=='off':
            regressior = KNeighborsRegressor()
            params=[{'regressior__n_neighbors':[2,3,4,5,6,7,8,9,10,15,20]}]
        pipe1=Pipeline(steps=[('regressior',regressior)]) 
        pipe2=Pipeline(steps=[('regressior',regressior)]) 
    if model== 'DT':
        if chain=='on':
            mini_model=DecisionTreeRegressor()
            regressior = RegressorChain(mini_model)
            params={'regressior__base_estimator__max_depth':[1,2,3,4,5,6,7,8,9,10],
            'regressior__base_estimator__max_leaf_nodes':[2,3,4,5,6,7,8,10]}       
        if chain=='off':
            regressior = DecisionTreeRegressor()
            params={'regressior__max_depth':[1,2,3,4,5,6,7,8,9,10],
            'regressior__max_leaf_nodes':[2,3,4,5,6,7,8,10]}
        pipe1=Pipeline(steps=[('regressior',regressior)])     
        pipe2=Pipeline(steps=[('regressior',regressior)])     
    if model=='RF':
        if chain=='on':
            mini_model=RandomForestRegressor(random_state=0)
            regressior = RegressorChain(mini_model)
            params={'regressior__base_estimator__n_estimators':[100,200],
                    'regressior__base_estimator__max_depth':[None,1,2,3,4,5,6]}
        if chain=='off':
            regressior=RandomForestRegressor(random_state=0)
            params={'regressior__n_estimators':[100,200],
                    'regressior__max_depth':[None,1,2,3,4,5,6]}
        pipe1=Pipeline(steps=[('regressior',regressior)])
        pipe2=Pipeline(steps=[('regressior',regressior)])
    if model=='SVR':
        scaler=MinMaxScaler()
        poly=PolynomialFeatures()
        mini_model = SVR(max_iter=10000)
        if chain=='on':
            regressior = RegressorChain(mini_model)
            params=[{'poly__degree':[1,2,3,4],
                     'regressior__base_estimator__kernel': ['rbf'],
                    'regressior__base_estimator__gamma':[0.0001,0.1,1,10,100],
                    'regressior__base_estimator__C':[0.0001,0.1,1,10,100]},
                    {'poly__degree':[1,2,3,4],
                     'regressior__base_estimator__kernel':['linear'],
                    'regressior__base_estimator__C':[0.0001,0.1,1,10,100]}]
        if chain=='off':
            params=[{'poly__degree':[1,2,3,4],
                     'regressior__estimator__kernel': ['rbf'],
                    'regressior__estimator__gamma':[0.0001,0.1,1,10,100],
                    'regressior__estimator__C':[0.0001,0.1,1,10,100]},
                    {'poly__degree':[1,2,3,4],
                     'regressior__estimator__kernel':['linear'],
                    'regressior__estimator__C':[0.0001,0.1,1,10,100]}]       
            regressior = MultiOutputRegressor(mini_model)   
        pipe1=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
        pipe2=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
    if model=='ANN':   
        mini_model=MLPRegressor(solver='lbfgs',activation='tanh',alpha=0.001,max_iter=10000,
                                random_state=40)
        scaler=MinMaxScaler()
        poly=PolynomialFeatures() 
        if chain=='on':
            regressior = RegressorChain(mini_model)
            params={'regressior__base_estimator__hidden_layer_sizes':[[10,10],[100,100],
                                                                 [100,200]],
                    'poly__degree':[1,2,3,4]}  
        if chain=='off':
            regressior = MultiOutputRegressor(mini_model)
            params={'regressior__estimator__hidden_layer_sizes':[[10,10],[100,100],
                                                                 [100,200]],
                    'poly__degree':[1,2,3,4]}
        pipe1=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
        pipe2=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
    kfold=KFold(n_splits=5,shuffle=False)
    scoring_list=['neg_mean_absolute_error','neg_mean_squared_error',
                      'neg_mean_absolute_percentage_error']
    pipe_gc1 = GridSearchCV(pipe1,
                                   param_grid=params,
                                   cv=kfold,scoring=scoring_list,
                                   refit='neg_mean_absolute_error',n_jobs=-1)
    kfold2=KFold(n_splits=7,shuffle=True,random_state=42)
    pipe_gc2 = GridSearchCV(pipe2,
                                   param_grid=params,
                                   cv=kfold2,scoring=scoring_list,
                                   refit='neg_mean_absolute_error',n_jobs=-1)
    if n_m == '1':
        return pipe_gc1
    if n_m == '2':
        return pipe_gc2
    else:
        return pipe_gc
def predictor3(percent,model1,model2):
    x=np.array((0,4,8,16,24)).reshape(-1,1)
    #y1=pd.concat((data0_1,data4_1,data8_1,data16_1,data24_1),axis=1).T
    y2=pd.concat((data0_2,data4_2,data8_2,data16_2,data24_2),axis=1).T     
    pipe_gc1=model_select(model1, 'on','1') 
    pipe_gc1.fit(x, y2)
    row=[percent]
    yhat2 = pipe_gc1.predict([row]).reshape(-1,1)
    x1=np.array(pd.concat((data0_1,data4_1,data8_1,data16_1,data24_1),axis=0).reset_index(drop=True)).reshape(-1,1)
    x2=np.array(pd.concat((data0_2,data4_2,data8_2,data16_2,data24_2),axis=0).reset_index(drop=True)).reshape(-1,1)
    pipe_gc2=model_select(model2, 'off','2')
    pipe_gc2.fit(x2,x1)
    y_final=pipe_gc2.predict(yhat2).reshape(-1,1)
    summ=y_final+yhat2
    xx1=y_final/summ
    diff=list(abs(xx1- 0.5) )
    minn=min(diff)
    ii=diff.index(minn)
    c=ii*20
    print('our final T 1/2 ',c)
    return c
def score3():
def plot3(n,model1,model2):
    for i in range(0,n+1):
        if i==0:
            liss=[]
        pre=predictor(i,model1,model2)
        liss.append(pre)
        if i==n:
            axx=np.arange(n+1).reshape(-1,1)
            lisst=np.array(liss).reshape(-1,1)
            axx_real=np.array((0,4,8,16,24)).reshape(-1,1)
            lisst_real=np.array((60,87.5,200,280,260)).reshape(-1,1)
            plt.scatter(axx,lisst)
            plt.scatter(axx_real,lisst_real,c='r')
            plt.xlabel('GPA %') # X-Label
            plt.ylabel('T/12') # Y-Label
            plt.show()   
plot2(40,'LR','LR')
#============================================================================
'                              Model Evaluation                             '
#============================================================================
def cross_val():
def final_evaluation():
    labels = ['Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5','label6']
    data1 = [0.97,0.93,0.80,0.6,0.70,0.4]
    data2=[0.8,0.85,0.33,0.95,0.8,0.3]
    data3=[0.4,0.33,0.23,0.1,0.8,0.2]
    data4=[0.95,0.97,1,1,0.89,0.91]
    data5=[0.7,0.75,0.73,0.64,0.8,0.84]
    data6=[0.56,0.78,0.98,0.43,0.1,0.2]
    #data3=
    # Calculate angles for each label
    #angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles=['c1','c2','c3','c4','c5','c6']
    # Close the plot
    data1 = np.concatenate((data1,[data1[0]]))
    data2 = np.concatenate((data2,[data2[0]]))
    data3 = np.concatenate((data3,[data3[0]]))
    data4 = np.concatenate((data4,[data4[0]]))
    data5 = np.concatenate((data5,[data5[0]]))
    data6 = np.concatenate((data6,[data6[0]]))
    angles = np.concatenate((angles,[angles[0]]))
    # Initialize the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    # Plot the data
    ax.plot()
    ax.plot(angles, data3, 'o-', linewidth=1,markersize=3,label='ANN')
    ax.plot(angles, data4, 'o-', linewidth=1,markersize=3,label='SVR')
    ax.plot(angles, data5, 'o-', linewidth=1,markersize=3,label='LR')
    ax.plot(angles, data6, 'o-', linewidth=1,markersize=3,label='RF')
    ax.plot(angles, data2, 'o-', linewidth=1,markersize=3,label='KN')
    ax.plot(angles, data1, 'o-', linewidth=1,markersize=3,label='OH')
    # Fill the area
    ax.fill(angles, data1, alpha=0)
    ax.fill(angles, data2, alpha=0)
    ax.fill(angles, data3, alpha=0)
    ax.fill(angles, data4, alpha=0)
    ax.fill(angles, data5, alpha=0)
    ax.fill(angles, data6, alpha=0)
    # Add labels and ticks
    #ax.set_thetagrids(angles * 180/np.pi, labels)
    #ax.set_thetagrids(angles, labels)
    ax.set_title('Radar Plot')
    ax.legend()
    plt.show()
    
