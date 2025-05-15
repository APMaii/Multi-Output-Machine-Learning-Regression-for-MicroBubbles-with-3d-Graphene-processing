"""
FINAL 9 MAY ( after that we have supplementory)
"""

'''
note:
    description ydmon nre
    hamon distplot bmone
    correlation 3 ro ok konim
    # ino hesab konim chghd frgh mikone agar biaym transformer bzarim single ro
    mohem trin hyperparameter haro dr biarim

'''

#=============================================================================
'                              Import                                        '
#=============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
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
from mlxtend.regressor import StackingCVRegressor


#============================================================================
'                              Loading data                                 '
#============================================================================


def load_data():
    #====================================DATA1===================================

    global x_data1,y_data1
    x_data1=np.array((0,4,8,16,24,32,40)).reshape(7,1)
    y_data1=np.array([(83.2,10.2,0.122596),(65.8,11.3,0.171733),(49.7,11.9,0.239437),
                    (39.0,12.6,0.323077),(43.2,12.8,0.296296),(47.5,12.5,0.263158),
                    (62.6,11.7,0.186901)])

    #====================================DATA2===================================
    file2_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\finalcsv.csv'
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
    file3_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\section2.csv'
    f3=open(file3_location,'r')
    global x_data3,data3,data3_0,data3_4,data3_8,data3_16,data3_24,data3_32,data3_40,y_data3_drain,y_data3_foam
    
    data3=pd.read_csv(f3)
    x_data3=np.array((0,4,8,16,24,32,40)).reshape(7,1)
    data3_0=pd.concat([data3['Unnamed: 1'],data3['Unnamed: 0']],axis=0).reset_index(drop=True)
    data3_4=pd.concat([data3['Unnamed: 3'],data3['Unnamed: 2']],axis=0).reset_index(drop=True)
    data3_8=pd.concat([data3['Unnamed: 5'],data3['Unnamed: 4']],axis=0).reset_index(drop=True)
    data3_16=pd.concat([data3['Unnamed: 7'],data3['Unnamed: 6']],axis=0).reset_index(drop=True)
    data3_24=pd.concat([data3['Unnamed: 9'],data3['Unnamed: 8']],axis=0).reset_index(drop=True)
    data3_32=pd.concat([data3['Unnamed: 11'],data3['Unnamed: 10']],axis=0).reset_index(drop=True)
    data3_40=pd.concat([data3['Unnamed: 13'],data3['Unnamed: 12']],axis=0).reset_index(drop=True)
    #y_data3=pd.concat([data3_0,data3_4,data3_8,data3_16,data3_24,data3_32,data3_40],axis=1).T
    y_data3_foam=pd.concat([data3['Unnamed: 1'],data3['Unnamed: 3'],data3['Unnamed: 5'],
                       data3['Unnamed: 7'],data3['Unnamed: 9'],data3['Unnamed: 11'],
                       data3['Unnamed: 13']],axis=1).T.reset_index(drop=True)
    y_data3_drain=pd.concat([data3['Unnamed: 0'],data3['Unnamed: 2'],data3['Unnamed: 4'],
                       data3['Unnamed: 6'],data3['Unnamed: 8'],data3['Unnamed: 10'],
                       data3['Unnamed: 12']],axis=1).T.reset_index(drop=True)

load_data()

#============================================================================
'                              DATA distribution                            '
#============================================================================

def distribution(data,scale,scaler):
    
    if scaler=='standard':
        scaler=StandardScaler()        
    if scaler=='minmax':
        scaler=MinMaxScaler()        
    if scaler=='power':
        scaler=PowerTransformer()
    if scaler=='normalize': 
        scaler=Normalizer()
        
        
    global data3_foam_scaled,y_data1_scaled    
    if data=='1':
        if scale=='on':
            for i in range(0,3):
                scaler.fit(y_data1)
                y_data1_scaled=scaler.transform(y_data1)
                output = y_data1_scaled[:,i]
                sns.distplot(output, label={'Output {}'.format(i+1)})
        if scale=='off':
            for i in range(0,3):
                output = y_data1[:,i]
                sns.distplot(output, label={'Output {}'.format(i+1)})
    if data=='2':
        if scale=='on':
            scaler.fit(data2_outputs)
            data2_outputs_scaled=scaler.transform(data2_outputs)
            data22=np.array(data2_outputs_scaled)
            for i in range(0,1010):
                output = data22[:,i]
                sns.distplot(output)
        if scale=='off':
            data22=np.array(data2_outputs)
            for i in range(0,1010):
                output = data22[:,i]
                sns.distplot(output)
    if data=='3_f':
        if scale=='on':
            scaler.fit(y_data3_foam)
            y_data3_foam_scaled=scaler.transform(y_data3_foam)
            np_data3=np.array(y_data3_foam_scaled)
            for i in range(0,21):
                output=np_data3[:,i]
                sns.distplot(output)
        if scale=='off':
            np_data3=np.array(y_data3_foam)
            for i in range(0,21):
                output=np_data3[:,i]
                sns.distplot(output)   
    if data=='3_d':
        if scale=='on':
            scaler.fit(y_data3_drain)
            y_data3_drain_scaled=scaler.transform(y_data3_drain)
            np_data3=np.array(y_data3_drain_scaled)
            for i in range(0,21):
                output=np_data3[:,i]
                sns.distplot(output)
        if scale=='off':
            np_data3=np.array(y_data3_drain)
            for i in range(0,21):
                output=np_data3[:,i]
                sns.distplot(output)

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution of Outputs')
    
    if data=='1':
        plt.legend()

    plt.show()


#============================================================================
'                              DATA correlation                           '
#============================================================================

def correlation(data):
    if data=='1':
        dataa=np.concatenate((x_data1,y_data1),axis=1)
        xt=['input','outpu1','output2','output3']
        yt=['input','outpu1','output2','output3']
        datacorr=pd.DataFrame(dataa)
    if data=='2':
        xt='auto'
        yt='auto'
        dataa=pd.concat((data2_input,data2_outputs),axis=1)
        datacorr=pd.DataFrame(dataa)
    if data=='3_f':
        xt='auto'
        yt='auto'
        dataa=pd.concat((pd.DataFrame(x_data3),y_data3_foam),axis=1)
        datacorr=pd.DataFrame(dataa)
    if data=='3_d':
        xt='auto'
        yt='auto'
        dataa=pd.concat((pd.DataFrame(x_data3),y_data3_drain),axis=1)
        datacorr=pd.DataFrame(dataa)
    
    plt.figure()
    correlation = datacorr.corr()  
    sns.heatmap(correlation,cmap="coolwarm",xticklabels=xt,
                yticklabels=yt)


#============================================================================
'                             Sect_model                           '
#============================================================================

def single_select_model(x,y,model,cvv,which):
    if model=='LR':
        poly=PolynomialFeatures()
        regressior=LinearRegression()        
        params=[{'poly__degree':[1,2,3]}]
        pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model== 'KNN':
        regressior = KNeighborsRegressor()
        params=[{'regressior__n_neighbors':[1,3,5,7,9,11,13,15]}]
        pipe=Pipeline(steps=[('regressior',regressior)]) 
    if model== 'DT':
        regressior=DecisionTreeRegressor(random_state=0)
        params={'regressior__max_depth':[1,2,3,4,5,6,7,8,9,10],
         'regressior__min_samples_split':[2,3,4,5,6,7,8,10]}      
        pipe=Pipeline(steps=[('regressior',regressior)]) 
    if model=='RF':
        regressior=RandomForestRegressor(random_state=0)
        params={'regressior__n_estimators':[2,3,4,5,6,7,8,9,10],
                    'regressior__max_depth':[1,2,3,4,8,10],
                    'regressior__min_samples_split':[2,3,4,5,8,10]}
        pipe=Pipeline(steps=[('regressior',regressior)])
    if model=='SVR':
        scaler=MinMaxScaler()
        regressior = SVR(max_iter=10000)
        params=[{ 'regressior__kernel': ['rbf'],
                    'regressior__gamma':[0.0001,0.1,1,10],
                    'regressior__C':[0.0001,0.1,1,10]},
                    { 'regressior__kernel':['linear'],
                    'regressior__C':[0.0001,0.1,1,10]},
                    {'regressior__kernel': ['poly'],
                    'regressior__C':[0.0001,0.1,1,10],
                    'regressior__degree':[1,2,3]}]       
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
    if model=='MLP':   
        regressior=MLPRegressor(solver='adam',max_iter=10000,random_state=40)
        scaler=MinMaxScaler()
        params={'regressior__hidden_layer_sizes':[(2,),(3,),(5,),(10,),(5,5),(10,10)],
                    'regressior__activation':['logistic', 'tanh', 'relu'],
                    'regressior__alpha':[0.0001,0.001,0.01,0.1]}
        pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        
        
    kfold1=KFold(n_splits=cvv,shuffle=False)
    kfold2=KFold(n_splits=cvv+1,shuffle=False)
        
    
    scoring_list=['neg_root_mean_squared_error','neg_mean_absolute_error']
    
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list,
                                    refit='neg_mean_absolute_error',n_jobs=-1)
    if which=='gridsearch':
        
        grid.fit(x,y)
        return grid
    if which=='score':
        cross_score=cross_val_score(grid, x,y,cv=kfold2,scoring='neg_mean_absolute_percentage_error',
                                n_jobs=-1)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid, x,y,cv=kfold2,
                                n_jobs=-1)
        return cross_pred

def select_model(x,y,model,cvv,which,sc='neg_mean_absolute_error'):

    if model=='LR':
        poly=PolynomialFeatures()
        base_model=LinearRegression()        
        regressior = MultiOutputRegressor(base_model) 
        params=[{'regressor__poly__degree':[1,2,3]}]
        mini_pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
    if model== 'KNN':
        base_model = KNeighborsRegressor()
        regressior = MultiOutputRegressor(base_model)
        params=[{'regressor__regressior__estimator__n_neighbors':[1,3,5,7,9,11,13,15]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
    if model== 'DT':
        base_model=DecisionTreeRegressor(random_state=0)
        regressior = MultiOutputRegressor(base_model)
        params={'regressor__regressior__estimator__max_depth':[1,2,3,4,5,6,7,8,9,10],
            'regressor__regressior__estimator__min_samples_split':[2,3,4,5,6,7,8,10]}      
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
    if model=='RF':
        base_model=RandomForestRegressor(random_state=0)
        regressior = MultiOutputRegressor(base_model)
        params={'regressor__regressior__estimator__n_estimators':[2,3,4,5,6,7,8,9,10],
                    'regressor__regressior__estimator__max_depth':[1,2,3,4,8,10],
                    'regressor__regressior__estimator__min_samples_split':[2,3,4,5,8,10]}
        mini_pipe=Pipeline(steps=[('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
    if model=='SVR':
        scaler=MinMaxScaler()
        base_model = SVR(max_iter=10000)
        regressior = MultiOutputRegressor(base_model)        
        params=[{'regressor__regressior__estimator__kernel': ['rbf'],
                    'regressor__regressior__estimator__gamma':[0.0001,0.1,1,10],
                    'regressor__regressior__estimator__C':[0.0001,0.1,1,10]},
                    {'regressor__regressior__estimator__kernel':['linear'],
                    'regressor__regressior__estimator__C':[0.0001,0.1,1,10]},
                    {'regressor__regressior__estimator__kernel':['poly'],
                     'regressor__regressior__estimator__C':[0.0001,0.1,1,10],
                     'regressor__regressior__estimator__degree':[1,2,3]}]
       
        mini_pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
    if model=='MLP':   
        base_model=MLPRegressor(solver='adam',max_iter=10000,random_state=40)
        scaler=MinMaxScaler()
        regressior = MultiOutputRegressor(base_model)
        params={'regressor__regressior__estimator__hidden_layer_sizes':[(2,),(3,),(5,),(10,),(5,5),(10,10)],
                 'regressor__regressior__estimator__activation':['logistic', 'tanh', 'relu'],
                 'regressor__regressior__estimator__alpha':[0.0001,0.001,0.01,0.1]}
        mini_pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
        
    kfold1=KFold(n_splits=cvv,shuffle=False)

    kfold2=KFold(n_splits=cvv+1,shuffle=False)
        
    
    scoring_list=['neg_root_mean_squared_error','neg_mean_absolute_error','r2']
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list,
                                    refit=sc,n_jobs=-1)
    if which=='gridsearch':
        
        grid.fit(x,y)
        return grid
    if which=='score':
        cross_score=cross_val_score(grid, x,y,cv=kfold2,scoring='neg_mean_absolute_percentage_error',
                                n_jobs=-1)
        return cross_score
    if which=='prediction':
        cross_pred=cross_val_predict(grid, x,y,cv=kfold2,
                                n_jobs=-1)
        return cross_pred


def select_model_chain(x,y,model,cv_out,cv_in,which,data):
    
    
    if data=='2': 
        orderr=list(np.arange(0,1010))
        orderr.reverse()
    if data=='2>':
        orderr=list(np.arange(0,1010))
    if data=='3<':
        orderr=list(np.arange(0,21))
        orderr.reverse()
    if data=='3>':
        orderr=list(np.arange(0,21))
        
    
    if model=='LR':
        poly=PolynomialFeatures()
        mini_model= LinearRegression()
        regressior = RegressorChain(mini_model,order=orderr,cv=cv_in)  
        params=[{'regressor__poly__degree':[1,2,3]}]
        mini_pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
    if model== 'KNN':
        mini_model=KNeighborsRegressor()
        regressior = RegressorChain(mini_model,order=orderr,cv=cv_in)  
        params=[{'regressor__regressior__base_estimator__n_neighbors':[1,3,5,7,9,11,13,15]}]
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
    if model== 'DT':
        mini_model=DecisionTreeRegressor(random_state=0)
        regressior = RegressorChain(mini_model,order=orderr,cv=cv_in)  
        params={'regressor__regressior__base_estimator__max_depth':[1,2,3,4,5,6,7,8,9,10],
            'regressor__regressior__base_estimator__min_samples_split':[2,3,4,5,6,7,8,10]}      

        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
    if model=='RF':
        mini_model=RandomForestRegressor(random_state=0)
        regressior = RegressorChain(mini_model,order=orderr,cv=cv_in)  
        params={'regressor__regressior__base_estimator__n_estimators':[2,3,4,5,6,7,8,9,10],
                    'regressor__regressior__base_estimator__max_depth':[1,2,3,4,8,10],
                    'regressor__regressior__base_estimator__min_samples_split':[2,3,4,5,8,10]}
        mini_pipe=Pipeline(steps=[('regressior',regressior)]) 
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
        
    if model=='SVR':
        scaler=MinMaxScaler()
        mini_model = SVR(max_iter=10000)
        regressior = RegressorChain(mini_model,order=orderr,cv=cv_in)  
        params=[{'regressor__regressior__base_estimator__kernel': ['rbf'],
                    'regressor__regressior__base_estimator__gamma':[0.0001,0.1,1,10],
                    'regressor__regressior__base_estimator__C':[0.0001,0.1,1,10]},
                    {'regressor__regressior__base_estimator__kernel':['linear'],
                    'regressor__regressior__base_estimator__C':[0.0001,0.1,1,10]},
                    {'regressor__regressior__base_estimator__kernel':['poly'],
                     'regressor__regressior__base_estimator__C':[0.0001,0.1,1,10],
                     'regressor__regressior__base_estimator__degree':[1,2,3]}]
       
       
        mini_pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
    if model=='ANN':
        mini_model=MLPRegressor(solver='adam',max_iter=10000,random_state=40)
        scaler=MinMaxScaler()
        regressior = RegressorChain(mini_model,order=orderr,cv=cv_in)  
        params={'regressor__regressior__base_estimator__hidden_layer_sizes':[(2,),(3,),(5,),(10,),(5,5),(10,10)],
                 'regressor__regressior__base_estimator__activation':['logistic', 'tanh', 'relu'],
                 'regressor__regressior__base_estimator__alpha':[0.0001,0.001,0.01,0.1]}

        mini_pipe=Pipeline(steps=[('scaler',scaler),('regressior',regressior)])
        pipe=TransformedTargetRegressor(regressor=mini_pipe, transformer=MinMaxScaler())
        
     
        
    kfold1=KFold(n_splits=cv_out,shuffle=False)

    kfold2=KFold(n_splits=cv_out+1,shuffle=False)
        
    
    scoring_list=['neg_root_mean_squared_error','neg_mean_absolute_error']
    grid= GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=kfold1,scoring=scoring_list,
                                    refit='neg_mean_absolute_error',n_jobs=-1)
    if which=='gridsearch':
        
        grid.fit(x,y)
        return grid
    if which=='score':
        cross_score=cross_val_score(grid, x,y,cv=kfold2,scoring='neg_mean_absolute_percentage_error',
                                n_jobs=-1)
        return cross_score
    if which=='prediction':
        
        cross_pred=cross_val_predict(grid ,x,y,cv=kfold2,
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
    grid=select_model(x,y,model,7,'gridsearch')
    x_list=[]
    for i in range(0,number):
        x_list.append(i)
    x_list=np.array(x_list).reshape(-1,1)
    y_pred=grid.predict(x_list)
    return y_pred
    




#number gozashtim
def base_model(x,y,model,cv,which,number=40):
    
    grid=select_model(x,y,model,cv,'gridsearch') 
    
    if which=='p':  
        y_pred=grid.predict(x)
        return y_pred,grid
    
    if which=='s':
        xf=x_data1[number].reshape(-1,1)
        y_pred=grid.predict(xf)
        return y_pred
    
    
def metaa_model(model,meta_model,which):

    if which=='p':
        pre_x=x_data1
        y=y_data1
        x1=x_data1.reshape(-1)
        x1=pd.Series(x1)
        y_pred=base_model(pre_x,y,model,7,'p',0)[0]
        x2=pd.DataFrame(y_pred)
        x=pd.concat([x1,x2],axis=1)
        x=np.array(x).reshape(-1,4)
        #r2 ezafe krdim
        meta_grid=select_model(x,y,meta_model,7,'gridsearch','r2')
        return meta_grid

    if which=='s':
        score1_list=[]
        for i in range(0,7):
            pre_x=x_data1
            pre_x=np.delete(pre_x,i)
            x1=pre_x.reshape(-1)
            x1=pd.Series(x1) 
            y_d=y_data1
            y_d=np.delete(y_d,i,0)   
            y_pred=base_model(pre_x.reshape(-1,1),y_d,model,6,'p',0)[0]
            x2=pd.DataFrame(y_pred)  
            y=y_data1  
            x=pd.concat([x1,x2],axis=1)
            x=np.array(x).reshape(-1,4)
            
            #r2 ezafe krdim
            meta_grid=select_model(x,y_d,meta_model,6,'gridsearch','r2')   
            

            x11=x_data1[i].reshape(-1)
            x11=pd.Series(x11) 
            y_predd=base_model(pre_x.reshape(-1,1),y_d,model,6,'s',i)
            x22=pd.DataFrame(y_predd)  
            x=pd.concat([x11,x22],axis=1)
            x=np.array(x).reshape(-1,4)
            
            meta_pred=meta_grid.predict(x)
            
            #score2=
            score1=mean_absolute_percentage_error(meta_pred,y_data1[i,:].reshape(1,-1))
            score1_list.append(score1)
            
        return np.array(score1_list).mean()  



     

def base_model_predict (n,model):
    pre_x=x_data1
    y=y_data1
    #inja 0 ro hzf krdim
    grid=base_model(pre_x,y,model,7,'p')[1]
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
    
 









    
    #what is comparison
def plot_prediction_section1(single_model,model,meta_model,range_number,comparison):
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
    
    
       
    plt.scatter(x_count,plot_two,label='two per output stacking')
    plt.scatter(x_count,plot_list[:,2],label='one output stacking')
    plt.scatter(x_count,yyy,label='classic machine leaning')
    plt.scatter(x_count,y_mo[:,2],label='one seperate multi output')
    #dota balae shabihe hame
    plt.scatter(x_count,y_mo_two,label='two per multi output')
    
    
    
    plt.scatter(x_data1,y_data1[:,2].reshape(-1,1),c='k',label='Experimental')
    plt.xlabel('GPA %') # X-Label
    plt.ylabel('T/D') # Y-Label
    plt.legend(loc='lower center')
    plt.show()
    
    
#plot_prediction_section1('LR','SVR','LR',50)

#avalk abi bad narenji bad asabz bad ghermez bad banafsj


def score_section1():
    global final
    models=['LR','KNN','DT','RF','SVR','MLP']
    score_list=[]
    for i in range(0,6):
        model=models[i]
        for i in range(0,6):
            meta_model=models[i]
        
            score=metaa_model(model, meta_model, 's')
            print('this Loop passed===========================================================')
            score_list.append(score)
    
    final=pd.DataFrame(data=((score_list[0:6]),(score_list[6:12]),(score_list[12:18]),
                       (score_list[18:24]),(score_list[24:30]),(score_list[30:36])),
                       index=models,
                       columns=models)
            
    return final

def plot_score_section1(which):
    data=score_section1()  
    if which=='table':
        df = pd.DataFrame(data)
        df.to_excel('output.xlsx', index=False)   
    if which=='heatmap':
        #'coolwarm'
        sns.heatmap(data, cmap="viridis", annot=True, fmt='.2f')
    if which=='spider':
        labels = ['Label 1', 'Label 2', 'Label 3', 'Label 4','label5','label6','label7']
        data1 = [0.97,0.93,0.80,0.6,0.70,0.4,0.3]
        data2=[0.8,0.85,0.33,0.95,0.8,0.3,0.5]
        data3=[0.4,0.33,0.23,0.1,0.8,0.2,0.6]
        data4=[0.95,0.97,1,1,0.89,0.91,0.3]
        #angles=['c1','c2','c3','c4','c5','c6','c7']
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        data1 = np.concatenate((data1,[data1[0]]))
        data2 = np.concatenate((data2,[data2[0]]))
        data3 = np.concatenate((data3,[data3[0]]))
        data4 = np.concatenate((data4,[data4[0]]))
        angles = np.concatenate((angles,[angles[0]]))
        fig = plt.figure()
        #if it is False it can be a good but it must not connected
        ax = fig.add_subplot(111, polar=True)
        ax.plot()
        ax.plot(angles, data3, 'o-', linewidth=1,markersize=3,label='ANN')
        ax.plot(angles, data4, 'o-', linewidth=1,markersize=3,label='SVR')
        ax.plot(angles, data2, 'o-', linewidth=1,markersize=3,label='KN')
        ax.plot(angles, data1, 'o-', linewidth=1,markersize=3,label='OH')
        ax.fill(angles, data1, alpha=0)
        ax.fill(angles, data2, alpha=0)
        ax.fill(angles, data3, alpha=0)
        ax.fill(angles, data4, alpha=0)
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        ax.set_thetagrids(angles*180/np.pi, labels)
        ax.set_title('Radar Plot')
        ax.legend()
        plt.show()
        
        
plot_score_section1('heatmap')








#============================================================================
'                             Section2 : P                                 '
#============================================================================

#havasemon b transformer bashe
#va hmchnin too section 1 too meta bayad scoremon ychi dg bashe


def prediction_section2(model,which,n):

    x=x_data2
    y=data2_outputs
    
    grid=select_model_chain(x, y, model, 5,4, 'gridsearch','2')
    
    if which=='p':
        x_list=[]
        for i in range(0,n):
            x_list.append(i)
        pred=grid.predict(x_list)
        
        return pred
        
    if which=='s':
        #in motasefane kol ro mige
        #score=grid.best_score_
        best=grid.best_estimator_
        x_list=[]
        for i in range(0,5):
            x=np.delete(x,i)
            y=np.delete(y,i,0)
            best.fit(x,y)
            pr=best.predict(x[i])
            x_list.append(pr)
            #inn pre mone va bayad x e akharesho bkshim va 
            #score bgirim
        score=np.array(x_list).mean()

        print(score)


pred[:,0]
#bayad biaym az cross_val bar roye grid estefade konim



#single ha mitonan frgh konan 
#too suplementory mitonim single haro baham moghayesee konim hata

def plot_prediction_section2(single_model,model,which,range_number):
    #single and other
    if which=='':
        j=range_number
        plot_list=prediction_section2(model,j) #akharin columns
        x=x_data2
        y=y_data2[] #akharin
        x_count=np.arange(j)
        x_count=x_count.reshape(-1,1)
        
        
        yyy=single_section2().reshape(1,-1)
        y_mo=multi_output_section2()[].reshape(1,-1)#akharin column 
        #** ya inja mishe column moshakhas kard ya too scatter,, okeye
        
        
        plt.scatter(x_count,plot_list,label='chain output ')
        plt.scatter(x_count,yyy,label='classic machine leaning')

        #inja y_data [columni ke mikhaym]
        plt.scatter(x_data1,y_data1[:,2].reshape(-1,1),c='k',label='Experimental')
        plt.xlabel('GPA %') # X-Label
        plt.ylabel('T/D') # Y-Label
        plt.legend(loc='lower center')
        plt.show()
        
    
    #each one
    if which=='':
    

    

#niaz b taghir dre ama inja bjaye do loope 36 tae az yek loop estefade mikonim
def score_section2():
    global final
    models=['LR','KNN','DT','RF','SVR','MLP']
    score_list=[]
    for i in range(0,6):
        model=models[i]
        for i in range(0,6):
            meta_model=models[i]
        
            score=metaa_model(model, meta_model, 's')
            score_list.append(score)
    
    final=pd.DataFrame(data=((score_list[0:6]),(score_list[6:12]),(score_list[12:18]),
                       (score_list[18:24]),(score_list[24:30]),(score_list[30:36])),
                       index=models,
                       columns=models)
            
    return final

def plot_score_section2(which):
    data=score_section2()  
    if which=='table':
        df = pd.DataFrame(data)
        df.to_excel('output.xlsx', index=False)   
    if which=='heatmap':
        #'coolwarm'
        sns.heatmap(data, cmap="viridis", annot=True, fmt='.2f')
    if which=='spider':
        labels = ['Label 1', 'Label 2', 'Label 3', 'Label 4','label5','label6','label7']
        data1 = [0.97,0.93,0.80,0.6,0.70,0.4,0.3]
        data2=[0.8,0.85,0.33,0.95,0.8,0.3,0.5]
        data3=[0.4,0.33,0.23,0.1,0.8,0.2,0.6]
        data4=[0.95,0.97,1,1,0.89,0.91,0.3]
        #angles=['c1','c2','c3','c4','c5','c6','c7']
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        data1 = np.concatenate((data1,[data1[0]]))
        data2 = np.concatenate((data2,[data2[0]]))
        data3 = np.concatenate((data3,[data3[0]]))
        data4 = np.concatenate((data4,[data4[0]]))
        angles = np.concatenate((angles,[angles[0]]))
        fig = plt.figure()
        #if it is False it can be a good but it must not connected
        ax = fig.add_subplot(111, polar=True)
        ax.plot()
        ax.plot(angles, data3, 'o-', linewidth=1,markersize=3,label='ANN')
        ax.plot(angles, data4, 'o-', linewidth=1,markersize=3,label='SVR')
        ax.plot(angles, data2, 'o-', linewidth=1,markersize=3,label='KN')
        ax.plot(angles, data1, 'o-', linewidth=1,markersize=3,label='OH')
        ax.fill(angles, data1, alpha=0)
        ax.fill(angles, data2, alpha=0)
        ax.fill(angles, data3, alpha=0)
        ax.fill(angles, data4, alpha=0)
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        ax.set_thetagrids(angles*180/np.pi, labels)
        ax.set_title('Radar Plot')
        ax.legend()
        plt.show()
        
    
    
#============================================================================
'                             Section3 : T 1/2                                '
#============================================================================

#havasemon bashe ke t 1/2 ro nddim behesha

model='SVR'

def half_time(a,b):

    pred_list=[]
    for i in range(0,7):
        
        pred_f=a[i]/b[i,0]
        
        pred_list.append(pred_f)
    y_pred=[]
    for j in range(0,7):
        predd=pred_list[j]
        diff_list=[]
        for k in range(0,21):
            diff=abs(predd[k]-0.5)
            diff_list.append(diff)
        minn=min(diff_list)
        ii=diff_list.index(minn)
        y_pred_n=ii*20
        y_pred.append(y_pred_n)
    return y_pred

half_time(a,b)


a=np.array(y_data3_drain)
b=np.array(y_data3_foam)
def prediction_section3(model,which,n):
    x=x_data3
    y1=y_data3_foam
    y2=y_data3_drain
    if which=='p':
        grid1=select_model_chain(x, y1, model, 7, 6, 'gridsearch', '3<')
        grid2=select_model_chain(x, y2, model, 7, 6, 'gridsearch', '3>')
                                 
        x_list=[]
        for i in range(0,n):
            x_list.append(i)  
        
        pred1=grid1.predict(np.array(x_list).reshape(-1,1))
        pred2=grid2.predict(np.array(x_list).reshape(-1,1))
        return pred1,pred2
    if which=='s':
        pred1=select_model_chain(x,y1,model,6,5,'prediction','3<')
        pred2=select_model_chain(x,y2,model,6,5,'prediction','3>')
        pred_list=[]
        for i in range(0,7):
            
            pred_f=pred2[i]/pred1[i,0]
            
            pred_list.append(pred_f)
        y_pred=[]
        for j in range(0,7):
            predd=pred_list[j]
            diff_list=[]
            for k in range(0,21):
                diff=abs(predd[k]-0.5)
                diff_list.append(diff)
            minn=min(diff_list)
            ii=diff_list.index(minn)
            y_pred_n=ii*20
            y_pred.append(y_pred_n)
    
        
        
        score=mean_absolute_percentage_error(np.array(y_pred).reshape(1,-1),np.array(y_real[0]).reshape(1,-1))
        print('not yet')

prediction_section3('SVR','p', 40)


    
    
def plot_prediction_section3():
    
def plot_score_section3():
print('salm')


#aval train roo alll

