"""
Description:
    
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
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
#=========================================================================
'''
section 1
'''
#=========================================================================
"""
Description:1-kfold 2-plot 3-multiple model 4-deeplearning
    
"""

def section1(model,j):
    x=np.array((0.00,0.04,0.08,0.16,0.24)).reshape(5,1)
    y=np.array([(83.2,10.2,12),(65.8,11.3,17),(49.7,11.9,24),
                (39.0,12.6,32),(43.2,12.8,30)])
    if model=='LR':
        poly=PolynomialFeatures()
        regressior = LinearRegression()
        params=[{'poly':[None]},{'poly__degree':[1,2,3,4]}]
        pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model== 'KNN':
        regressior = KNeighborsRegressor()
        params=[{'regressior__n_neighbors':[2,3,4,5,6,7,8,9,10,15,20]}]
        pipe=Pipeline(steps=[('regressior',regressior)]) 
    if model== 'DT':
        regressior = DecisionTreeRegressor()
        pipe=Pipeline(steps=[('regressior',regressior)])
        params={'regressior__max_depth':[1,2,3,4,5,6,7,8,9,10],
        'regressior__max_leaf_nodes':[2,3,4,5,6,7,8,10]}
    if model=='RF':
        regressior=RandomForestRegressor(random_state=0)
        pipe=Pipeline(steps=[('regressior',regressior)])
        params={'regressior__n_estimators':[100,200],
                'regressior__max_depth':[None,1,2,3,4,5,6]}
    if model=='SVR':
        mini_model = SVR(max_iter=10000)
        scaler=MinMaxScaler()
        poly=PolynomialFeatures()
        regressior = MultiOutputRegressor(mini_model)
        pipe=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
        params=[{'poly__degree':[1,2,3,4],
                 'regressior__estimator__kernel': ['rbf'],
                'regressior__estimator__gamma':[0.0001,0.1,1,10,100],
                'regressior__estimator__C':[0.0001,0.1,1,10,100]},
                {'poly__degree':[1,2,3,4],
                 'regressior__estimator__kernel':['linear'],
                'regressior__estimator__C':[0.0001,0.1,1,10,100]}]
    if model=='ANN':
        
        mini_model=MLPRegressor(solver='lbfgs',activation='tanh',alpha=0.001,max_iter=10000,
                                random_state=40)
        scaler=MinMaxScaler()
        poly=PolynomialFeatures()
        regressior = MultiOutputRegressor(mini_model)
        pipe=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
        params={'regressior__estimator__hidden_layer_sizes':[[10,10],[100,100],
                                                             [100,200]],
                'poly__degree':[1,2,3,4]}
    kfold=KFold(n_splits=4,shuffle=False)
    scoring_list=['neg_mean_absolute_error','neg_mean_squared_error',
                  'neg_mean_absolute_percentage_error']
    pipe_gc = GridSearchCV(pipe,
                              param_grid=params,
                              cv=kfold,scoring=scoring_list,
                              refit='neg_mean_absolute_error',n_jobs=-1)
    pipe_gc.fit(x, y)
    cvv=KFold(n_splits=5,shuffle=False)
    cross=cross_val_score(pipe_gc, x,y,scoring='neg_mean_absolute_percentage_error',cv=cvv,n_jobs=-1)
    global cv_results
    cv_results = pipe_gc.cv_results_
    print('our cross is ', cross)
    print('mean cross',cross.mean())
    print('best score',pipe_gc.best_score_) # ino ome az mean_t_n_m_a_e vrdashte refit
    a= np.argmax(pipe_gc.cv_results_['mean_test_neg_mean_absolute_error'])
    print('a:',a)
    print('best mae',pipe_gc.cv_results_['mean_test_neg_mean_absolute_error'][pipe_gc.best_index_])
    print('best mape',pipe_gc.cv_results_['mean_test_neg_mean_absolute_percentage_error'][pipe_gc.best_index_])
    print('best mape index a',pipe_gc.cv_results_['mean_test_neg_mean_absolute_percentage_error'][a])
    print('best mse index a',pipe_gc.cv_results_['mean_test_neg_mean_squared_error'][a])
    for i in range(0,j):
        if i==0:
            y1=[12,17,24,32,30]
            y2=[]
            y3=[]
            y4=[]
            x = np.arange(0,j).reshape(1,-1)
        xx=i/100
        xxx=np.array(xx).reshape(1,-1)
        yp=pipe_gc.predict(xxx)
        yp2=(yp[0,1]/yp[0,0])*100
        yp3=yp[0,2]
        yp4=(yp2+yp3)/2
        y2.append(yp2)
        y3.append(yp3)
        y4.append(yp4)
        if i==(j-1):
            yy1=np.array(y1).reshape(1,-1)
            yy2=np.array(y2).reshape(1,-1)
            yy3=np.array(y3).reshape(1,-1)
           # yy4=np.array(y4).reshape(1,-1)
            for j in range(0,11):
                global mape_list
                if j==0: 
                    mape_list=[]
                a=j/10
                b=(10-j)/10
                yyy4= (a*yy3 + b*yy2)
                y_s_final=np.array([yyy4[0,0],yyy4[0,4],yyy4[0,8],yyy4[0,16],yyy4[0,24]]).reshape(1,-1)
                mape=mean_absolute_percentage_error(y_s_final,yy1)
                mape_list.append(mape) 
            global minn
            global ii
            minn=min(mape_list)
            ii=mape_list.index(minn)
            aa=ii/10
            bb=(10-ii)/10
            y_final=(aa*yy3+bb*yy2)        
            y_finall=np.array([y_final[0,0],y_final[0,4],y_final[0,8],y_final[0,16],y_final[0,24]]).reshape(1,-1)
            mae = mean_absolute_error(y_finall,yy1)
            print('revolusion mae',mae)
            mape=mean_absolute_percentage_error(y_finall,yy1)
            print('revolusion maep',mape)
            xxxx=np.array([0,4,8,16,24]).reshape(1,-1) 
            plt.scatter(x,yy2,c='b')
            plt.scatter(x,yy3,c='k')
            plt.scatter(x,y_final,c='g')
            plt.scatter(xxxx,yy1,marker='*',s=200,c='r')
            plt.title('it is') # Title of the plot
            plt.xlabel('Pressure step(from 1 to 1001)') # X-Label
            plt.ylabel('Dencity difference') # Y-Label
            plt.show()
    
section1('SVR',30)

#=========================================================================
'''
section 2
'''
#=========================================================================
file_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\finalcsv.csv'
f=open(file_location,'r')
data=pd.read_csv(f)
data.drop(1010,axis=0,inplace=True)
data0 = data['ob'] - data['0f']
data4=data['4b']-data['4f']
data8=data['8b']-data['8f']
data16=data['16b']-data['16f']
data24=data['24b']-data['24f']
data_output=pd.concat([data0,data4,data8,data16,data24],axis=1)
data_outputs=data_output.T
data_input=pd.DataFrame(data=[0,4,8,16,24] ,columns=['%HAGP'])
finall_data= pd.concat([data_input, data_outputs], axis=1)
final_data=finall_data.rename(index={0:'0',1:'4',2:'8',3:'16',4:'24'})
def section2(percent,p_or_r,model,chain):
    if p_or_r=='r':
        if percent in [0,4,8,16,24]:
            if percent==0:
                number=0
            if percent==4:
                number=1
            if percent==8:
                number=2
            if percent==16:
                number=3
            if percent==24:
                number=4   
            y=data_outputs.loc[number]
            x = np.arange(1010)
            plt.plot(x,y)
            a='Dencity difference per pressure steps for'
            b=str(percent)
            c="%"
            titles=a+b+c
            plt.title(titles) # Title of the plot
            plt.xlabel('Pressure step(from 1 to 1001)') # X-Label
            plt.ylabel('Dencity difference') # Y-Label
            plt.show()
        else:
            print('we dont have this number in our real data')   
    if p_or_r=='p':
        number=percent
        XX=data_input
        yy=data_outputs
        row = [number]
        if model=='LR':
            poly=PolynomialFeatures()
            if chain=='on':
                mini_model= LinearRegression()
                regressior = RegressorChain(mini_model,order=list(range(0,1011)))    
            if chain=='off':
                regressior = LinearRegression()  
            params=[{'poly__degree':[1,2]}]
            pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
        if model== 'KNN':
            if chain=='on':
                mini_model=KNeighborsRegressor()
                regressior = RegressorChain(mini_model)
                params=[{'regressior__base_estimator__n_neighbors':[2,3,4,5,6,7,8,9,10,15,20]}]
            if chain=='off':
                regressior = KNeighborsRegressor()
                params=[{'regressior__n_neighbors':[2,3,4,5,6,7,8,9,10,15,20]}]
            pipe=Pipeline(steps=[('regressior',regressior)])
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
            pipe=Pipeline(steps=[('regressior',regressior)])
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
            pipe=Pipeline(steps=[('regressior',regressior)])
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
            pipe=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
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
            pipe=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
        kfold=KFold(n_splits=4,shuffle=False)
        scoring_list=['neg_mean_absolute_error','neg_mean_squared_error',
                      'neg_mean_absolute_percentage_error']
        pipe_gc = GridSearchCV(pipe,
                                  param_grid=params,
                                  cv=kfold,scoring=scoring_list,
                                  refit='neg_mean_absolute_error',n_jobs=-1)
        pipe_gc.fit(XX, yy)
        cvv=KFold(n_splits=5,shuffle=False)
        cross=cross_val_score(pipe_gc, XX,yy,scoring='neg_mean_absolute_percentage_error',cv=cvv,n_jobs=-1)
        global cv_results
        cv_results = pipe_gc.cv_results_
        print('our cross is ', cross)
        print('mean cross',cross.mean())
        print('best score',pipe_gc.best_score_) # ino ome az mean_t_n_m_a_e vrdashte refit
        a= np.argmax(pipe_gc.cv_results_['mean_test_neg_mean_absolute_error'])
        print('a:',a)
        print('best mae',pipe_gc.cv_results_['mean_test_neg_mean_absolute_error'][pipe_gc.best_index_])
        print('best mape',pipe_gc.cv_results_['mean_test_neg_mean_absolute_percentage_error'][pipe_gc.best_index_])
        print('best mape index a',pipe_gc.cv_results_['mean_test_neg_mean_absolute_percentage_error'][a])
        print('best mse index a',pipe_gc.cv_results_['mean_test_neg_mean_squared_error'][a])
        yhat = pipe_gc.predict([row]).reshape(-1,1)
        x = np.arange(1010)
        plt.plot(x,yhat)
        a='Dencity difference per pressure steps for'
        b=str(percent)
        c="%"
        titles=a+b+c
        plt.title(titles) # Title of the plot
        plt.xlabel('Pressure step(from 1 to 1001)') # X-Label
        plt.ylabel('Dencity difference') # Y-Label
        plt.show()
    
#section2(14,'p','SVR','off') 
#=========================================================================
'''
section 3
'''
#=========================================================================
file_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\section3.csv'
f=open(file_location,'r')
data=pd.read_csv(f)
data0=pd.concat([data['Unnamed: 0'],data['Unnamed: 1']])
data4=pd.concat([data['Unnamed: 2'],data['Unnamed: 3']])
data8=pd.concat([data['Unnamed: 4'],data['Unnamed: 5']])
data16=pd.concat([data['Unnamed: 6'],data['Unnamed: 7']])
data24=pd.concat([data['Unnamed: 8'],data['Unnamed: 9']])
data0 = data0.reset_index(drop=True)
data4 = data4.reset_index(drop=True)
data8 = data8.reset_index(drop=True)
data16 = data16.reset_index(drop=True)
data24 = data24.reset_index(drop=True)
def section3(model,percent,chain):
    x=np.array((0,4,8,16,24)).reshape(-1,1)
    y=pd.concat((data0,data4,data8,data16,data24),axis=1).T
    row = [percent]
    if model=='LR':
        poly=PolynomialFeatures()
        if chain=='on':
            mini_model= LinearRegression()
            regressior = RegressorChain(mini_model)
        if chain=='off':
            regressior = LinearRegression()
        params=[{'poly__degree':[1,2]}]
        pipe=Pipeline(steps=[('poly',poly),('regressior',regressior)])
    if model== 'KNN':
        if chain=='on':
            mini_model=KNeighborsRegressor()
            regressior = RegressorChain(mini_model)
            params=[{'regressior__base_estimator__n_neighbors':[2,3,4,5,6,7,8,9,10,15,20]}]
        if chain=='off':
            regressior = KNeighborsRegressor()
            params=[{'regressior__n_neighbors':[2,3,4,5,6,7,8,9,10,15,20]}]
        pipe=Pipeline(steps=[('regressior',regressior)]) 
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
        pipe=Pipeline(steps=[('regressior',regressior)])     
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
        pipe=Pipeline(steps=[('regressior',regressior)])
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
        pipe=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
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
        pipe=Pipeline(steps=[('scaler',scaler),('poly',poly),('regressior',regressior)])
        

    kfold=KFold(n_splits=5,shuffle=False)
    scoring_list=['neg_mean_absolute_error','neg_mean_squared_error',
                  'neg_mean_absolute_percentage_error']
    pipe_gc = GridSearchCV(pipe,
                              param_grid=params,
                              cv=kfold,scoring=scoring_list,
                              refit='neg_mean_absolute_error',n_jobs=-1)
    pipe_gc.fit(x, y)
    cvv=KFold(n_splits=5,shuffle=False)
    cross=cross_val_score(pipe_gc, x,y,scoring='neg_mean_absolute_percentage_error',
                          cv=cvv,n_jobs=-1)

    global cv_results
    cv_results = pipe_gc.cv_results_
    print('best score',pipe_gc.best_score_) # ino ome az mean_t_n_m_a_e vrdashte refit
    a= np.argmax(pipe_gc.cv_results_['mean_test_neg_mean_absolute_error'])
    print('a:',a)
    print('best mae',pipe_gc.cv_results_['mean_test_neg_mean_absolute_error'][pipe_gc.best_index_])
    print('best mape',pipe_gc.cv_results_['mean_test_neg_mean_absolute_percentage_error'][pipe_gc.best_index_])
    print('best mape index a',pipe_gc.cv_results_['mean_test_neg_mean_absolute_percentage_error'][a])
    print('best mse index a',pipe_gc.cv_results_['mean_test_neg_mean_squared_error'][a])
    yhat = pipe_gc.predict([row]).reshape(-1,1)
    y_left=yhat[0:20]
    y_right=yhat[21:41]
    ax = np.arange(20).reshape(-1,1)
    axx=ax*20
    x_line=np.arange(401).reshape(-1,1)
    a_line=LinearRegression()
    b_line=LinearRegression()
    a_line.fit(axx,y_left)
    b_line.fit(axx,y_right)
    a_pred=a_line.predict(x_line)
    b_pred=b_line.predict(x_line)
    diff=list(abs(a_pred - b_pred) )
    minn=min(diff)
    ii=diff.index(minn)
    c=ii
    print(c)
    t1=' T 1/2 : '
    t2=str(c)
    text_legend=t1+t2
    plt.scatter(axx,y_right)
    plt.scatter(axx,y_left)
    plt.text(320, 1, text_legend, bbox=dict(facecolor='red', edgecolor='black'))
    plt.title('it is') # Title of the plot
    plt.xlabel('Pressure step(from 1 to 1001)') # X-Label
    plt.ylabel('Dencity difference') # Y-Label
    plt.show()
            
#section3('ANN', 8, 'off')
