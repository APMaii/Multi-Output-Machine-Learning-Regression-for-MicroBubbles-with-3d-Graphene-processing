"""
---------------IN THE NAME OF GOD------------------
1-dr akh;laghi
2-framework
3-microbubble project
4-final_microbubble
5-mirobubbl_twosection
6-final 3section
7-section3_def
8-starting_final
9-F_3_D_for_myself
10-F_3_D_for_hypercomputer


1--> from zero
2----> without def
3--->


6--final 3 section is best
DEscription:
This project is about pressure per mw

"""



#=============================================================================
' import'
#=============================================================================
import pandas as pd

import numpy as np

from numpy import mean

from numpy import std

from sklearn.datasets import make_regression

from sklearn.model_selection import RepeatedKFold

#from keras.models import Sequential

#from keras.layers import Dense
 



from numpy import absolute

from numpy import mean

from numpy import std

from sklearn.datasets import make_regression

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedKFold




from sklearn.datasets import make_regression

from sklearn.linear_model import LinearRegression




from sklearn.datasets import make_regression

from sklearn.neighbors import KNeighborsRegressor





from sklearn.datasets import make_regression

from sklearn.tree import DecisionTreeRegressor




from sklearn.datasets import make_regression

from sklearn.multioutput import MultiOutputRegressor

from sklearn.svm import LinearSVR






from numpy import mean

from numpy import std

from numpy import absolute

from sklearn.datasets import make_regression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedKFold

from sklearn.multioutput import RegressorChain

from sklearn.svm import LinearSVR






from sklearn.datasets import make_regression

from sklearn.multioutput import RegressorChain

from sklearn.svm import LinearSVR


#=============================================================================
' FILE AND DATA '
#=============================================================================

'using https://edit-csv.net/'

#OUR FILE
file_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\finalcsv.csv'
#df = pd.read_excel(file_location)
#print(df)
f=open(file_location,'r')
data=pd.read_csv(f)
#f.close()
#data.drop(1011,axis=0,inplace=True) index start from 0
data.drop(1010,axis=0,inplace=True)

#file is ready to work

#we want to produce new file wich one has 5 row and inputs is its mwe 
#and its output is substract that


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


#our data is ready to feed our model



#=============================================================================
'visulation of our data'
#=============================================================================

import matplotlib.pyplot as plt 

def first_plot_2d(percent):
    
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
    # plot() is used for plotting a line plot
    plt.plot(x,y)
    # Adding title, xlabel and ylabel
    a='Dencity difference per pressure steps for'
    b=str(percent)
    c="%"
    titles=a+b+c

    plt.title(titles) # Title of the plot
    plt.xlabel('Pressure step(from 1 to 1001)') # X-Label
    plt.ylabel('Dencity difference') # Y-Label
    leg = ax.legend()
    plt.show()
    print('out of else')
    



def if_sanj(a):
    if a in [1,2,3]:
        print('yes')
    else:
        print('no')
        


def plot_2d(percent,p_or_r,model):
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
            # plot() is used for plotting a line plot
            plt.plot(x,y)
            # Adding title, xlabel and ylabel
            a='Dencity difference per pressure steps for'
            b=str(percent)
            c="%"
            titles=a+b+c
            plt.title(titles) # Title of the plot
            plt.xlabel('Pressure step(from 1 to 1001)') # X-Label
            plt.ylabel('Dencity difference') # Y-Label
            leg = ax.legend()
            plt.show()
            print('out of else')
        else:
            print('we dont have this number in our real data')
            
    if p_or_r=='p':
        number=percent
        XX=data_input
        yy=data_outputs
        
        if model=='LR':
            model = LinearRegression()
        if model=='KNN':
            model = LinearRegression()
        if model=='DT':
            model = LinearRegression()
        if model=='SVR':
            model = LinearRegression()
        if model=='ANN':
            model = LinearRegression()
            
        # fit model
        model.fit(XX, yy)

        # make a prediction
        #*****************
        row = [number]
        yhat = model.predict([row]).reshape(-1,1)
        x = np.arange(1010)
        # plot() is used for plotting a line plot
        plt.plot(x,yhat)
        # Adding title, xlabel and ylabel
        a='Dencity difference per pressure steps for'
        b=str(percent)
        c="%"
        titles=a+b+c

        plt.title(titles) # Title of the plot
        plt.xlabel('Pressure step(from 1 to 1001)') # X-Label
        plt.ylabel('Dencity difference') # Y-Label
        # show() is used for displaying the plot
        leg = ax.legend()
        plt.show()
        print('only in else ')
        
        
        
plot_2d(30,'p','LR')




def allpics_once():
    fig, ax = plt.subplots()


    # Plot 
    y1=data_outputs.loc[0]
    y2=data_outputs.loc[1]
    y3=data_outputs.loc[2]
    y4=data_outputs.loc[3]
    y5=data_outputs.loc[4]
    x = np.arange(1010)

    ax.plot(x, y1, color= 'blue',label='0% HAGP')
    ax.plot(x,y2, color = 'red',label='4% HAGP')
    ax.plot(x,y3, color = 'lime',label='8% HAGP')
    ax.plot(x,y4, color = 'cyan',label='16% HAGP')
    ax.plot(x,y5, color = 'darkviolet',label='24% HAGP')
    plt.title('Dencity difference per pressure steps ') # Title of the plot
    plt.xlabel('Pressure step(from 1 to 1001)') # X-Label
    plt.ylabel('Dencity difference') # Y-Label
    leg = ax.legend()

    # Show

    plt.show()
    
allpics_once()


def alpics_separate():
    y1=data_outputs.loc[0]
    y2=data_outputs.loc[1]
    y3=data_outputs.loc[2]
    y4=data_outputs.loc[3]
    y5=data_outputs.loc[4]
    x = np.arange(1010)
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(x, y1)
    axs[0, 0].set_title("0% HAGP")
    axs[0, 1].plot(x, y2)
    axs[0, 1].set_title("4% HAGP")
    axs[1, 0].plot(x, y3)
    axs[1, 0].set_title("8% HAGP")
    axs[1, 0].sharex(axs[0, 0])
    axs[1, 1].plot(x, y4)
    axs[1, 1].set_title("16% HAGP")
    axs[2, 0].plot(x, y5)
    axs[2, 0].set_title("24% HAGP")
    fig.tight_layout()
    # Show

    
    
alpics_separate()



#=============================================================================
' Model selection '
#=============================================================================
from numpy import absolute
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

#---------------------------------------------------------------
'linear regression'
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
# create datasets
X=data_input
y=data_outputs

# define model
model = LinearRegression()
# fit model
model.fit(X, y)


cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force the scores to be positive
n_scores = absolute(n_scores)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# make a prediction
#*****************
row = [24]
yhat = model.predict([row])
# summarize prediction
print(yhat)

#result
#MAE: 0.004 (0.002)


#-----------------------------------------------------------------
'KNN'
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
# create datasets
X=data_input
y=data_outputs
# define model
model = KNeighborsRegressor()
# fit model
model.fit(X, y)

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force the scores to be positive
n_scores = absolute(n_scores)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))



# make a prediction
row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
yhat = model.predict([row])

# summarize prediction
print(yhat[0])
#result


#------------------------------------------------------------------
'decision tree'
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
# create datasets
X=data_input
y=data_outputs
# define model
model = DecisionTreeRegressor()
# fit model
model.fit(X, y)


cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force the scores to be positive
n_scores = absolute(n_scores)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))



#MAE: 0.004 (0.002)


# make a prediction
row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
yhat = model.predict([row])
# summarize prediction
print(yhat[0])



#-----------------------------------------------------------------
'svr'
# example of making a prediction with the direct multioutput regression model
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR

model = LinearSVR()
# define the direct multioutput wrapper model
wrapper = MultiOutputRegressor(model)
# fit the model on the whole dataset
wrapper.fit(X, y)
# make a single prediction
#*******************
row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
yhat = wrapper.predict([row])
# summarize the prediction
print('Predicted: %s' % yhat[0])


# example of evaluating chained multioutput regression with an SVM model
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import RegressorChain
from sklearn.svm import LinearSVR
X=data_input
y=data_outputs
# define base model
model = LinearSVR()
# define the chained multioutput wrapper model
wrapper = RegressorChain(model, order=[0,1])
model = LinearSVR()
# define the chained multioutput wrapper model
wrapper = RegressorChain(model)
# define the evaluation procedure
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(wrapper, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force the scores to be positive
n_scores = absolute(n_scores)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


#result: MAE: 0.005 (0.003)

# example of making a prediction with the chained multioutput regression model
from sklearn.datasets import make_regression
from sklearn.multioutput import RegressorChain
from sklearn.svm import LinearSVR
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# define base model
model = LinearSVR()
# define the chained multioutput wrapper model
wrapper = RegressorChain(model)
# fit the model on the whole dataset
wrapper.fit(X, y)
# make a single prediction
row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
yhat = wrapper.predict([row])
# summarize the prediction
print('Predicted: %s' % yhat[0])
#---------------------------------------------------------------
'neural network'


# get the dataset
#def get_dataset():
# X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=3, random_state=2)
#return X, y
########X=data_input
#######y=data_outputs
 
# get the model
def get_model(n_inputs, n_outputs):
 model = Sequential()
 model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
 model.add(Dense(n_outputs))
 model.compile(loss='mae', optimizer='adam')
 return model
 
# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
 results = list()
 n_inputs, n_outputs = X.shape[1], y.shape[1]
 # define evaluation procedure
 cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
 # enumerate folds
 for train_ix, test_ix in cv.split(X):
 # prepare data
 X_train, X_test = X[train_ix], X[test_ix]
 y_train, y_test = y[train_ix], y[test_ix]
 # define model
 model = get_model(n_inputs, n_outputs)
 # fit model
 model.fit(X_train, y_train, verbose=0, epochs=100)
 # evaluate model on test set
 mae = model.evaluate(X_test, y_test, verbose=0)
 # store result
 print('>%.3f' % mae)
 results.append(mae)
 return results
 
# load dataset
#X, y = get_dataset() 
X=data_input
y=data_outputs
# evaluate model
results = evaluate_model(X, y)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(results), std(results)))

#prediction
# load dataset
X=data_input
y=data_outputs
n_inputs, n_outputs = X.shape[1], y.shape[1]
# get model
model = get_model(n_inputs, n_outputs)
# fit the model on all data
model.fit(X, y, verbose=0, epochs=100)
# make a prediction for new data
#********************
#it is new MWE
#********************
row = []
newX = asarray([row])
yhat = model.predict(newX)
print('Predicted: %s' % yhat[0]











