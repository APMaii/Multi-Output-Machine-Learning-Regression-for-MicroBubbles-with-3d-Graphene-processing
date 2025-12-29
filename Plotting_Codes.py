"""
Created on Wed Oct  4 14:22:48 2023
last edit = OCT -6 


@author: Ali Pilehvar Meibody

Description: The final figer genertor because we are close to submission

#****** Important NOTES
it is from 13june-rasme avalie / final_plot_9june / idk plot / asli



plan: kari ndre 
aval baayd format taeen beshe
bad miaymo havasemon b too bodane thick ha va dp 600 va format bashe fght

bad aval distribution ha va correlation ha k kari ndre

badesh plot haye score k asan kari ndre va plot haye spyder yekam zaman mibare
k onm iradi ndre kolan begir 9 ta hast 


badesh ham 3 ta plot has k baraye rasme onam kari ndre sakht nagir ;)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from  sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.font_manager as fm



import matplotlib.font_manager as fm


#============================================================================
'                                 LOAD DATA                                '
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
'                              DATA distribution                           '
#============================================================================
#linestyle='--', marker='x'
#labelpad in xlabel space between thick and label

xyfont={'family':'Arial','size':20,'fontweight':'bold'}

tfont={'family':'Arial','size':20}
lfont={'family':'Arial','size':20}

#tfont={'family':'Arial','size':10}

import matplotlib.font_manager as fm

def distribution(data,scale,scaler='minmax'):
    
    if scaler=='standard':
        scaler=StandardScaler()
        n='standard'        
    if scaler=='minmax':
        scaler=MinMaxScaler()  
        n='min' 
    if scaler=='power':
        scaler=PowerTransformer()
        n='power' 
    if scaler=='normal': 
        scaler=Normalizer() 
        n='normal' 
    if scaler=='ro': 
        scaler=RobustScaler()
        
    if scale=='off':
        n='off'
        
    global data3_foam_scaled,y_data1_scaled   
    plt.figure(figsize=(12, 8))  # Set the figure size (width: 10 inches, height: 6 inches)
    plt.rcParams["figure.dpi"] = 600 
    plt.rcParams['font.family']='Arial'
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
    name=data+'_'+n+'.pdf'
    plt.xlabel('Value',fontdict=xyfont,labelpad=20)
    plt.ylabel('Data Density',fontdict=xyfont,labelpad=20)
    plt.title('Distribution of Outputs',fontdict=tfont) 
    if data=='1':
        #font = fm.FontProperties(family='Arial')
        #plt.legend(prop=font,fontsize=40)
        plt.legend(fontsize=20)
    ax=plt.gca()
    ax.tick_params(axis='both',which='both',direction='in',
                   bottom=True,top=False,left=True,
                   right=False,labelsize=20)
    #labelsize = 16, pad = 12
    
    plt.savefig(name,dpi=600,format='pdf')
    #plt.show()



distribution('1','off')
distribution('1', 'on','normal')
distribution('1', 'on')
distribution('1', 'on','standard')
distribution('1', 'on','power')

distribution('2','off')
distribution('2', 'on','normal')
distribution('2', 'on')
distribution('2', 'on','standard')
distribution('2', 'on','power')



distribution('3_f','off')
distribution('3_f', 'on','normal')
distribution('3_f', 'on')
distribution('3_f', 'on','standard')
distribution('3_f', 'on','power')

distribution('3_d','off')
distribution('3_d', 'on','normal')
distribution('3_d', 'on')
distribution('3_d', 'on','standard')
distribution('3_d', 'on','power')



#============================================================================
'                              DATA correlation                           '
#============================================================================
def correlation(data):
    if data=='1':
        dataa=np.concatenate((x_data1,y_data1),axis=1)
        xt=['HAG-MP %','MB Size (D)','Shell thickness (T)','T/D']
        yt=['HAG-MP %','Mb Size (D)','Shell thickness (T)','T/D']
        datacorr=pd.DataFrame(dataa)
    if data=='2':
        xt='auto'
        #['wt%']+list(np.arange(1,1011))
        yt='auto'
        #['wt%']+list(np.arange(1,1011))
        dataa=pd.concat((data2_input,data2_outputs),axis=1)
        dataa = dataa.rename(columns={'%HAGP': 'HAG-MP %'})
        #dataa=pd.DataFrame(data=dataa,columns=['HAG-MP %']+list(np.arange(1,1011)))
        datacorr=pd.DataFrame(dataa)
    if data=='3_f':
        xt=['HAG-MP Concentration (wt%)','t(1)','t(2)','t(3)','t(4)','t(5)','t(6)','t(7)','t(8)','t(9)',
            't(10)','t(11)','t(12)','t(13)','t(14)','t(15)','t(16)','t(17)','t(18)'
            ,'t(19)' ,'t(20)']
        yt=['HAG-MP Concentration (wt%)','t(1)','t(2)','t(3)','t(4)','t(5)','t(6)','t(7)','t(8)','t(9)',
            't(10)','t(11)','t(12)','t(13)','t(14)','t(15)','t(16)','t(17)','t(18)'
            ,'t(19)' ,'t(20)']
        dataa=pd.concat((pd.DataFrame(x_data3),y_data3_foam),axis=1)
        datacorr=pd.DataFrame(dataa)
    if data=='3_d':
        xt=['HAG-MP Concentration (wt%)','t(1)','t(2)','t(3)','t(4)','t(5)','t(6)','t(7)','t(8)','t(9)',
            't(10)','t(11)','t(12)','t(13)','t(14)','t(15)','t(16)','t(17)','t(18)'
            ,'t(19)' ,'t(20)']
        yt=['HAG-MP Concentration (wt%)','t(1)','t(2)','t(3)','t(4)','t(5)','t(6)','t(7)','t(8)','t(9)',
            't(10)','t(11)','t(12)','t(13)','t(14)','t(15)','t(16)','t(17)','t(18)'
            ,'t(19)' ,'t(20)']
        dataa=pd.concat((pd.DataFrame(x_data3),y_data3_drain),axis=1)
        datacorr=pd.DataFrame(dataa)   
    #plt.figure()
    plt.figure(figsize=(10, 6))  # Set the figure size (width: 10 inches, height: 6 inches)
    plt.rcParams["figure.dpi"] = 600 
    plt.rcParams['font.family']='Arial'
    correlation = datacorr.corr()  
   # correlation.drop(0,axis=0,inplace=True)
   # correlation.drop(3,axis=1,inplace=True)
   # correlation=np.array(correlation)[1:,:3]
    #mask=np.triu(np.ones_like(correlation,dtype=bool))
    sns.heatmap(correlation,cmap="coolwarm",xticklabels=xt,
                yticklabels=yt)
    
    
    name=data+'_'+'corr.pdf'
    ax=plt.gca()
    ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=False,
                   left=True,right=False,labelsize=10)
    plt.savefig(name,dpi=600,format='pdf')
    
    
correlation('1')
correlation('2')
correlation('3_f')
correlation('3_d')




#============================================================================
'                             SCORING PLOT                                '
#============================================================================
#**check konim bahash
#** badesh biaym doone doone 1-Single / 1-Multi / 1-MTRS badesh 2,3 ke rang avaz she


section=1
whi='MTRS'

#C:\Users\sunhouse\.spyder-py3\codha\projects\exel\plot

    
if section==1:
    sec='1_'
        
if section==2:
    sec='2_'
        
if section==3:
    sec='3_'
        
file1_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\'+sec+whi+'_MAPE' +'.csv'
#f1=open(file1_location,'r')
data1=pd.read_csv(file1_location,index_col=0)
#data1.drop(labels='Unnamed: 0',axis=1,inplace=True)
file2_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\'+sec+whi+'_MAE' +'.csv'
#f2=open(file2_location,'r')
data2=pd.read_csv(file2_location,index_col=0)

file3_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\'+sec+whi+'_MSE' +'.csv'
#f3=open(file3_location,'r')
data3=pd.read_csv(file3_location,index_col=0)
file4_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\'+sec+whi+'_RMSE' +'.csv'
#f4=open(file4_location,'r')
data4=pd.read_csv(file4_location,index_col=0)
    
    
data1=data1
data2=data2
data3=data3*100
data4=data4

#1-->"cividis"
#2-->"inferno"
#3-->"viridis"


# 
plt.figure(figsize=(10, 6))  # Set the figure size (width: 10 inches, height: 6 inches)
plt.rcParams["figure.dpi"] = 600 
plt.rcParams['font.family']='Arial'
yt=['Only T/D','Tp / Dp']
xt=['LR','KNN','DT','RF','SVR','MLP']

#np.array(data1).reshape(1,-1)
#yticklabels=yt,xticklabels=xt

ax=sns.heatmap(data1, cmap="cividis", annot=True,
            fmt='.2f',
            annot_kws={"size":20})


cbar=ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

ax=plt.gca()
ax.tick_params(axis='both',which='both',direction='in',
               bottom=True,top=False,left=True,right=False,labelsize=20,
               pad=12)

#'_RMSE_x10'
#10 100 10
name=sec+whi+'MAPE'+'.pdf'
plt.savefig(name,dpi=600,format='pdf')
    
#*** annot adade dakhel ro minevise
#labelfontfamily oon labele thick haro mzine ama ma labele khdoe adad haro mikhaym

    
'''
for tick in ax.get_xticklabels():
    tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')
    
    

'''
    

#====================spyder=======================
#====================spyder=======================
#====================spyder=======================


section=3
whi='MTRS'


#dfor s1,s2,m2,s3,m3
#ind=[0,0,0,0,0,0]
#col=[0,1,2,3,4,5]

#for m1
#ind=[0,0,0,1,1,1]
#col=[4,0,5,0,4,5]



#for mt1
#ind=[5 ,5 ,4 , 0,5 ,0 ]
#col=[4 ,3 ,4 ,3 ,5 ,4 ]

#for mt2
#ind=[ 0,0 , 5, 5,5 ,5 ]
#col=[ 4,1 ,5 , 0, 1, 3]

#for mt3
ind=[0 ,4 ,4 , 4,5 ,5 ]
col=[1 ,1 ,3 , 4, 1,4 ]


if section==1:
    sec='1_'       
if section==2:
    sec='2_'       
if section==3:
    sec='3_'
        
file1_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\'+sec+whi+'_MAPE' +'.csv'
#f1=open(file1_location,'r')
data1=pd.read_csv(file1_location,index_col=0)
#data1.drop(labels='Unnamed: 0',axis=1,inplace=True)
file2_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\'+sec+whi+'_MAE' +'.csv'
#f2=open(file2_location,'r')
data2=pd.read_csv(file2_location,index_col=0)

file3_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\'+sec+whi+'_MSE' +'.csv'
#f3=open(file3_location,'r')
data3=pd.read_csv(file3_location,index_col=0)
file4_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\'+sec+whi+'_RMSE' +'.csv'
#f4=open(file4_location,'r')
data4=pd.read_csv(file4_location,index_col=0)
    
#data1=1-data1
#data2=1-data2*10
#data3=1-data3*100
#data4=1-data4*10

data1=1-data1
data2=1-data2/100
data3=1-data3/7000
data4=1-data4/100

labels = ['1-MAPE', '1-MAE/100', '1-MSE/7000', '1-RMSE/100']


#.reshape(1,-1)

dataa1=np.array(data1)
dataa2=np.array(data2)
dataa3=np.array(data3)
dataa4=np.array(data4)
    
    



data1 = [dataa1[ind[0]][col[0]],dataa2[ind[0]][col[0]],
             dataa3[ind[0]][col[0]],dataa4[ind[0]][col[0]]]
    
data2=[dataa1[ind[1]][col[1]],dataa2[ind[1]][col[1]],
             dataa3[ind[1]][col[1]],dataa4[ind[1]][col[1]]]
    
data3=[dataa1[ind[2]][col[2]],dataa2[ind[2]][col[2]],
             dataa3[ind[2]][col[2]],dataa4[ind[2]][col[2]]]
    
data4=[dataa1[ind[3]][col[3]],dataa2[ind[3]][col[3]],
             dataa3[ind[3]][col[3]],dataa4[ind[3]][col[3]]]
    
data5=[dataa1[ind[4]][col[4]],dataa2[ind[4]][col[4]],
             dataa3[ind[4]][col[4]],dataa4[ind[4]][col[4]]]
    
data6=[dataa1[ind[5]][col[5]],dataa2[ind[5]][col[5]],
             dataa3[ind[5]][col[5]],dataa4[ind[5]][col[5]]]
    
models=['LR','KNN','DT','RF','SVR','MLP']
    #angles=['c1','c2','c3','c4','c5','c6','c7']
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
data1 = np.concatenate((data1,[data1[0]]))
data2 = np.concatenate((data2,[data2[0]]))
data3 = np.concatenate((data3,[data3[0]]))
data4 = np.concatenate((data4,[data4[0]]))
data5 = np.concatenate((data5,[data5[0]]))
data6 = np.concatenate((data6,[data6[0]]))
angles = np.concatenate((angles,[angles[0]]))
#fig = figsize=(10, 6)
#plt.rcParams["figure.dpi"] = 600 
    #angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)

    #if it is False it can be a good but it must not connected
#fig=plt.figure(figsize=(16,12))  # Set the figure size (width: 10 inches, height: 6 inches)
#[6.4, 4.8]
fig=plt.figure(figsize=(12.8,9.6))
plt.rcParams["figure.dpi"] = 600 
#fig = plt.figure()
plt.rcParams['font.family']='Arial'

ax = fig.add_subplot(111, polar=True)
ax.plot()

ax.plot(angles, data1, 'o-', linewidth=1,markersize=5,
           # label=models[col[0]])
            #label='Only - '+ models[col[0]])
           label=models[ind[0]]+'--'+models[col[0]])
ax.plot(angles, data2, '^-', linewidth=1,markersize=5,
            #label=models[col[1]])
            #label='Only - '+models[col[1]])
            label=models[ind[1]]+'--'+models[col[1]])            
ax.plot(angles, data3, 's-', linewidth=1,markersize=5,
            #label=models[col[2]])
            #label='Only - '+models[col[2]])
            label=models[ind[2]]+'--'+models[col[2]])
ax.plot(angles, data4, 'p-', linewidth=1,markersize=5,
            #label=models[col[3]])
            #label='Two P - '+models[col[3]])
            label=models[ind[3]]+'--'+models[col[3]])
ax.plot(angles, data5, '*-', linewidth=1,markersize=5,
            #label=models[col[4]])
            #label='Two P - '+models[col[4]])
            label=models[ind[4]]+'--'+models[col[4]])
ax.plot(angles, data6, 'd-', linewidth=1,markersize=5,
            #label=models[col[5]])
            #label='Two P - '+models[col[5]])
            label=models[ind[5]]+'--'+models[col[5]])
    
ax.fill(angles, data1, alpha=0)
ax.fill(angles, data2, alpha=0)
ax.fill(angles, data3, alpha=0)
ax.fill(angles, data4, alpha=0)
ax.fill(angles, data5, alpha=0)
ax.fill(angles, data6, alpha=0)
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
#angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
#font = fm.FontProperties(family='Arial')

ax.set_thetagrids(angles*180/np.pi, labels)
#ax.set_title('Radar Plot',fontdict=tfont)

#ax.legend(loc='upper left',prob=font)
font = fm.FontProperties(size=20)
ax.legend(loc='upper left',bbox_to_anchor=(-0.15,1.1),fontsize=20)

ax.annotate(labels[0],xy=(1,0),fontsize=20,fontweight='bold')
ax.annotate(labels[1],xy=(0,1),fontsize=20,fontweight='bold')
ax.annotate(labels[2],xy=(-0.15,0),fontsize=20,fontweight='bold')
ax.annotate(labels[3],xy=(0,0),fontsize=20,fontweight='bold')


#ax.tick_params(labelsize=20,pad=45,fontweihght='bold')

ax.set_ylim(0.9,1)
#ax.set_rmax(1)
#axis='both',which='major',direction='in',bottom=True,top=False,
               #left=True,right=False,
name=sec+'Spyder'+whi+'_'+'section'+'.pdf'

plt.savefig(name,dpi=600,format='pdf')









#============================================================================
'                             PREDICTION PLOT 1                          '
#============================================================================
xyfont={'family':'Arial','size':20,'fontweight':'bold'}

tfont={'family':'Arial','size':20}
lfont={'family':'Arial','size':20}


x_count=np.arange(50)/100
x_count=x_count.reshape(-1,1)


file1_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\y_single_1' 
#f1=open(file1_location,'r')
y_single=np.array(pd.read_csv(file1_location,index_col=0))

file2_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\plot_list_1' 
#f2=open(file2_location,'r')
plot_list=np.array(pd.read_csv(file2_location,index_col=0))

file3_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\y_mo_1'
#f3=open(file3_location,'r')
y_mo=np.array(pd.read_csv(file3_location,index_col=0))

#plott----------------------------------------------------------------

import matplotlib.colors as mcolors

#mcolors.CSS4_COLORS




plt.figure(figsize=(12, 8))  # Set the figure size (width: 10 inches, height: 6 inches)
plt.rcParams["figure.dpi"] = 600 
plt.rcParams['font.family']='Arial'

#gray=#000080  
plt.scatter(x_count,y_single,label='Classical ML',c='#000080',marker='^',alpha=0.7,s=60)
plt.scatter(x_count,y_mo[:,2],label='ST multi outputs',c='#808080',marker='s',alpha=0.8,s=80)
plt.scatter(x_count,plot_list[:,2],label='MTRS',c='#FF1493',marker='d',alpha=0.7,s=60)

#plt.scatter(x_count,y_mo[:,2],label='ST multi outputs')
x_countt=np.array((0,0.04,0.08,0.16,0.24,0.32,0.40)).reshape(7,1)
#marker=7
plt.scatter(x_countt,y_data1[:,2].reshape(-1,1),c='#00FF00',label='Experimental',marker='*',s=140)
plt.xlabel('HAG-MP Concentration (wt%)',fontdict=xyfont,labelpad=20) # X-Label
plt.ylabel('T/D',fontdict=xyfont,labelpad=20) # Y-Label
plt.legend(loc='upper center',bbox_to_anchor=(0.83,1.01),fontsize=15)
ax=plt.gca()
ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=False,left=True,right=False
               ,labelsize=20,pad=12)
plt.grid(alpha=0.5,zorder=1)

name='Prediction_td.pdf'
plt.savefig(name,dpi=600,format='pdf')
#pad vaseye adad ha hast na chize dg
    
plt.show()



#============================================================================
'                             PREDICTION PLOT 2                         '
#============================================================================


#===============================================================
#===============================================================
#===============================================================
file1_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\plot_list_2' 
plot_all=np.array(pd.read_csv(file1_location,index_col=0))
plot_list=plot_all[:,0]


file2_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\yyy_2' 
yyy=np.array(pd.read_csv(file2_location,index_col=0))


file3_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\y_mo_2' 
y_mo=np.array(pd.read_csv(file3_location,index_col=0))

#j=50
#x_count=np.arange(j)/100
#x_count=x_count.reshape(-1,1)

plt.figure(figsize=(12, 8))  # Set the figure size (width: 10 inches, height: 6 inches)
plt.rcParams["figure.dpi"] = 600 
plt.rcParams['font.family']='Arial'
plt.scatter(x_count,y_mo,label='ST multi outputs',c='#808080',marker='s',alpha=0.6,s=100)
plt.scatter(x_count,yyy,label='Classical ML',c='#000080',marker='^',alpha=0.8,s=50)

plt.scatter(x_count,plot_list,label='MTRS',c='#FF1493',marker='d',alpha=0.7,s=60)
#inja bayad yebar hashtag dobar hastag bashe


x_countt=np.array((0,0.04,0.08,0.16,0.24)).reshape(5,1)
y5=np.array((0.419601,0.352938,0.241008,0.150044,0.196248)).reshape(5,1)
plt.scatter(x_countt,y5,c='#00FF00',label='Experimental',marker='*',s=140)
plt.xlabel('HAG-MP Concentration (wt%)',fontdict=xyfont,labelpad=20) # X-Label
plt.ylabel('Density difference at P=1 bar',fontdict=xyfont,labelpad=20) # Y-Label
plt.legend(loc='upper center',bbox_to_anchor=(0.23,1),fontsize=15)
ax=plt.gca()
ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=False,left=True,right=False,
               labelsize=20,pad=12)
plt.grid(alpha=0.5,zorder=1)

name='Prediction_density.pdf'
plt.savefig(name,dpi=600,format='pdf')
plt.show()



#===============================================================
#===============================================================
#===============================================================


def plot2(percentage,ii):
    plt.figure(figsize=(12, 8))  # Set the figure size (width: 10 inches, height: 6 inches)
    plt.rcParams["figure.dpi"] = 600 
    plt.rcParams['font.family']='Arial'


    y=plot_all[percentage]
    x = np.arange(1,1011)
    # plot() is used for plotting a line plot
    
    yy=np.array(data2_outputs)
    yy=yy[ii][:]
    
    plt.plot(x,yy,'r',linewidth=2,label='Experimental Value')
    plt.plot(x,y,'--',c='b',linewidth=2,label='Predicted Value')
    
    
    a='Density difference per pressure steps for '
    b=str(percentage/100)
    c=" HAG-MP Concentration (wt%)"
    titles=a+b+c
    #plt.title(titles) # Title of the plot
    #plt.xlabel('Pressure step (from 1 to 400 bar)',fontdict=xyfont,labelpad=20) # X-Label
    #plt.ylabel('Density difference',fontdict=xyfont,labelpad=20) # Y-Label
    #plt.legend(fontsize=20)
    ax=plt.gca()
    ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=False,left=True,right=False,
                   labelsize=20,pad=12)
    plt.ylim(0,0.15)
    plt.xlim(0,200) # 1 nmidonam
    
    name='zoom_Prediction_density_'+str(percentage)+'_.pdf'
    plt.savefig(name,dpi=600,format='pdf')
    plt.show()

    
plot2(0,0)
plot2(4,1)
plot2(8,2)
plot2(16,3)
plot2(24,4)



def plot2_exp_all():
    plt.figure(figsize=(12, 8))  # Set the figure size (width: 10 inches, height: 6 inches)
    plt.rcParams["figure.dpi"] = 600 
    plt.rcParams['font.family']='Arial'


    x = np.arange(1,1011)
    # plot() is used for plotting a line plot
    
    #yy=np.array(data2_outputs)
    #yy=yy[ii][:]
    plt.plot(x,data2_0,label='0 wt% HAG-MP')
    plt.plot(x,data2_4,label='0.04 wt% HAG-MP')
    plt.plot(x,data2_8,label='0.08 wt% HAG-MP')
    plt.plot(x,data2_16,label='0.16 wt% HAG-MP')
    plt.plot(x,data2_24,label='0.24 wt% HAG-MP')
    
  
   # a='Density difference per pressure steps for '
    #b=str(percentage/100)
    ##c=" HAG-MP Concentration (wt%)"
    #titles=a+b+c
    #plt.title(titles) # Title of the plot
    plt.xlabel('Pressure step (from 1 to 400 bar)',fontdict=xyfont,labelpad=20) # X-Label
    plt.ylabel('Density difference',fontdict=xyfont,labelpad=20) # Y-Label
    plt.legend(fontsize=20)
    ax=plt.gca()
    ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=False,left=True,right=False,
                   labelsize=20,pad=12)
    #plt.ylim(0,0.15)
    #plt.xlim(0,200) # 1 nmidonam
    
    name='all_exp_density_.pdf'
    plt.savefig(name,dpi=600,format='pdf')
    plt.show()

plot2_exp_all()


#============================================================================
'                             PREDICTION PLOT 3                         '
#============================================================================
xyfont={'family':'Arial','size':20,'fontweight':'bold'}

tfont={'family':'Arial','size':20}
lfont={'family':'Arial','size':20}
#===============================================================
#===============================================================
#===============================================================

file1_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\0-90\\y_single_3' 
y_single=np.array(pd.read_csv(file1_location,index_col=0))

file2_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\0-90\\y_multi_3' 
y_multi=np.array(pd.read_csv(file2_location,index_col=0))

file3_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\0-90\\y_mtrs_3'
y_mtrs=np.array(pd.read_csv(file3_location,index_col=0))

#j bayad age 0-50 50 bashe ag 0-90 90 bashe
j=90

x=x_data3/100
y=half_time
x_count=np.arange(j)/100
x_count=x_count.reshape(-1,1) 

plt.figure(figsize=(12, 8))  # Set the figure size (width: 10 inches, height: 6 inches)
plt.rcParams["figure.dpi"] = 600 
plt.rcParams['font.family']='Arial'

plt.scatter(x_count,y_single,label='Classical ML',c='#000080',marker='^',alpha=0.7,s=60)
plt.scatter(x_count,y_multi,label='ST multi outputs',c='#808080',marker='s',alpha=0.8,s=80)

plt.scatter(x_count,y_mtrs,label='MTRS',c='#FF1493',marker='d',alpha=0.7,s=60)
plt.scatter(x,y ,c='#00FF00',label='Experimental',marker='*',s=140)

plt.xlabel('HAG-MP Concentration (wt%)',fontdict=xyfont,labelpad=20) # X-Label
plt.ylabel('Half-life Time (min)',fontdict=xyfont,labelpad=20) # Y-Label
plt.legend(loc='upper center',bbox_to_anchor=(0.78,1),fontsize=15)
#plt.legend(loc='upper center',bbox_to_anchor=(0,1.2))
ax=plt.gca()
#ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=False,left=True,right=False,
               labelsize=20,pad=12)
plt.grid(alpha=0.5,zorder=1)

         
name='Prediction_halftime.pdf'
plt.savefig(name,dpi=600,format='pdf')
plt.show()




#===============================================================
#===============================================================
#===============================================================



file_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\0-90\\ydrain'
f1=open(file_location,'r')
data1=pd.read_csv(f1)
data1=data1.drop('Unnamed: 0',axis=1)
y1=np.array(data1)


file_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\plot\\0-90\\yfoam'
f1=open(file_location,'r')
data1=pd.read_csv(f1)
data1=data1.drop('Unnamed: 0',axis=1)
y2=np.array(data1)


def normal_vol(pp,ii):
    x = np.array((0,20,40,60,80,100,120,140,160,180,200,
                  220,240,260,280,300,320,340,360,380,400))
    drainn=np.array(y_data3_drain)
    foamm=np.array(y_data3_foam)
    f=foamm[ii]
    d=drainn[ii]


    drain=y1[pp][:]
    foam=y2[pp][:]

    drain=drain/max(drain)
    foam=foam/max(foam)
    f=f/max(f)
    d=d/max(d)

          
    plt.figure(figsize=(12, 8))  # Set the figure size (width: 10 inches, height: 6 inches)
    plt.rcParams["figure.dpi"] = 600 
    plt.rcParams['font.family']='Arial'

    plt.plot(x,drain,'--',label='ML Predicted',c='purple',alpha=0.6,linewidth=2)
    plt.plot(x,foam,'--',c='purple',alpha=0.6,linewidth=2)

    plt.scatter(x,d,label='Drainage Phase',marker='s',s=50)
    plt.scatter(x,f,label='MB Phase',s=50)




    #plt.title('Half time') # Title of the plot
    #plt.title('The Dr')
    plt.xlabel('Time (min)',fontdict=xyfont,labelpad=20) # X-Label
    plt.ylabel('Normalized Volume',fontdict=xyfont,labelpad=20) # Y-Label
    ax=plt.gca()
    ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=False,left=True,right=False,
                   labelsize=20,pad=12)

    #plt.ylim(0,1.1)
    plt.legend(loc='upper center',bbox_to_anchor=(0.85,0.65),fontsize=15)
    plt.grid(alpha=0.5,zorder=1)



    name='Prediction_'+str(pp)+'halftime.pdf'
    plt.savefig(name,dpi=600,format='pdf')
    plt.show()


normal_vol(0,0)   
normal_vol(4,1)
normal_vol(8,2) 
normal_vol(16,3)
normal_vol(24,4)
normal_vol(32,5)
normal_vol(40,6)    
