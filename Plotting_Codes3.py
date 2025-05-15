'''
INSTRUCTION:
    
First you must run FINAL_8JUNE.py and all of that but plot_prediction.
we have 3 type ( 'Single' , 'Multi' , 'MTRS or RCC')
For each section we have 1_ , 2_ and 3_ 
and because we have 4 titsc ( MAPE,MSE,MAE,RMSE)

actually we have 36 file, we must first draw all 36 file heatmap

then among all of that which one 6 of the best score, 
we must adjust ind and col we have 3 figure

then among all of that the we choose the best model and then we insert that in
the FINAL_8JUNE model and in PLOT_PREDICTION def


'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

def plot_score(NAME,which,titsc='MAPE',indcol='F',ind=[0,0,0,0,0,0],col=[0,1,2,3,4,5],
               lable_name=['LR','KNN','DT','RF','SVR','MLP']):

    file2_location = 'C:\\Users\\sunhouse\\' + NAME+'_'+titsc + '.csv' 
    f2=open(file2_location,'r')
    final=pd.read_csv(f2)
    final=final.set_index('Unnamed: 0')
    if which=='table':
        if 'Single' in NAME:
            tit='Single output '+titsc+' score'    
        if 'Multi' in NAME:
            tit='Seperate Multi Output '+titsc+ ' scores'  
        if 'MTRS' in NAME:
            tit='Multi Target Stacking (MTSR) ' +titsc + ' scores'
        if 'RCC' in NAME:
            tit='Regression Chain Corrected (RCC) ' +titsc + ' scores'
        val1=list(final.columns)
        val2=list(final.index)
        val3=final.values
        fig, ax = plt.subplots() 
        ax.set_axis_off() 
        table = ax.table( 
            cellText = val3,  
            rowLabels = val2,  
            colLabels = val1, 
            rowColours =["mediumpurple"]*len(val2),  
            colColours =["mediumpurple"]*len(val1), 
            cellLoc ='center',  
            loc ='center')         
        ax.set_title(tit, 
                     fontweight ="bold") 
        plt.show()

    if which=='heatmap':
        #'coolwarm'
        sns.heatmap(final, cmap="viridis", annot=True, fmt='.2f')

        #sns.heatmap(final, cmap="viridis", annot=True, fmt='.2f')
        #sns.heatmap(final, cmap="flare", annot=True, fmt='.2f')

        #a=sns.color_palette("light:#5A9", as_cmap=True)
        #sns.heatmap(final, cmap=a, annot=True, fmt='.2f')

        #a=sns.color_palette("blend:#7AB,#EDA", as_cmap=True)
        #sns.heatmap(final, cmap=a, annot=True, fmt='.2f')
    if which=='spider':
        if 'Single' in NAME:
            settitle='SPYDER Plot for Classic Single-Target model' 
        if 'Multi' in NAME:
            settitle='SPYDER Plot for Seperate Multi outputs model'
        if 'MTRC' in NAME:
            settitle='SPYDER Plot for MTRS model'
        if 'RCC' in NAME:
            settitle='SPYDER Plot for RCC model'
      
        labels = ['RMSE', 'MSE', 'MAE', 'MAPE']
        label=['LR','KNN','DT','RF','SVR','MLP']
        if indcol=='F':
            #flable=label
            des_list=[]
            #tit=['MAPE','MAE','MSE'+'RMSE']
            for i in range(0,4):
                file2_location = 'C:\\Users\\sunhouse\\' + NAME + '_'+labels[i]+'.csv' 
                f2=open(file2_location,'r')
                final=pd.read_csv(f2)
                final=final.set_index('Unnamed: 0')
                ff=list[final]
                des_list.append(ff)   
        if indcol=='T':
            #ind=[1,1,2,3,3,5]
            #col=[1,4,0,3,4,5]
            des_list=[]
            for i in range(0,4):
                file2_location ='C:\\Users\\sunhouse\\'+NAME+'_'+labels[i]+'.csv' 
                f2=open(file2_location,'r')
                final=pd.read_csv(f2)
                final=final.set_index('Unnamed: 0')
                for i in range(0,6):
                    ii=ind[i]
                    ii=label[ii]
                    cc=col[i]
                    cc=label[cc]
                    dd=final[cc][ii]
                    des_list.append(dd)

        data1=list((des_list[0],des_list[6],des_list[12], des_list[18]))
        data2 =list((des_list[1],des_list[7],des_list[13], des_list[19]))
        data3=list((des_list[2],des_list[8],des_list[14],des_list[20]))
        data4=list((des_list[3],des_list[9],des_list[15],des_list[21]))
        data5=list((des_list[4],des_list[10],des_list[16],des_list[22]))
        data6=list((des_list[5],des_list[11],des_list[17],des_list[23]))
        #angles=['c1','c2','c3','c4','c5','c6','c7']
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        data1 = np.concatenate((data1,[data1[0]]))
        data2 = np.concatenate((data2,[data2[0]]))
        data3 = np.concatenate((data3,[data3[0]]))
        data4 = np.concatenate((data4,[data4[0]]))
        data5 = np.concatenate((data5,[data5[0]]))
        data6 = np.concatenate((data6,[data6[0]]))

        angles = np.concatenate((angles,[angles[0]]))
        fig = plt.figure()
        #if it is False it can be a good but it must not connected
        ax = fig.add_subplot(111, polar=True)
        ax.plot()
        ax.plot(angles, data6, 'o-', linewidth=1,markersize=3,label=lable_name[5])
        ax.plot(angles, data5, 'o-', linewidth=1,markersize=3,label=lable_name[4])
        ax.plot(angles, data3, 'o-', linewidth=1,markersize=3,label=lable_name[2])
        ax.plot(angles, data4, 'o-', linewidth=1,markersize=3,label=lable_name[3])
        ax.plot(angles, data2, 'o-', linewidth=1,markersize=3,label=lable_name[1])
        ax.plot(angles, data1, 'o-', linewidth=1,markersize=3,label=lable_name[0])

        ax.fill(angles, data6, alpha=0)
        ax.fill(angles, data5, alpha=0)
        ax.fill(angles, data1, alpha=0)
        ax.fill(angles, data2, alpha=0)
        ax.fill(angles, data3, alpha=0)
        ax.fill(angles, data4, alpha=0)
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        ax.set_thetagrids(angles*180/np.pi, labels)
        ax.set_title(settitle)
        ax.legend(loc='lower left')
        plt.show()
    if which=='Linear':
        data1 = [0.77,0.93,0.80,0.6,0.70,0.4,0.3]
        data2=[0.8,0.45,0.33,0.45,0.8,0.3,0.5]
        data3=[0.54,0.83,0.73,0.1,0.8,0.2,0.6]
        data4=[0.95,0.97,1,1,0.89,0.91,0.93]
        data5=[0.37,0.4,0.2,1,0.49,0.71,0.13]
        data6=[0.35,0.21,1,0.53,0.79,0.41,0.1]
        angles=['rang1','range2','range3','range4','range5','range6','range7']
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=False)
        ax.plot()
        ax.plot(angles, data6, 'o-', linewidth=1,markersize=3,label='MLP')
        ax.plot(angles, data5, 'o-', linewidth=1,markersize=3,label='SVR')
        ax.plot(angles, data3, 'o-', linewidth=1,markersize=3,label='RF')
        ax.plot(angles, data4, 'o-', linewidth=1,markersize=3,label='DT')
        ax.plot(angles, data2, 'o-', linewidth=1,markersize=3,label='KNN')
        ax.plot(angles, data1, 'o-', linewidth=1,markersize=3,label='LR')
        ax.set_title('Cross-validation score : '+titsc)
        ax.legend()
        plt.show()

    
def plot_prediction1(j=50):
    
    x_data1=np.array((0,4,8,16,24,32,40)).reshape(7,1)
    y_data1=np.array([(83.2,10.2,0.122596),(65.8,11.3,0.171733),(49.7,11.9,0.239437),
                    (39.0,12.6,0.323077),(43.2,12.8,0.296296),(47.5,12.5,0.263158),
                    (62.6,11.7,0.186901)])
    
    x=x_data1/100
    x_count=np.arange(j)/100
    x_count=x_count.reshape(-1,1)
    
    
    
    #maybe need final=final.set_index('Unnamed: 0')
    
    file1_location = 'C:\\Users\\sunhouse\\y_single_1.csv' 
    f1=open(file1_location,'r')
    y_single=np.array(pd.read_csv(f1))
    
    file2_location = 'C:\\Users\\sunhouse\\plot_list_1.csv' 
    f2=open(file2_location,'r')
    plot_list=np.array(pd.read_csv(f2))
    
    file3_location = 'C:\\Users\\sunhouse\\y_mo_1.csv'
    f3=open(file3_location,'r')
    y_mo=np.array(pd.read_csv(f3))
    
    
    plot_two=plot_list[:,1]/plot_list[:,0]


    y_mo_two=y_mo[:,1]/y_mo[:,0]
    
   
    plt.scatter(x_count,plot_two,label='Tpred/Dpred MTRS')
    plt.scatter(x_count,plot_list[:,2],label='T/D predicted MTRS')
    plt.scatter(x_count,y_single,label='Classic Machine learning')
    plt.scatter(x_count,y_mo[:,2],label='T/D pred ST')
    #dota balae shabihe hame
    plt.scatter(x_count,y_mo_two,label='Tpred/Dpred ST')

    plt.scatter(x,y_data1[:,2].reshape(-1,1),c='k',label='Experimental data')
    plt.xlabel('HAG-MP wt%') # X-Label
    plt.ylabel('Thickness / Diameter') # Y-Label
    plt.legend(loc='lower center')
    plt.show()
    
def plot_prediction2(which,range_number=50,percentage=0):
    #maybe need final=final.set_index('Unnamed: 0')
    file0_location = 'C:\\Users\\sunhouse\\.spyder-py3\\codha\\projects\\exel\\finalcsv.csv'
    f0=open(file0_location,'r')
    data2=pd.read_csv(f0)
    data2.drop(1010,axis=0,inplace=True)
    x_data2=np.array((0,4,8,16,24)).reshape(5,1)
    data2_0 = data2['ob'] - data2['0f']
    data2_4=data2['4b']-data2['4f']
    data2_8=data2['8b']-data2['8f']
    data2_16=data2['16b']-data2['16f']
    data2_24=data2['24b']-data2['24f']
    data2_output=pd.concat([data2_0,data2_4,data2_8,data2_16,data2_24],axis=1)
    data2_outputs=data2_output.T

    file1_location = 'C:\\Users\\sunhouse\\plot_list_2.csv' 
    f1=open(file1_location,'r')
    plot_list=np.array(pd.read_csv(f1))
    
    file2_location = 'C:\\Users\\sunhouse\\yyy_2.csv' 
    f2=open(file2_location,'r')
    yyy=np.array(pd.read_csv(f2))
    
    file3_location = 'C:\\Users\\sunhouse\\plot_list_order_2.csv'
    f3=open(file3_location,'r')
    plot_list_order=np.array(pd.read_csv(f3))
        
    file4_location = 'C:\\Users\\sunhouse\\plot_list_nocorrectr_2.csv' 
    f4=open(file4_location,'r')
    plot_list_nocorrectr=np.array(pd.read_csv(f4))
    
    file5_location = 'C:\\Users\\sunhouse\\y_mo_2.csv' 
    f5=open(file5_location,'r')
    y_mo=np.array(pd.read_csv(f5))


    j=range_number
    #plot_list=prediction_section2(rc_model,'p',j)
    if which=='omega':
        #single and other
        x=x_data2/100
        y=data2_outputs
        x_count=np.arange(j)/100
        x_count=x_count.reshape(-1,1) 
        #yyy=single_section2(single_model,'p',j).reshape(1,-1)
        plot_list=plot_list[:,0]
       # plot_list_order=prediction_section2(rc_model,'o',j)[:,0]
        #plot_list_nocorrectr=prediction_section2(rc_model,'noC',j)[:,0]
       # y_mo=multi_section2(model,'p',j)[:,0]
        plt.scatter(x_count,plot_list,label='RCC Left Direction')
        plt.scatter(x_count,plot_list_order,label='RCC Right Direction')
        plt.scatter(x_count,plot_list_nocorrectr,label='RC (not corrected)')
        plt.scatter(x_count,yyy,label='Classic Machine learning')
        plt.scatter(x_count,y_mo ,label='ST')
        plt.scatter(x,data2_outputs[:,0].reshape(-1,1),c='k',label='Experimental')
        plt.xlabel('HAG-MP wt%') # X-Label
        plt.ylabel('Density Differences at P=0 ') # Y-Label
        plt.legend(loc='lower center')
        plt.show()
        
    if which=='ALL':
        
        #inja msihe for gozasht va hame ro hesab krd
        y=plot_list[percentage]
        x = np.arange(1010)
        # plot() is used for plotting a line plot
        plt.plot(x,y)
        # Adding title, xlabel and ylabel
        a='Dencity difference per pressure steps for'
        b=str(percentage)
        c="wt %"
        titles=a+b+c
        plt.title(titles) # Title of the plot
        plt.xlabel('Pressure step(from 1 to 400 BAR)') # X-Label
        plt.ylabel('Dencity difference') # Y-Label
        plt.ylim(0,0.5)

        plt.show()
            

def plot_prediction3(which,range_number=50,percentage=0):

    x_data3=np.array((0,4,8,16,24,32,40)).reshape(7,1)
    half_time=np.array((58,87.5,200,280,260,210,176)).reshape(7,1)
    
    file1_location = 'C:\\Users\\sunhouse\\y_single_3.csv' 
    f1=open(file1_location,'r')
    y_single=np.array(pd.read_csv(f1))
    
    file2_location = 'C:\\Users\\sunhouse\\y_multi_3.csv' 
    f2=open(file2_location,'r')
    y_multi=np.array(pd.read_csv(f2))
    
    file3_location = 'C:\\Users\\sunhouse\\y_rc_3.csv'
    f3=open(file3_location,'r')
    y_rc=np.array(pd.read_csv(f3))
    
    
    
    
    j=range_number
    if which=='T1/2':

        x=x_data3/100
        y=half_time
        x_count=np.arange(j)/100
        x_count=x_count.reshape(-1,1) 
       # y_single=single_section3(single_model,'p',j).reshape(1,-1)
       # pre_y_multi=multi_section3(model,'p',j)[1]
      #  y_multi=half_timee(pre_y_multi)
      #  pre_y_rc=prediction_section3(rc_model, 'p',j)[1]
       # y_rc=half_timee(pre_y_rc)
  
        plt.scatter(x_count,y_rc,label='RCC')
        plt.scatter(x_count,y_multi,label='ST')
        plt.scatter(x_count,y_single,label='classic Machine learning')
        plt.scatter(x,y ,c='k',label='Experimental data')

        plt.xlabel('HAG-MP wt%') # X-Label
        plt.ylabel('Half-Time') # Y-Label
        plt.legend(loc='lower center')
        plt.show()
        
    if which=='drain_foam':
        
        file4_location = 'C:\\Users\\sunhouse\\yfoam_'+str(percentage) +'.csv' 
        f4=open(file4_location,'r')
        yfoam=np.array(pd.read_csv(f4))
        
        file5_location = 'C:\\Users\\sunhouse\\ydrain_'+str(percentage) +'.csv' 
        f5=open(file5_location,'r')
        ydrain=np.array(pd.read_csv(f5))
        
        file6_location = 'C:\\Users\\sunhouse\\t_'+str(percentage) +'.csv' 
        f6=open(file6_location,'r')
        t=np.array(pd.read_csv(f6))
        
        #foam=prediction_section3(rc_model, 'p',j)[0]
       # drain=prediction_section3(rc_model, 'p',j)[1]
       # t=half_timee(drain)
       # yfoam=foam[percentage,:]
       # ydrain=drain[percentage,:]
        n=np.array(range(0,401,20))
        
        plt.scatter(n,yfoam)
        plt.scatter(n,ydrain)
        lg='T 1/2 : ' + str(t)
        plt.legend(lg,loc='Upper Right')
        plt.xlabel('TIME') # X-Label
        plt.ylabel('Normalized') # Y-Label
        plt.show()    
        
        
