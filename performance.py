import pandas as pd, numpy as np
import matplotlib.pyplot as plt 

def KS(proba, actual, n_bins=10, draw=True): 
    df = pd.DataFrame({'proba':proba, 'target':actual})
    df['target'] = df['target'].map({0:'good',1:'bad'}) 
    
    # add random noise to duplicated values
    v_count = df.proba.value_counts()
    if (v_count>=int(len(df)*1./n_bins)).any()==True:
        v_count = v_count[v_count>=int(len(df)*1./n_bins)].index.values
        for v in v_count:
            df.loc[df.proba==v,['proba']]=df[df.proba==v].proba\
                            .apply(lambda x: x-np.random.randint(100)/100000)
    
    # 分段 
    df = df.sort_values(by='proba',ascending=False) 
    df['bin_id'] = [int(i*n_bins/len(df)) for i in range(len(df))]   
    
    # 计算区间边界
    split_points = sorted(np.append(df.groupby('bin_id').proba.max().values, 0))  
    df['bin_id']=pd.cut(df.proba, bins=split_points)

    bin_count = df.groupby(['bin_id','target']).proba.count()
    bin_count = bin_count.unstack(level=1, fill_value=.0).sort_index(ascending=False)
    cum_count = bin_count.cumsum(axis=0) 
    
    # 计算KS 
    cum_count = cum_count.divide(bin_count.sum(axis=0).values,axis=1) 
    cum_count['KS']=cum_count['good']-cum_count['bad'] 
    ks = cum_count['KS'].max()
    
    if draw == True:
        fig = cum_count[['good','bad']].plot(kind='line',grid=True, rot=45, xticks=range(n_bins),\
                                            yticks=np.linspace(0,1,11), title='KS = %.4f'%ks)
        fig.plot()
        fig.plot([0,n_bins-1],[0,1], color='g', linestyle='dashed', label='random')
        fig.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
        
    return {'KS':ks, 'cumcount':cum_count, 'count':bin_count}

def confusion_matrix(pre, act):
    df = pd.DataFrame({'predict':pre, 'actual':act})
    cm = df.groupby('predict').actual.value_counts().unstack(level=1) 
    return cm

def ROC(proba, actual, draw=True):
    cutoff_list = np.linspace(0, 1, 1001)
    tpr = []
    fpr = []
    for cutoff in cutoff_list:
        cut = np.greater_equal(proba, cutoff)
        tpr.append(np.sum(cut & np.equal(actual, 1)))
        fpr.append(np.sum(cut & np.equal(actual, 0)))
    tpr = np.divide(tpr, np.sum(actual))
    fpr = np.divide(fpr, len(actual)-np.sum(actual))
    auc = np.sum(((tpr[:-1]+tpr[1:])/2)*0.001) #sum of height*width
    
    if draw==True:
        plt.plot(fpr, tpr, label='ROC')
        plt.plot([0,1], [0,1], color='g', linestyle='dashed', label='random')
        plt.xticks(np.linspace(0,1,11))
        plt.yticks(np.linspace(0,1,11))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('AUC = %.4f'%auc)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
        plt.show()
        
    return {'AUC':auc, 'ROC':pd.DataFrame({'tpr':tpr,'fpr':fpr},index=cutoff_list)}
    