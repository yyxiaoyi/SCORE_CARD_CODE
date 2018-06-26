import re
import pandas as pd, numpy as np
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier

def cut(arr, method='equal_depth', n_bins=10, bin_edges=None, target=None):
    if len(arr)<50:
        raise ValueError('The sample is too small!')
    
    # get bin edges
    if method=='equal_width':
        _, bin_edges = pd.cut(arr, bins=n_bins, retbins=True)
    elif method=='tree':
        if target is None or len(arr)!=len(target):
            raise ValueError('Invalid target labels!')
        # pick out not-null values
        idx = np.argwhere(np.isnan(arr)==False).flatten() 
        tf = DecisionTreeClassifier(random_state=11, max_depth=3, min_samples_leaf=0.1) \
                                    .fit(arr[idx].reshape((len(idx),1)), target[idx])
        tf_model = export_graphviz(tf, out_file=None)
        tf_splitter = [float(re.search('-?\d+\.?\d*', s).group(0)) for s in re.findall('<=\s*-?\d+\.?\d*', tf_model)]
        bin_edges = np.sort(tf_splitter+[-np.inf, np.inf])
    elif method=='bin_edges':    
        if bin_edges is None or len(bin_edges)<3:
            raise ValueError('Invalid bin edges!')    
    else: 
        _, bin_edges = pd.qcut(arr, q=n_bins, retbins=True, duplicates='drop')       
        
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    return pd.cut(arr, bin_edges, retbins=True)

def WOE(arr, target, include_null_group=False):
    df = pd.DataFrame({'data':arr, 'label':target})
    if include_null_group==True:
        if df.data.dtype.name == 'category':
            df.data = df.data.cat.add_categories('null group')
        df.data = df.data.fillna('null group')
    p = df.groupby('label').data.value_counts(normalize=True, ascending=True).unstack(level=0, fill_value=1.0/max(len(arr),10000))
    p['WOE'] = np.log(p).diff(axis=1).dropna(axis=1) # log(p=1)-log(p=0)
    p['IV'] = (p[1]-p[0])*p['WOE']
    return {'IV':p['IV'].sum(), 'WOE':p['WOE']}

def PSI(arr_new, arr_old, include_null_group=False):
    arr = np.append(arr_new, arr_old)
    target = np.append(np.ones(arr_new.shape), np.zeros(arr_old.shape))
    
    return WOE(arr, target, include_null_group)['IV']


if __name__ == '__main__':    

    # testing WOE
    #连续变量
    sample = np.random.randint(-5, 11, 1000)
    dst_target = np.random.randint(0,2,1000)
    dst_sample, splitter = cut(sample, method='tree', target=dst_target)
    print(WOE(dst_sample, dst_target, include_null_group=True))
    print(WOE(dst_sample, dst_target, include_null_group=False))

    #类别变量
    cat_sample = np.array(['CAT-1','CAT-2','CAT-3','CAT-4'])[np.random.randint(0,4,1000)].tolist()
    cat_sample[-1]=np.nan
    cat_target = np.random.randint(0,2,1000)
    print(WOE(cat_sample, cat_target, include_null_group=True))
    print(WOE(cat_sample, cat_target, include_null_group=False))

    #离散变量
    idx_sample = np.random.randint(0,4,1000).tolist() # np.nan是float型
    idx_sample[-1]=np.nan
    idx_target = np.random.randint(0,2,1000)
    print(WOE(idx_sample, idx_target, include_null_group=True))
    print(WOE(idx_sample, idx_target, include_null_group=False))