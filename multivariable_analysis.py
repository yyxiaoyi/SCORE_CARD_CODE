
import numpy as np, pandas as pd

def varsel(data, IV, fnames, corr_th=0.6):
    if len(fnames) != len(IV) or np.shape(data)[1]!=len(fnames):
        raise ValueError('Invalid input!')
    
    corr = pd.DataFrame(data, columns=fnames).corr()  

    found_list = dict.fromkeys(fnames, [])
    for i in range(len(fnames)):
        crow = corr.loc[fnames[i],:]
        # scanning this row
        for j in range(len(fnames)):
            if i!=j and np.abs(crow[j])>=corr_th:
                found_list[fnames[i]]=found_list[fnames[i]]+[fnames[j]]
        
    idx = np.argsort(IV)[::-1]
    keep_list=[]
    remove_list=[]
    for f in np.take(fnames, idx):
        if f not in remove_list:
            keep_list.append(f)
            remove_list+=found_list[f]
            print('Keep %s, remove'%f, found_list[f])

    return keep_list

if __name__ == '__main__':   
    
    test_data=np.random.randn(100,5)
    test_IV=np.random.randn(5)
    fnames=['c'+str(i) for i in range(0,5)]

    varsel(test_data, test_IV, fnames, corr_th=0.1)