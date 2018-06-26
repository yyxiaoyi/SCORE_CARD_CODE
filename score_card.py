import numpy as np

P0 = 600
PDO = 40
B = PDO / np.log(2)
A = P0 + B * np.log(0.1)
    
def get_raw_score(feature_list, scorer_list):
    total_score = A - B * scorer_list['intercept']
    for feature_name, scorer in scorer_list.items():
        if feature_name=='intercept':
            continue
        idx = np.searchsorted(scorer['bins'], feature_list[feature_name])-1
        total_score = total_score - scorer['WOE'][idx] * scorer['theta']
    return total_score

if __name__ == '__main__':  
    scorer_list = {'f1':{
                    'bins':[-np.inf, -5, 0, 5, 10, np.inf],
                    'WOE':[2, -3, 4.5, 0, 0.67],
                    'theta':6.7
                    },
               'f2':{
                    'bins':[-np.inf, 2, 4, 6, 8, 10, np.inf],
                    'WOE':[1.2, -0.36, -3.1, 4.6, 0.15, 0.76],
                    'theta':-3.4
                    },
                'intercept':-35
              }
    feature_list = {'f1':5, 'f2':1.5}
    
    print('Output %.3f should be %.3f'%(get_raw_score(feature_list, scorer_list), base_score-4.5*6.7-1.2*(-3.4)))
  