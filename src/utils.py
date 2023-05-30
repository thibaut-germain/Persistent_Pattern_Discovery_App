import numpy as np
from scipy.signal import correlation_lags

def transfrom_label(L:np.ndarray)->list: 
    """Transfom binary mask to a list of start and ends

    Args:
        L (np.ndarray): binary mask, shape (n_label,length_time_series)

    Returns:
        list: start and end list. 
    """
    lst = []
    for line in L: 
        line = np.hstack(([0],line,[0]))
        diff = np.diff(line)
        start = np.where(diff==1)[0]+1
        end = np.where(diff==-1)[0]
        lst.append(np.array(list(zip(start,end))))
    return np.array(lst,dtype=object)


def pearson_correlation(s0,s1,wlen): 
    m = s0.shape[0]
    n = s1.shape[0]
    avg_coeff = np.convolve(np.ones(m),np.ones(n))
    mask = avg_coeff >= wlen
    dot_prod = np.convolve(s0,s1[::-1])/avg_coeff
    mean0 = np.convolve(s0,np.ones(n))/avg_coeff
    mean1 = np.convolve(np.ones(m),s1[::-1])/avg_coeff
    std0 = np.sqrt((np.convolve(s0**2,np.ones(n))-avg_coeff*mean0**2)/avg_coeff)
    std1 = np.sqrt((np.convolve(np.ones(m),s1[::-1]**2)-avg_coeff*mean1**2)/avg_coeff)
    pearson = (dot_prod - mean0*mean1)[mask]/(std0*std1)[mask]
    offset = correlation_lags(m,n)[mask]
    return pearson,offset

def get_optimal_lag(s0,s1,wlen): 
    corr,offsets = pearson_correlation(s0,s1,wlen)
    return offsets[np.argmax(corr)]

def get_optimal_lag_matrix(dataset,wlen=None): 
    n_ts = len(dataset)
    lags = np.zeros((n_ts,n_ts))
    if wlen is None: 
        wlen = np.mean([len(ts) for ts in dataset])//4
    for i,j in np.vstack(np.triu_indices(n_ts,1)).T: 
        lag = get_optimal_lag(dataset[i],dataset[j],wlen)
        lags[i,j] = lag
        lags[j,i] = -lag
    return lags

def get_relative_lag(dataset,wlen=None): 
    lags = get_optimal_lag_matrix(dataset,wlen)
    best_idx = np.argmin(np.sum(np.abs(lags),axis=0))
    return lags[best_idx]

def get_barycenter(dataset,lags): 
    lags = lags.copy()
    min_lag = np.min(lags)
    lags -= min_lag
    length = np.max(np.array([len(ts) for ts in dataset])+lags).astype(int)
    arr = np.zeros((len(dataset),length))
    for i,(lag,ts) in enumerate(zip(lags.astype(int),dataset)):
        arr[i, lag : lag+len(ts)] = ts
    avg_pattern = np.mean(arr,axis=0)
    x = np.arange(min_lag,min_lag + len(avg_pattern))
    return x,avg_pattern