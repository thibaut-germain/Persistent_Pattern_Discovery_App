### Heuristics to compute persitence and birth threshold ### 

import numpy as np
from scipy.stats import norm

def _infinite_point_treatement(persistence:np.ndarray)->np.ndarray:
    """Change death value of infinite point to the maximum death observed

    Args:
        persistence (np.ndarray): persisence attribute of a persistence class

    Returns:
        np.ndarray: transfromed persistence
    """
    mask = persistence[:,1] == np.inf
    if np.any(mask):
        max_death = np.max(persistence[np.invert(mask),1])
        t_persistence = persistence.copy()
        t_persistence[mask,1] = max_death
        return t_persistence
    else: 
        return persistence



def _jump_cut(vector : np.ndarray, offset = 1):
    """set the cut to the maximum jump

    Args:
        vector (np.ndarray): _description_
    """
    arr = np.sort(vector)
    thresholds = np.diff(arr)
    threshold_idx = np.argmax(thresholds)+offset
    return arr[threshold_idx]


def _basic_otsu(vector: np.ndarray,nbins=1024): 
    """Compute Otsu threshold

    Args:
        vector (np.ndarray): pixel array. shape: (N,)
        nbins (int, optional): number of bins for the histogram. Defaults to 1024.
    """

    #intialisation
    count, values = np.histogram(vector,nbins)
    count = count.astype(float)/vector.size
    wl = 0
    wr = 1
    ml = 0
    mr = np.mean(vector)
    lst =[]
    #loop
    for prob,value in zip(count[:-1],values[:-1]): 
        ml = (ml*wl + prob*value)/(wl + prob)
        mr = (mr*wr - prob*value)/(wr - prob)
        wl += prob
        wr -= prob
        var = wl*wr*(ml-mr)**2
        lst.append(var)

    return values[np.argmax(lst)]

def otsu_jump(X:np.ndarray,jump=1,nbins=1024)->tuple:
    """Compute persistance cut and birth cut.

    Args:
        X (np.ndarray): persistance array from fresistence module. 
        threshold (float, optional): threshold for outlier. Defaults to 0.95.
        nbins (int, optional): number of bins for otsu. Defaults to 1024.

    Returns:
        tuple: persistance cut, birth cut.
    """
    births = X[:,0]
    b_cut = _basic_otsu(births,nbins)
    #b_cut = _basic_otsu(births[births<b_cut],nbins)
    pers = np.diff(X[X[:,0]<b_cut],axis=1).reshape(-1)
    pers = pers[pers>0]
    pers = np.sort(pers)[::-1]
    diff = pers[:-1] - pers[1:]
    for _ in range(jump):
        idx = np.argmax(diff)
        diff[:idx+1] = -np.inf
    p_cut = (pers[idx]+pers[idx+1])/2
    return p_cut,b_cut

    

