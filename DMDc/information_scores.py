import numpy as np
from scipy.ndimage import gaussian_filter

def KL_divergence(x, y, sigma=1):
    # histogram
    hist_xy = np.histogram2d(x, y, bins=100)[0]

    # smooth it out for better results
    gaussian_filter(hist_xy, sigma=sigma, mode='constant', output=hist_xy)

    EPS = 1e-3
    # compute marginals
    hist_xy = hist_xy + EPS # prevent division with 0
    hist_xy = hist_xy / np.sum(hist_xy)
    hist_x = np.sum(hist_xy, axis=0)
    hist_y = np.sum(hist_xy, axis=1)

    kl = -np.sum(hist_x * np.log(hist_y / hist_x ))
    return kl

def AIC(model,data,k):
    n = len(data)
    RSS = np.linalg.norm(model-data)**2
    sigma2 = RSS / n
    logL = - n* np.log(2*np.pi) / 2 - n * np.log(sigma2) - n / 2
    return 2*k - 2*logL

def BIC(model,data,k):
    n = len(data)
    RSS = np.linalg.norm(model-data)**2
    sigma2 = RSS / n
    logL = - n* np.log(2*np.pi) / 2 - n * np.log(sigma2) - n / 2
    return np.log(n) * k - 2 * logL






    
