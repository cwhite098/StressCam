# import the necessary packages
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd

def nothing(x):
    pass

def get_hull(pts):
    # Function that gets the hull (outline) of a set of points
    hull = ConvexHull(pts)
    # Returns those points that form the outline of the shape, in the correct order.
    return pts[hull.vertices]


def save_data(HRM, HT, BMD, ET, RT, path):
    '''
    Function that takes all the metric-tracking classes and produces and saves a dataframe.
    '''
    print('SAVING DATA')

    # Get POS signal from S
    signal = np.transpose(HRM.totalPOS)
    coeff = (np.std(signal[0,:])/np.std(signal[1,:]))
    POS_signal = signal[0,:] + (coeff*signal[1,:])
    POS_signal = POS_signal - np.mean(POS_signal)

    df = pd.DataFrame()
    df['POS'] = POS_signal # this POS signal needs processing
    df['EYE_RATIO'] = BMD.ratios_list
    df['LEYEBROW_RATIO'] = BMD.l_eyebrow_ratio_list
    df['REYEBROW_RATIO'] = BMD.r_eyebrow_ratio_list
    df['MOUTH_RATIO'] = BMD.mouth_ratios_list
    df['HEAD_PITCH'] = HT.x_list
    df['HEAD_YAW'] = HT.y_list
    df['HEAD_TRANS'] = HT.translation_list

    df.to_csv(path)
