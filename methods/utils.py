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

    Parameters
    ----------
    HRM : heart rate monitoring class

    HT : head tracing class

    BMD : blink and mouth detection class

    ET : eye tracking class

    RT : respiration rate tracking class

    path : string
        The path the save the data to.
    '''
    print('SAVING DATA')

    # Get POS signal from S
    signal = np.transpose(HRM.totalPOS)
    coeff = (np.std(signal[0,:])/np.std(signal[1,:]))
    POS_signal = signal[0,:] + (coeff*signal[1,:])
    POS_signal = POS_signal - np.mean(POS_signal)

    eye_array = ET.get_history()

    df = pd.DataFrame()
    df['POS'] = POS_signal # this POS signal needs processing
    df['EYE_RATIO'] = BMD.ratios_list
    df['LEYEBROW_RATIO'] = BMD.l_eyebrow_ratio_list
    df['REYEBROW_RATIO'] = BMD.r_eyebrow_ratio_list
    df['MOUTH_RATIO'] = BMD.mouth_ratios_list
    df['HEAD_PITCH'] = HT.x_list
    df['HEAD_YAW'] = HT.y_list
    df['HEAD_TRANS'] = HT.translation_list
    
    l_eye = np.array(eye_array)[0]
    r_eye = np.array(eye_array)[1]

    df['LEYE_X'] = l_eye[:,0]
    df['LEYE_Y'] = l_eye[:,1]
    df['REYE_X'] = r_eye[:,0]
    df['REYE_Y'] = r_eye[:,1]

    df['RESP_SIGNAL'] = RT.p_norm

    df.to_csv(path)


def get_summary_stats(df, labels, range=None):
    """
    Calculates some summary stats for the timeseries'
    :param: df = df containing the whole dataset for a sensor
    :param: labels = the variables to get the stats for
    :returns: stats_df = pandas dataframe containing the stats
    """
    stats_df = np.array(['Label', 'Mean', 'Max', 'Min', 'Std'])

    for label in labels:
        timeseries = df[label]

        stats = np.array([label, np.mean(timeseries), np.max(timeseries), np.min(timeseries), np.std(timeseries)])
        stats_df = np.vstack((stats_df, stats))

    return pd.DataFrame(data=stats_df[1:, :], columns=stats_df[0, :])
