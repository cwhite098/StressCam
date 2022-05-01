import tensorflow as tf
import pandas as pd
import heartpy as hp
import numpy as np
import tsfresh
import pickle
import keras
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from timeit import default_timer as timer


class StressPredictor:
    def __init__(self, model_file_path):
        self.model = tf.keras.models.load_model(model_file_path)

    def pre_process(self, input_csv_path):
        """
        Takes the CSV and cleans it up, calculates heartrate features
        :param input_csv_path: input CSV
        :return: tuple containing time series dataframe and heartrate features
        """
        labels = ['POS', 'EYE_RATIO', 'LEYEBROW_RATIO', 'REYEBROW_RATIO',
                  'MOUTH_RATIO', 'HEAD_PITCH', 'HEAD_YAW', 'HEAD_TRANS', 'EYE_RATIO',
                  'RESP_SIGNAL']

        df = pd.read_csv(input_csv_path)
        X = np.array(df[labels])

        where_is_pinf = np.array(np.where(np.isposinf(X)))

        for i in range(len(where_is_pinf[0])):
            X[where_is_pinf[0, i], where_is_pinf[1, i], where_is_pinf[2, i]] = np.mean(
                (X[where_is_pinf[0, i] - 1, where_is_pinf[1, i], where_is_pinf[2, i]],
                 X[where_is_pinf[0, i] + 1, where_is_pinf[1, i], where_is_pinf[2, i]])
            )

        where_is_pinf = np.array(np.where(np.isnan(X)))
        for i in range(len(where_is_pinf[0])):
            X[where_is_pinf[0, i], where_is_pinf[1, i], where_is_pinf[2, i]] = np.mean(
                (X[where_is_pinf[0, i] - 1, where_is_pinf[1, i], where_is_pinf[2, i]],
                 X[where_is_pinf[0, i] + 1, where_is_pinf[1, i], where_is_pinf[2, i]])
            )

        where_is_pinf = np.array(np.where(np.isnan(X)))
        for i in range(len(where_is_pinf[0])):
            X[where_is_pinf[0, i], where_is_pinf[1, i], where_is_pinf[2, i]] = 0

        HRV_features = []
        # Bandpass filter the POS signal
        filtered = hp.filtering.filter_signal(X[:, 0], cutoff=(1, 2), sample_rate=35.0, order=3, filtertype='bandpass')
        # Extract the HRV measures
        try:
            working_data, measures = hp.process(filtered, sample_rate=35)
            for measure in measures.values():
                HRV_features.append(measure)
        except hp.exceptions.BadSignalWarning:
            measures = [0] * 12
            HRV_features = measures

        # Save the HRV measures for each example
        HRV_features.append(HRV_features)

        return pd.DataFrame(X), pd.DataFrame(HRV_features)

    def extract_features(self, X, HRV_features):
        """
        Takes an input timeseries csv and calculates the feature extraction
        :param X: time series dataframe
        :param HRV_features: Features from the HR extraction
        :return: features
        """

        # dictionary containing the features to be extracted
        params_file = 'model/params.pkl'
        with open(params_file, 'rb') as f:
            params = pickle.load(f)

        # names of the variables stored in the params dict
        var_names = [f'var_{i}' for i in range(10)]

        # Remove cols from dataframe if there are no selected features in those cols (VERY IMPORTANT)
        X.columns = var_names
        for col in X.columns:
            if col not in list(params.keys()):
                del X[col]

        t = TSFreshFeatureExtractor(kind_to_fc_parameters=params, show_warnings=False)
        X_fit = t.fit_transform(X)

        selected_HRV_features = [1, 3, 4, 8]
        HRV_features = pd.DataFrame(HRV_features[0][selected_HRV_features])

        # Add the selected HRV features to df
        feature_df = pd.concat([X_fit, HRV_features.transpose()], axis=1).astype('float32')

        return feature_df

    def predict(self, ts_file_path):
        X, HRV_features = self.pre_process(ts_file_path)
        feature_df = self.extract_features(X, HRV_features)
        prediction = self.model.predict(feature_df)
        return prediction


def main():
    p = StressPredictor('model/tf')
    prediction = p.predict('datas/vid_s8_T1.csv')
    print(prediction)
    # print(f'Timings\n'
    #       f'Pre-processing ------- {t2 - t1}\n'
    #       f'Feature Extraction --- {t3 - t2}\n'
    #       f'Prediction ----------- {t4 - t3}\n'
    #       f'==============================\n'
    #       f'Total----------------- {t4 - t1}')


if __name__ == '__main__':
    main()
