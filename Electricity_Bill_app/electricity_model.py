# electricity_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

class ElectricityUsageModel:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
        self.user_encoder = LabelEncoder()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.feature_names = []
        self.lag_lookup = {}
        self.df_cluster_map = None

    def preprocess_and_train(self, df):
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month_Num'].astype(str).str.zfill(2) + '-01')
        df.sort_values(by=['UserID', 'Date'], inplace=True)

        df['UserID_encoded'] = self.user_encoder.fit_transform(df['UserID'])
        df['is_summer'] = df['Month_Num'].isin([5, 6, 7, 8]).astype(int)
        df['is_winter'] = df['Month_Num'].isin([12, 1, 2]).astype(int)
        df['prev_units'] = df.groupby('UserID')['Units'].shift(1)
        df['diff_from_last_month'] = df['Units'] - df['prev_units']
        df.dropna(inplace=True)

        user_avg_units = df.groupby('UserID')['Units'].mean().reset_index()
        user_avg_units.columns = ['UserID', 'AvgUnits']
        user_avg_units['Cluster'] = self.kmeans.fit_predict(user_avg_units[['AvgUnits']])
        df = df.merge(user_avg_units[['UserID', 'Cluster']], on='UserID', how='left')
        self.df_cluster_map = user_avg_units[['UserID', 'Cluster']]
        self.lag_lookup = df.groupby('UserID')['Units'].last().to_dict()

        self.feature_names = ['UserID_encoded', 'Year', 'Month_Num', 'is_summer', 'is_winter',
                              'prev_units', 'diff_from_last_month', 'Cluster']
        X = df[self.feature_names]
        y = df['Units']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
        print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
        print("R²:", round(r2_score(y_test, y_pred), 4))

    def get_feature_importance(self):
        return dict(zip(self.feature_names, self.model.feature_importances_))
