"""Training-only numeric imputation, scaling and schema validation for flow data."""
import numpy as np
import pandas as pd
from .feature_processing import PureStandardScaler


class DataPreprocessor:
    """Reusable flow preprocessor. Pass features without the label column.

    Fit on the training partition only. Validation, test and inference use transform.
    Median values and feature order are retained; unexpected columns are rejected.
    """
    def __init__(self,config=None):
        self.config=config or {}
        self.scaler=None
        self.feature_names=None
        self.medians=None

    def _frame(self,data,fit=False):
        frame=pd.DataFrame(data).copy()
        if frame.empty:raise ValueError('Feature data must not be empty.')
        if frame.columns.duplicated().any():raise ValueError('Duplicate feature names.')
        if fit:self.feature_names=list(frame.columns)
        if set(frame.columns)!=set(self.feature_names):raise ValueError('Feature schema differs from the training schema.')
        frame=frame[self.feature_names].apply(pd.to_numeric,errors='raise')
        return frame.replace([np.inf,-np.inf],np.nan)

    def fit_transform(self,data):
        frame=self._frame(data,fit=True)
        self.medians=frame.median().fillna(0)
        self.scaler=PureStandardScaler()
        return self.scaler.fit_transform(frame.fillna(self.medians).to_numpy())

    def transform(self,data):
        if self.scaler is None:raise RuntimeError('Fit on training features before transforming.')
        frame=self._frame(data)
        return self.scaler.transform(frame.fillna(self.medians).to_numpy())
