from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor

from ..nn.models import NNRegressor
from .stacker import RegressionStacker


def train_stack(x_train, y_train, x_test,
                lgb_params_1={}, lgb_params_2={}, nn_params={},
                n_splits=5, seed=42):
    lgb_model_1 = LGBMRegressor(**lgb_params_1)
    lgb_model_2 = LGBMRegressor(**lgb_params_2)
    nn_model = NNRegressor(x_train.shape[1], **nn_params)

    log_model = LinearRegression()
    et_model = ExtraTreesRegressor(n_estimators=100,
                                   max_depth=6,
                                   min_samples_split=10,
                                   random_state=seed)
    mlp_model = MLPRegressor(max_iter=7, random_state=seed)

    stack = NNRegressor(mode='woc',
                        n_splits=3,
                        stacker_2=(log_model, et_model),         
                        stacker_1=(log_model, et_model, mlp_model),
                        base_models=(lgb_model_1, lgb_model_2, nn_model))       

    train_preds, y_pred = stack.fit_predict(x_train, y_train, x_test)

    return train_preds, y_pred
