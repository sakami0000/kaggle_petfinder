import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold


def run_lgb(params, X_train, y_train, X_test,
            n_splits=10, num_rounds=60000, early_stop=500, plot=True):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)

    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0]))
    feature_importances = np.zeros((X_test.shape[1]))

    for trn_idx, val_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):
        train_fold_x = X_train.iloc[trn_idx, :]
        train_fold_y = y_train[trn_idx]
        valid_fold_x = X_train.iloc[val_idx, :]
        valid_fold_y = y_train[val_idx]
        
        dtrain = lgb.Dataset(train_fold_x, train_fold_y)
        dvalid = lgb.Dataset(valid_fold_x, valid_fold_y, reference=dtrain)

        model = lgb.train(params, dtrain, num_boost_round=num_rounds, valid_sets=dvalid,
                          early_stopping_rounds=early_stop, verbose_eval=1000)

        valid_pred = model.predict(valid_fold_x, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)

        oof_train[val_idx] = valid_pred
        oof_test += test_pred / n_splits
        feature_importances += model.feature_importance(importance_type='gain') / n_splits
        
    print('feature importances:')
    for i in np.argsort(feature_importances):
        print(f'\t{model.feature_name()[i]:35s}: {feature_importances[i]:.1f}')
    
    if plot:
        feature_imp = pd.DataFrame(sorted(zip(feature_importances, model.feature_name())),
                                   columns=['value', 'feature'])
        
        plt.figure(figsize=(22, 44))
        sns.barplot(x='value', y='feature',
                    data=feature_imp.sort_values(by='value', ascending=False))
        plt.title('LightGBM Feature Importances')
        plt.tight_layout()
        plt.show()

    return oof_train, oof_test
