import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold


class RegressionStacker(object):    
    def __init__(self, base_models, stacker_1, stacker_2=None,
                 n_splits=5, metrics='rmse', mode='woc', random_state=None):
        assert metrics in ('rmse', 'mse')
        assert mode in ('oof', 'woc')

        self.base_models = base_models
        self.stacker_1 = stacker_1
        self.stacker_2 = stacker_2
        self.n_splits = n_splits
        self.mode = mode
        self.seed = random_state

        if metrics == 'rmse':
            self.eval = self.rmse
        elif metrics == 'mse':
            self.eval = mean_squared_error

        if isinstance(self.stacker_2, (list, tuple)):
            self.num_models = len(self.stacker_2)
            if self.num_models == 1:
                self.stacker_2 = self.stacker_2[0]
        elif self.stacker_2:
            self.num_models = 1
        else:
            self.num_models = 0

    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, pred))

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits,
                                     shuffle=True, 
                                     random_state=self.seed).split(X, y))

        oof_columns = []
        s_train = np.zeros((X.shape[0], len(self.base_models)))
        s_test = np.zeros((T.shape[0], len(self.base_models)))
        
        for i, clf in enumerate(self.base_models):
            s_test_i = np.zeros((T.shape[0], self.n_splits))
            start_time = time.time()

            for j, (train_idx, valid_idx) in enumerate(folds):
                f_start_time = time.time()
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_valid = X[valid_idx]
                y_valid = y[valid_idx]

                clf.fit(X_train, y_train)
                valid_preds = clf.predict(X_valid)
                s_train[valid_idx, i] = valid_preds
                s_test_i[:, j] = clf.predict(T)

                valid_score = self.eval(y_valid, valid_preds)
                f_elapsed_time = time.time() - f_start_time
                print(f'{str(clf).split('(')[0]}_{i + 1} fold {j + 1}')
                print(f'  score: {valid_score:.5f}, time: {f_elapsed_time / 60:.1f} min')

            s_test[:, i] = s_test_i.mean(axis=1)

            valid_score = self.eval(y, s_train[:, i])
            elapsed_time = time.time() - start_time
            print(f'Base model_{i + 1}')
            print(f'  score: {valid_score:.5f}, time: {elapsed_time / 60:.1f} min\n')        
            oof_columns.append('Base model_' + str(i + 1))

        oof_s_train = pd.DataFrame(s_train, columns=oof_columns)
        print('\n')
        print('Correlation between out-of-fold predictions from Base models:')
        print('\n')
        print(oof_s_train.corr())
        print('\n')

        if self.mode == 'oof':
            folds_2 = list(StratifiedKFold(n_splits=self.n_splits,
                                           shuffle=True,
                                           random_state=self.seed).split(s_train, y))

            oof_columns = []
            s_train_2 = np.zeros((s_train.shape[0], len(self.stacker_1)))
            s_test_2 = np.zeros((s_test.shape[0], len(self.stacker_1)))
            
            for i, clf in enumerate(self.stacker_1):
                s_test_i_2 = np.zeros((s_test.shape[0], self.n_splits))
                start_time = time.time()

                for j, (train_idx, test_idx) in enumerate(folds_2):
                    f_start_time = time.time()
                    X_train = s_train[train_idx]
                    y_train = y[train_idx]
                    X_valid = s_train[valid_idx]
                    y_valid = y[valid_idx]

                    clf.fit(X_train, y_train)
                    valid_preds = clf.predict(X_valid)
                    s_train_2[test_idx, i] = valid_preds
                    s_test_i_2[:, j] = clf.predict(s_test)

                    valid_score = self.eval(y_valid, valid_preds)
                    f_elapsed_time = time.time() - f_start_time
                    print(f'{str(clf).split('(')[0]}_{i + 1} fold {j + 1}')
                    print(f'  score: {valid_score:.5f}, time: {f_elapsed_time / 60:.1f} min\n')

                s_test_2[:, i] = s_test_i_2.mean(axis=1)

                valid_score = self.eval(y, s_train_2.mean(axis=1))
                elapsed_time = time.time() - start_time
                print(f'1st level model_{i + 1}')
                print(f' score: {vaild_score:.5f}, time: {elapsed_time / 60:.1f} min\n')
                oof_columns.append('1st level model_' + str(i + 1))

            oof_s_train = pd.DataFrame(s_train_2, columns=oof_columns)
            print('\n')
            print('Correlation between out-of-fold predictions from 1st level models:')
            print('\n')
            print(oof_s_train.corr())
            print('\n')

        elif self.mode == 'woc':
            woc_columns = []
            s_train_2 = np.zeros((s_train.shape[0], len(self.stacker_1)))
            s_test_2 = np.zeros((s_test.shape[0], len(self.stacker_1)))

            for i, clf in enumerate(self.stacker_1):
                start_time = time.time()
                s_train_i_2= np.zeros((s_train.shape[0], s_train.shape[1]))
                s_test_i_2 = np.zeros((s_test.shape[0], s_train.shape[1]))

                for j in range(s_train.shape[1]):
                    f_start_time = time.time()
                    s_tr = s_train[:, np.arange(s_train.shape[1]) != j]
                    s_te = s_test[:, np.arange(s_test.shape[1]) != j]

                    clf.fit(s_tr, y)
                    s_train_i_2[:, j] = clf.predict(s_tr)
                    s_test_i_2[:, j] = clf.predict(s_te)

                    f_elapsed_time = time.time() - f_start_time
                    print(f'{str(clf).split('(')[0]}_{i + 1} subset {j + 1}')
                    print(f'  time: {f_elapsed_time / 60:.1f} min')

                s_train_2[:, i] = s_train_i_2.mean(axis=1)
                s_test_2[:, i] = s_test_i_2.mean(axis=1)

                valid_score = self.eval(y, s_train_2.mean(axis=1))
                elapsed_time = time.time() - start_time
                print(f'1st level model_{i + 1}')
                print(f' score: {valid_score:.5f}, time: {elapsed_time / 60:.1f} min\n')
                woc_columns.append('1st level model_' + str(i + 1))

            woc_s_train = pd.DataFrame(s_train_2, columns=woc_columns)
            print('\n')
            print('Correlation between without-one-column predictions from 1st level models:')
            print('\n')
            print(woc_s_train.corr())
            print('\n')
            
        if self.num_models == 0:
            stack_res = s_test_2.mean(axis=1)
            stack_score = s_train_2.mean(axis=1)
            print('2nd level model: average')
            print(f'  final score: {self.eval(y, stack_score)):.5f}')

        elif self.num_models == 1:
            self.stacker_2.fit(s_train_2, y)
            stack_res = self.stacker_2.predict(s_test_2)
            stack_score = self.stacker_2.predict(s_train_2)
            print(f'2nd level model: {str(self.stacker_2).split('(')[0])}')
            print(f'  final score: {self.eval(y, stack_score)):.5f}')

        else:
            f_columns = []
            stack_score = np.zeros((s_train_2.shape[0], len(self.stacker_2)))
            res = np.zeros((s_test_2.shape[0], len(self.stacker_2)))
            
            for i, clf in enumerate(self.stacker_2):
                clf.fit(s_train_2, y)
                res[:, i] = clf.predict(s_test_2)
                stack_score[:, i] = clf.predict(s_train_2)

                print(f'2nd level model_{i + 1}: {str(clf).split('(')[0]}_{i + 1}')
                print(f'  final score: {self.eval(y, stack_score[:, i])):.5f}\n')
                f_columns.append('2nd level model_' + str(i + 1))

            f_s_train = pd.DataFrame(stack_score, columns=f_columns)
            print('\n')
            print('Correlation between final predictions from 2nd level models:')
            print('\n')
            print(f_s_train.corr())
            print('\n')

            stack_res = res.mean(axis=1)
            print(f'2nd level models final score: {self.eval(y, stack_score.mean(axis=1)):.5f}')

        return stack_score.mean(axis=1), stack_res
