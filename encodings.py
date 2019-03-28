import numpy as np
import pandas as pd


def frequency_encoding(train, test, categorical_features):
    x_train = train.copy()
    x_test = test.copy()
    x = pd.concat([x_train, x_test], axis=0, ignore_index=True)
    for cat in categorical_features:
        cat_counts = x[cat].value_counts().to_dict()
        x_train[cat] = train[cat].map(cat_counts)
        x_test[cat] = test[cat].map(cat_counts).fillna(0)

    return x_train, x_test


def one_hot_encoding(df, categorical_features):
    dfs = [df]
    for cat in categorical_features:
        prefix = cat + '_'
        dfs.append(pd.get_dummies(df[cat]).add_prefix(prefix))

    x = pd.concat(dfs, axis=1).drop(categorical_features, axis=1)
    return x


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encoding(train, test, categorical_features, target,
                    min_samples_leaf=1, smoothing=1, noise_level=0):
    if isinstance(target, np.ndarray):
        target = pd.Series(target, name='target')

    x_train = train.copy()
    x_test = test.copy()
    for cat in categorical_features:
        temp = pd.concat([x_train[cat], target], axis=1)
        prior = target.mean()
        averages = temp.groupby(cat)[target.name].agg(['mean', 'count'])
        smoothing = 1 / (1 + np.exp(-(averages['count'] - min_samples_leaf) / smoothing))

        averages[target.name] = prior * (1 - smoothing) + averages['mean'] * smoothing
        averages.drop(['mean', 'count'], axis=1, inplace=True)

        ft_train_series = pd.merge(
            x_train[[cat]],
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=cat, how='left')['average'].fillna(prior).values
        x_train[cat] = add_noise(ft_train_series, noise_level)

        ft_test_series = pd.merge(
            x_test[[cat]],
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=cat, how='left')['average'].fillna(prior).values
        x_test[cat] = add_noise(ft_test_series, noise_level)
    
    return x_train, x_test
