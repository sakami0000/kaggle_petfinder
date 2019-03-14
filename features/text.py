import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_textual_features(X_text, n_components=16, seed=1337):
    text_features = []
    
    for i in X_text.columns:
        print(f'generating features from: {i}')
        tfv = TfidfVectorizer(min_df=2, max_features=None,
                              strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                              ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1)
        svd_ = TruncatedSVD(
            n_components=n_components, random_state=seed)
        
        tfidf_col = tfv.fit_transform(X_text.loc[:, i].values)
        
        svd_col = svd_.fit_transform(tfidf_col)
        svd_col = pd.DataFrame(svd_col)
        svd_col = svd_col.add_prefix('tfidf_{}_'.format(i))
        
        text_features.append(svd_col)
        
    text_features = pd.concat(text_features, axis=1)
    return text_features
