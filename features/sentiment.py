from joblib import delayed, Parallel
import pandas as pd

from .utils import open_json_file


def parse_sentiment_file(file):
    file_sentiment = file['documentSentiment']
    file_entities = [x['name'] for x in file['entities']]
    file_entities = ' '.join(file_entities)
    
    file_sentences_sentiment = [x['sentiment'] for x in file['sentences']]
    
    file_sentences_sentiment = pd.DataFrame.from_dict(
        file_sentences_sentiment, orient='columns')
    file_sentences_sentiment_df = pd.DataFrame(
        {
            'magnitude_sum': file_sentences_sentiment['magnitude'].sum(axis=0),
            'magnitude_mean': file_sentences_sentiment['magnitude'].mean(axis=0),
            'magnitude_var': file_sentences_sentiment['magnitude'].var(axis=0),
            'score_sum': file_sentences_sentiment['score'].sum(axis=0),
            'score_mean': file_sentences_sentiment['score'].mean(axis=0),
            'score_var': file_sentences_sentiment['score'].var(axis=0),
        }, index=[0]
    )
    
    df_sentiment = pd.DataFrame.from_dict(file_sentiment, orient='index').T
    df_sentiment = pd.concat([df_sentiment, file_sentences_sentiment_df], axis=1)

    df_sentiment['entities'] = file_entities
    df_sentiment = df_sentiment.add_prefix('sentiment_')
    
    return df_sentiment


def extract_sentiment_features(pet_id, mode='train'):
    sentiment_filename = f'../input/petfinder-adoption-prediction/{mode}_sentiment/{pet_id}.json'
    try:
        sentiment_file = open_json_file(sentiment_filename)
        df_sentiment = parse_sentiment_file(sentiment_file)
        df_sentiment['PetID'] = pet_id
    except FileNotFoundError:
        df_sentiment = pd.DataFrame()
        
    return df_sentiment
    

def aggrigate_sentiment_features(pet_ids, mode='train', agg=['sum']):
    dfs_sentiment = Parallel(n_jobs=-1, verbose=1)(
        delayed(extract_sentiment_features)(i, mode=mode) for i in pet_ids)
    dfs_sentiment = pd.concat(dfs_sentiment, ignore_index=True, sort=False)
    
    sentiment_desc = dfs_sentiment.groupby(['PetID'])['sentiment_entities'].unique()
    sentiment_desc = sentiment_desc.reset_index()
    sentiment_desc['sentiment_entities'] = sentiment_desc[
        'sentiment_entities'].apply(lambda x: ' '.join(x))
    
    sentiment_gr = dfs_sentiment.drop(['sentiment_entities'], axis=1)
    for i in sentiment_gr.columns:
        if 'PetID' not in i:
            sentiment_gr[i] = sentiment_gr[i].astype(float)
    
    sentiment_gr = sentiment_gr.groupby(['PetID']).agg(agg)
    sentiment_gr.columns = pd.Index([f'{c[0]}' for c in sentiment_gr.columns.tolist()])
    sentiment_gr = sentiment_gr.reset_index()
    
    return sentiment_desc, sentiment_gr
