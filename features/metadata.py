import glob

from joblib import delayed, Parallel
import numpy as np
import pandas as pd

from .utils import open_json_file


def parse_metadata_file(file):
    file_keys = list(file.keys())
    
    if 'labelAnnotations' in file_keys:
        file_annots = file['labelAnnotations']
        file_top_score = np.asarray([x['score'] for x in file_annots]).mean()
        file_top_desc = [x['description'] for x in file_annots]
    else:
        file_top_score = np.nan
        file_top_desc = ['']
        
    file_colors = file['imagePropertiesAnnotation']['dominantColors']['colors']
    file_crops = file['cropHintsAnnotation']['cropHints']

    file_color_score = np.asarray([x['score'] for x in file_colors]).mean()
    file_color_pixelfrac = np.asarray([x['pixelFraction'] for x in file_colors]).mean()

    file_crop_conf = np.asarray([x['confidence'] for x in file_crops]).mean()
    
    if 'importanceFraction' in file_crops[0].keys():
        file_crop_importance = np.asarray([x['importanceFraction'] for x in file_crops]).mean()
    else:
        file_crop_importance = np.nan

    df_metadata = {
        'annots_score': file_top_score,
        'color_score': file_color_score,
        'color_pixelfrac': file_color_pixelfrac,
        'crop_conf': file_crop_conf,
        'crop_importance': file_crop_importance,
        'annots_top_desc': ' '.join(file_top_desc)
    }
    
    df_metadata = pd.DataFrame.from_dict(df_metadata, orient='index').T
    df_metadata = df_metadata.add_prefix('metadata_')
    
    return df_metadata


def extract_metadata_features(pet_id, mode='train'):
    metadata_filenames = sorted(glob.glob(
        f'../input/petfinder-adoption-prediction/{mode}_metadata/{pet_id}*.json'))
    
    if len(metadata_filenames) > 0:
        dfs_metadata = []
        
        for f in metadata_filenames:
            metadata_file = open_json_file(f)
            df_metadata = parse_metadata_file(metadata_file)
            df_metadata['PetID'] = pet_id
            dfs_metadata.append(df_metadata)
        
        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)
    
    else:
        dfs_metadata = pd.DataFrame()
    
    return dfs_metadata
    
    
def aggrigate_metadata_features(pet_ids, mode='train', agg=['sum', 'mean', 'var']):
    dfs_metadata = Parallel(n_jobs=-1, verbose=1)(
        delayed(extract_metadata_features)(i, mode=mode) for i in pet_ids)
    dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)

    metadata_desc = dfs_metadata.groupby(['PetID'])['metadata_annots_top_desc'].unique()
    metadata_desc = metadata_desc.reset_index()
    metadata_desc['metadata_annots_top_desc'] = metadata_desc[
        'metadata_annots_top_desc'].apply(lambda x: ' '.join(x))
    
    metadata_gr = dfs_metadata.drop(['metadata_annots_top_desc'], axis=1)
    for i in metadata_gr.columns:
        if 'PetID' not in i:
            metadata_gr[i] = metadata_gr[i].astype(float)
    metadata_gr = metadata_gr.groupby(['PetID']).agg(agg)
    metadata_gr.columns = pd.Index(
        [f'{c[0]}_{c[1].upper()}' for c in metadata_gr.columns.tolist()])
    metadata_gr = metadata_gr.reset_index()
    
    return metadata_desc, metadata_gr
