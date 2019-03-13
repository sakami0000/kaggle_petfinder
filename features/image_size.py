import os

import pandas as pd
from PIL import Image


def get_size(filename):
    st = os.stat(filename)
    return st.st_size


def get_dimensions(filename):
    img_size = Image.open(filename).size
    return img_size 


def extract_image_size_features(image_files):
    df_imgs = pd.DataFrame(image_files)
    df_imgs.columns = ['image_filename']
    
    imgs_pets = df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    df_imgs = df_imgs.assign(PetID=imgs_pets)
    
    df_imgs['image_size'] = df_imgs['image_filename'].apply(get_size)
    df_imgs['temp_size'] = df_imgs['image_filename'].apply(get_dimensions)
    df_imgs['width'] = df_imgs['temp_size'].apply(lambda x : x[0])
    df_imgs['height'] = df_imgs['temp_size'].apply(lambda x : x[1])
    df_imgs.drop(['temp_size'], axis=1, inplace=True)
    
    aggs = {
        'image_size': ['sum', 'mean', 'var'],
        'width': ['sum', 'mean', 'var'],
        'height': ['sum', 'mean', 'var']
    }
    
    agg_imgs = df_imgs.groupby('PetID').agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    agg_imgs.columns = new_columns
    agg_imgs = agg_imgs.reset_index()
    
    return agg_imgs
