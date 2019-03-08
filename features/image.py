import cv2
from keras.applications.densenet import preprocess_input, DenseNet121
import keras.backend as K
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
from keras.models import Model
import numpy as np
import pandas as pd


def resize_to_square(image, img_size=256):
    old_size = image.shape[:2]
    ratio = float(img_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    color = [0, 0, 0]
    img = cv2.resize(image, (new_size[1], new_size[0]))
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
    return new_img


def load_image(path, pet_id, img_size=256):
    image = cv2.imread(f'{path}{pet_id}-1.jpg')
    new_image = resize_to_square(image, img_size)
    new_image = preprocess_input(new_image)
    return new_image
    

def get_model():
    inp = Input((256, 256, 3))
    backbone = DenseNet121(input_tensor=inp,
                           weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
                           include_top=False)
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
    x = AveragePooling1D(4)(x)
    outp = Lambda(lambda x: x[:, :, 0])(x)
    
    model = Model(inp, outp)
    return model


def extract_image_features(pet_ids, model, mode='train', img_size=256, batch_size=32):
    features = {}
    n_batches = len(pet_ids) // batch_size + 1
    
    for b in range(n_batches):
        start = b * batch_size
        end = (b + 1) * batch_size
        batch_pets = pet_ids[start:end]
        batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
        
        for i, pet_id in enumerate(batch_pets):
            try:
                batch_images[i] = load_image(
                    f'../input/petfinder-adoption-prediction/{mode}_images/', pet_id)
            except:
                pass
            
        batch_preds = model.predict(batch_images)
        
        for i, pet_id in enumerate(batch_pets):
            features[pet_id] = batch_preds[i]
    
    img_feats = pd.DataFrame.from_dict(features, orient='index')
    img_feats.columns = [f'pic_{i}' for i in range(img_feats.shape[1])]

    img_feats = img_feats.reset_index()
    img_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)
    
    return img_feats
