import numpy as np
from numpy.linalg import norm
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
import torch
from PIL import Image


url1='http://farm3.staticflickr.com/2519/4126738647_cc436c111b_z.jpg'
cap1='A motorcycle sits parked across from a herd of livestock'

url2='http://farm3.staticflickr.com/2046/2003879022_1b4b466d1d_z.jpg'
cap2='Motorcycle on platform to be worked on in garage'

url3='https://i.natgeofe.com/n/548467d8-c5f1-4551-9f58-6817a8d2c45e/NationalGeographic_2572187_3x2.jpg'
cap3='a cat laying down stretched out near a laptop'

img1 = {
  'flickr_url': url1,
  'caption': cap1,
  'image_path' : './shared_data/motorcycle_1.jpg',
  'tensor_path' : './shared_data/motorcycle_1'
}

img2 = {
    'flickr_url': url2,
    'caption': cap2,
    'image_path' : './shared_data/motorcycle_2.jpg',
    'tensor_path' : './shared_data/motorcycle_2'
}

img3 = {
    'flickr_url' : url3,
    'caption': cap3,
    'image_path' : './shared_data/cat_1.jpg',
    'tensor_path' : './shared_data/cat_1'
}

def bt_embeddings_from_local(text, image):

    model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
    processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

    processed_inputs  = processor(image, text, padding=True, return_tensors="pt")

    outputs = model(**processed_inputs)

    cross_modal_embeddings = outputs.cross_embeds
    text_embeddings = outputs.text_embeds
    image_embeddings = outputs.image_embeds
    return {
        'cross_modal_embeddings': cross_modal_embeddings,
        'text_embeddings': text_embeddings,
        'image_embeddings': image_embeddings
    }

def save_embeddings():
    for img in [img1, img2, img3]:
        embedding = bt_embeddings_from_local(img['caption'], Image.open(img['image_path']))
        print(embedding['cross_modal_embeddings'][0].shape) #<class 'torch.Tensor'>
        torch.save(embedding['cross_modal_embeddings'][0], img['tensor_path'] + '.pt')
    
save_embeddings()
