import json
import os
import numpy as np
from numpy.linalg import norm
import cv2
from io import StringIO, BytesIO
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tqdm import tqdm
import base64
from transformers import BridgeTowerProcessor, BridgeTowerModel, BridgeTowerForContrastiveLearning
from PIL import Image
import torch

url1='http://farm3.staticflickr.com/2519/4126738647_cc436c111b_z.jpg'
cap1='A motorcycle sits parked across from a herd of livestock'

url2='http://farm3.staticflickr.com/2046/2003879022_1b4b466d1d_z.jpg'
cap2='Motorcycle on platform to be worked on in garage'

url3='https://i.natgeofe.com/n/548467d8-c5f1-4551-9f58-6817a8d2c45e/NationalGeographic_2572187_3x2.jpg'
cap3='a cat laying down stretched out near a laptop'

img1 = {
  'flickr_url': url1,
  'caption': cap1,
  'image_path' : './shared_data/motorcycle_1.jpg'
}

img2 = {
    'flickr_url': url2,
    'caption': cap2,
    'image_path' : './shared_data/motorcycle_2.jpg'
}

img3 = {
    'flickr_url' : url3,
    'caption': cap3,
    'image_path' : './shared_data/cat_1.jpg'
}


def bt_embeddings_from_local(prompt, base64_image):

    model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

    inputs  = BridgeTowerProcessor(prompt, base64_image, padding=True, return_tensors="pt")
    outputs = model(**inputs)

    cross_modal_embeddings = outputs.cross_embeds
    text_embeddings = outputs.text_embeds
    # image_embeddings = outputs.image_embeds

    return text_embeddings.tolist()  # Return the embeddings as a list for easier use

#  encoding image at given path or PIL Image using base64
def encode_image(image_path_or_PIL_img):
    if isinstance(image_path_or_PIL_img, Image.Image):
        # this is a PIL image
        buffered = BytesIO()
        image_path_or_PIL_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        # this is a image_path
        with open(image_path_or_PIL_img, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

embeddings = []
for img in [img1, img2, img3]:
    img_path = img['image_path']
    caption = img['caption']
    base64_img = encode_image(img_path)
    embedding = bt_embeddings_from_local(caption, base64_img)
    embeddings.append(embedding)
# Each image-text pair is now converted into multimodal 
# embedding vector which has dimensions of 512.

print(len(embeddings[0]))