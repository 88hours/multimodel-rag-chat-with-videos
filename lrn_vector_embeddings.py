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
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning, BridgeTowerForImageAndTextRetrieval, BridgeTowerForMaskedLM
import requests
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

def bt_embeddings_from_local(text, image):

    model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
    processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

    processed_inputs  = processor(image, text, padding=True, return_tensors="pt")

    #inputs  = processor(prompt, base64_image, padding=True, return_tensors="pt")
    outputs = model(**processed_inputs)

    cross_modal_embeddings = outputs.cross_embeds
    text_embeddings = outputs.text_embeds
    image_embeddings = outputs.image_embeds
    return {
        'cross_modal_embeddings': cross_modal_embeddings,
        'text_embeddings': text_embeddings,
        'image_embeddings': image_embeddings
    }
    

def bt_scores_with_image_and_text_retrieval():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

    processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-gaudi")
    model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-gaudi")

# forward pass
    scores = dict()
    for text in texts:
        # prepare inputs
        encoding = processor(image, text, return_tensors="pt") 
        outputs = model(**encoding)
        scores[text] = outputs.logits[0,1].item()
    return scores


def bt_with_masked_input():
    url = "http://images.cocodataset.org/val2017/000000360943.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    text = "a <mask> looking out of the window"


    processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-gaudi")
    model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-gaudi")

    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)

    token_ids = outputs.logits.argmax(dim=-1).squeeze(0).tolist()
    if isinstance(token_ids, list):
        results = processor.tokenizer.decode(token_ids)
    else:
        results = processor.tokenizer.decode([token_ids])

    print(results)
    return results

if __name__ == "__main__":
    #res = bt_embeddingsl()
    #print((res['text_embeddings']))
    for img in [img1, img2, img3]:
        embeddings = bt_embeddings_from_local(img['caption'], Image.open(img['image_path']))
        print(embeddings['cross_modal_embeddings'][0].shape)
