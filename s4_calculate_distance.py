import numpy as np
from numpy.linalg import norm
import torch
from IPython.display import display
import cv2

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

def load_tensor(path):
    return torch.load(path)

def load_embeddings():
    ex1_embed = load_tensor(img1['tensor_path'] + '.pt')
    ex2_embed = load_tensor(img2['tensor_path'] + '.pt')
    ex3_embed = load_tensor(img3['tensor_path'] + '.pt')
    return ex1_embed.data.numpy(), ex2_embed.data.numpy(), ex3_embed.data.numpy()

def cosine_similarity(vec1, vec2):
    similarity = np.dot(vec1,vec2)/(norm(vec1)*norm(vec2))
    return similarity

def calculate_cosine_distance():
    ex1_embed, ex2_embed, ex3_embed = load_embeddings()
    similarity1 = cosine_similarity(ex1_embed, ex2_embed)
    similarity2 = cosine_similarity(ex1_embed, ex3_embed)
    similarity3 = cosine_similarity(ex2_embed, ex3_embed)
    return [similarity1, similarity2, similarity3]

def calcuate_euclidean_distance():
    ex1_embed, ex2_embed, ex3_embed = load_embeddings()
    distance1 = cv2.norm(ex1_embed,ex2_embed, cv2.NORM_L2)
    distance2 = cv2.norm(ex1_embed,ex3_embed, cv2.NORM_L2)
    distance3 = cv2.norm(ex2_embed,ex3_embed, cv2.NORM_L2)
    return [distance1, distance2, distance3]

def show_cosine_distance():
    distances = calculate_cosine_distance()
    print("Cosine similarity between ex1_embeded and ex2_embeded is:")
    display(distances[0])
    print("Cosine similarity between ex1_embeded and ex3_embeded is:")
    display(distances[1])
    print("Cosine similarity between ex2_embeded and ex2_embeded is:")
    display(distances[2])

def show_euclidean_distance():
    distances = calcuate_euclidean_distance()
    print("Euclidean distance between ex1_embeded and ex2_embeded is:")
    display(distances[0])
    print("Euclidean distance between ex1_embeded and ex3_embeded is:")
    display(distances[1])
    print("Euclidean distance between ex2_embeded and ex2_embeded is:")
    display(distances[2])

show_cosine_distance()
show_euclidean_distance()