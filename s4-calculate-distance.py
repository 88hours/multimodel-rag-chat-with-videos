import numpy as np
from numpy.linalg import norm
import torch
from IPython.display import display

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
    img1_tensor = load_tensor(img1['tensor_path'] + '.pt')
    img2_tensor = load_tensor(img2['tensor_path'] + '.pt')
    img3_tensor = load_tensor(img3['tensor_path'] + '.pt')
    return img1_tensor.data.numpy(), img2_tensor.data.numpy(), img3_tensor.data.numpy()

def cosine_similarity(vec1, vec2):
    similarity = np.dot(vec1,vec2)/(norm(vec1)*norm(vec2))
    return similarity

def calculate_distance():
    img1_tensor, img2_tensor, img3_tensor = load_embeddings()
    similarity1 = cosine_similarity(img1_tensor, img2_tensor)
    similarity2 = cosine_similarity(img1_tensor, img3_tensor)
    similarity3 = cosine_similarity(img2_tensor, img3_tensor)
    return [similarity1, similarity2, similarity3]

distances = calculate_distance()
print("Cosine similarity between ex1_embeded and ex2_embeded is:")
display(distances[0])
print("Cosine similarity between ex1_embeded and ex3_embeded is:")
display(distances[1])
print("Cosine similarity between ex2_embeded and ex2_embeded is:")
display(distances[2])
