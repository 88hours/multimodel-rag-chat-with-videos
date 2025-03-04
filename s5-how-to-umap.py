from os import path
from IPython.display import display
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from s3_data_to_vector_embedding import bt_embeddings_from_local
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# prompt templates
templates = [
    'a picture of {}',
    'an image of {}',
    'a nice {}',
    'a beautiful {}',
]
# function helps to prepare list image-text pairs from the first [test_size] data
def data_prep(hf_dataset_name, templates=templates, test_size=1000):
    # load Huggingface dataset by streaming the dataset which doesn’t download anything, and lets you use it instantly
    #dataset = load_dataset(hf_dataset_name, trust_remote_code=True, split='train', streaming=True)

    dataset = load_dataset(hf_dataset_name)
    # split dataset with specific test_size
    train_test_dataset = dataset['train'].train_test_split(test_size=test_size)
    test_dataset = train_test_dataset['test']
    print(test_dataset)
    # get the test dataset
    img_txt_pairs = []
    for i in range(len(test_dataset)):
        img_txt_pairs.append({
            'caption' : templates[random.randint(0, len(templates)-1)],
            'pil_img' : test_dataset[i]['image']
        })
    return img_txt_pairs

        

def load_all_dataset():
    
    car_img_txt_pairs = data_prep("tanganke/stanford_cars", test_size=50)
    cat_img_txt_pairs = data_prep("yashikota/cat-image-dataset", test_size=50)
    
    return cat_img_txt_pairs, car_img_txt_pairs
# compute BridgeTower embeddings for cat image-text pairs
def load_cat_and_car_embeddings():
    # prepare image_text pairs 
    cat_img_txt_pairs, car_img_txt_pairs = load_all_dataset()
    def save_embeddings(embedding, path):
        torch.save(embedding, path)

    def load_embeddings(img_txt_pair):
        pil_img = img_txt_pair['pil_img']
        caption = img_txt_pair['caption']
        return bt_embeddings_from_local(caption, pil_img)
    
    def load_all_embeddings_from_image_text_pairs(img_txt_pairs, file_name):
        embeddings = []
        for img_txt_pair in tqdm(
                            img_txt_pairs, 
                            total=len(img_txt_pairs)
                        ):
            
            embedding = load_embeddings(img_txt_pair)
            print(embedding)
            cross_modal_embeddings = embedding['cross_modal_embeddings'][0].detach().numpy() #this is not the right way to convert tensor to numpy
            #print(cross_modal_embeddings.shape) #<class 'torch.Tensor'>
            #save_embeddings(cross_modal_embeddings, file_name)
            embeddings.append(cross_modal_embeddings)
            return cross_modal_embeddings
    

    cat_embeddings = load_all_embeddings_from_image_text_pairs(cat_img_txt_pairs, './shared_data/cat_embeddings.pt')
    car_embeddings = load_all_embeddings_from_image_text_pairs(car_img_txt_pairs, './shared_data/car_embeddings.pt')
    
    return cat_embeddings, car_embeddings
                        

# function transforms high-dimension vectors to 2D vectors using UMAP
def dimensionality_reduction(embeddings, labels):
     
     
    print(embeddings)
    X_scaled = MinMaxScaler().fit_transform(embeddings.reshape(-1, 1)) # This is not the right way to scale the data
    mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
    df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
    df_emb["label"] = labels
    print(df_emb)
    return df_emb

def show_umap_visualization():
    def reduce_dimensions():
        cat_embeddings, car_embeddings = load_cat_and_car_embeddings()
        # stacking embeddings of cat and car examples into one numpy array
        all_embeddings = np.concatenate([cat_embeddings, car_embeddings]) # This is not the right way to scale the data

        # prepare labels for the 3 examples
        labels = ['cat'] * len(cat_embeddings) + ['car'] * len(car_embeddings)

        # compute dimensionality reduction for the 3 examples
        reduced_dim_emb = dimensionality_reduction(all_embeddings, labels)
        return reduced_dim_emb

    reduced_dim_emb = reduce_dimensions()
    # Plot the centroids against the cluster
    fig, ax = plt.subplots(figsize=(8,6)) # Set figsize

    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.scatterplot(data=reduced_dim_emb, 
                    x=reduced_dim_emb['X'], 
                    y=reduced_dim_emb['Y'], 
                    hue='label', 
                    palette='bright')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title('Scatter plot of images of cats and cars using UMAP')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def  an_example_of_cat_and_car_pair_data():
    cat_img_txt_pairs, car_img_txt_pairs = load_all_dataset()
    # display an example of a cat image-text pair data
    display(cat_img_txt_pairs[0]['caption'])
    display(cat_img_txt_pairs[0]['pil_img'])

    # display an example of a car image-text pair data
    display(car_img_txt_pairs[0]['caption'])
    display(car_img_txt_pairs[0]['pil_img'])


if __name__ == '__main__':
    show_umap_visualization()
