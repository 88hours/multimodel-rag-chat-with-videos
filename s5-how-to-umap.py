from IPython.display import display
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from s2_download_data import load_data_from_huggingface
from utils import prepare_dataset_for_umap_visualization as data_prep
from s3_data_to_vector_embedding import bt_embeddings_from_local
import random

# prompt templates
templates = [
    'a picture of {}',
    'an image of {}',
    'a nice {}',
    'a beautiful {}',
]
# function helps to prepare list image-text pairs from the first [test_size] data
def data_prep(hf_dataset_name, templates=templates, test_size=1000):
    # load Huggingface dataset (download if needed)
    
    #dataset = load_dataset(hf_dataset_name, trust_remote_code=True)
    dataset = load_data_from_huggingface(hf_dataset_name)
    # split dataset with specific test_size
    train_test_dataset = dataset['train'].train_test_split(test_size=test_size)
    # get the test dataset
    test_dataset = train_test_dataset['test']
    img_txt_pairs = []
    for i in range(len(test_dataset)):
        img_txt_pairs.append({
            'caption' : templates[random.randint(0, len(templates)-1)],
            'pil_img' : test_dataset[i]['image']
        })
    return img_txt_pairs
    
# prepare image_text pairs 

# for the first 50 data of Huggingface dataset 
#  "yashikota/cat-image-dataset"
cat_img_txt_pairs = data_prep("yashikota/cat-image-dataset", 
                             "cat", test_size=50)

# for the first 50 data of Huggingface dataset 
#  "tanganke/stanford_cars"
car_img_txt_pairs = data_prep("tanganke/stanford_cars", 
                             "car", test_size=50)

# display an example of a cat image-text pair data
display(cat_img_txt_pairs[0]['caption'])
display(cat_img_txt_pairs[0]['pil_img'])

# display an example of a car image-text pair data
display(car_img_txt_pairs[0]['caption'])
display(car_img_txt_pairs[0]['pil_img'])

# compute BridgeTower embeddings for cat image-text pairs
def load_cat_and_car_embeddings():
    
    def load_embeddings(img_txt_pair):
        pil_img = img_txt_pair['pil_img']
        caption = img_txt_pair['caption']
        return bt_embeddings_from_local(caption, pil_img)
    
    cat_embeddings = []
    for img_txt_pair in tqdm(
                            cat_img_txt_pairs, 
                            total=len(cat_img_txt_pairs)
                        ):
        pil_img = img_txt_pair['pil_img']
        caption = img_txt_pair['caption']
        embedding =load_embeddings(caption, pil_img)
        cat_embeddings.append(embedding)

    # compute BridgeTower embeddings for car image-text pairs
    car_embeddings = []
    for img_txt_pair in tqdm(
                            car_img_txt_pairs, 
                            total=len(car_img_txt_pairs)
                        ):
        pil_img = img_txt_pair['pil_img']
        caption = img_txt_pair['caption']
        embedding = load_embeddings(caption, pil_img)
        car_embeddings.append(embedding)
    return cat_embeddings, car_embeddings
                        

# function transforms high-dimension vectors to 2D vectors using UMAP
def dimensionality_reduction(embed_arr, label):
    X_scaled = MinMaxScaler().fit_transform(embed_arr)
    print(X_scaled)
    mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
    df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
    df_emb["label"] = label
    print(df_emb)
    return df_emb

def show_umap_visualization():
    def reduce_dimensions():
        cat_embeddings, car_embeddings = load_cat_and_car_embeddings()
        # stacking embeddings of cat and car examples into one numpy array
        all_embeddings = np.concatenate([cat_embeddings, car_embeddings])

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