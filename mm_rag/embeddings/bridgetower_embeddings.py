from typing import List
from langchain_core.embeddings import Embeddings
import torch
from transformers import (
    BridgeTowerProcessor, 
    BridgeTowerForContrastiveLearning
)
from langchain_core.pydantic_v1 import (
    BaseModel,
)
from lrn_vector_embeddings import bt_embeddings_from_local
from utility import encode_image, bt_embedding_from_prediction_guard
from tqdm import tqdm
from PIL import Image
class BridgeTowerEmbeddings(BaseModel, Embeddings):
    """ BridgeTower embedding model """
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using BridgeTower.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        
        embeddings = []
        img = Image.new('RGB', (100, 100))
        for text in texts:
            embedding = bt_embeddings_from_local(text, img)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using BridgeTower.
        
        Args:
            text: The text to embed.
        
        Returns:
            Embeddings for the text as a flat list of floats.
        """
        # Get embeddings
        embeddings = self.embed_documents([text])[0]
        
        # If embeddings is a dict, extract the text embeddings
        if isinstance(embeddings, dict):
            embeddings = embeddings["text_embeddings"]
        
        # If embeddings is a nested list or tensor, flatten it
        if isinstance(embeddings, (list, torch.Tensor)) and len(embeddings) == 1:
            embeddings = embeddings[0]
        
        # Convert tensor to list if needed
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().tolist()
            
        return embeddings

    
    def embed_image_text_pairs(self, texts: List[str], images: List[str], batch_size=2) -> List[List[float]]:
        """Embed a list of image-text pairs using BridgeTower.

        Args:
            texts: The list of texts to embed.
            images: The list of path-to-images to embed
            batch_size: the batch size to process, default to 2
        Returns:
            List of embeddings, one for each image-text pairs.
        """

        # the length of texts must be equal to the length of images
        assert len(texts)==len(images), "the len of captions should be equal to the len of images"
        
        processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
        model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
    
 
                
        embeddings = []
        for path_to_img, text in tqdm(zip(images, texts), total=len(texts)):
            inputs = processor(text=[text], images=[Image.open(path_to_img)], return_tensors="pt")
            outputs = model(**inputs)
            # Get embeddings and convert to list
            embedding = outputs.text_embeds.detach().numpy().tolist()[0]
            embeddings.append(embedding)
        
        return embeddings