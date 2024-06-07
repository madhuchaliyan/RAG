import torch
from sentence_transformers import util, SentenceTransformer
import os
import toml
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", 
                                      device=device)

def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=5):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, 
                                   convert_to_tensor=True) 

    # Get dot product scores on embeddings

    dot_scores = util.dot_score(query_embedding, embeddings)[0]

    scores, indices = torch.topk(input=dot_scores, 
                                 k=n_resources_to_return)

    return scores, indices


def split_list(input_list: list, slice_size: int) -> list[list]:
    """
    Splits a list into sublists of a specified size.
    """
    if slice_size <= 0:
        raise ValueError("slice_size must be a positive integer")
    # Split the input list into sublists of given size
    return [input_list[i * slice_size:(i + 1) * slice_size] for i in range(-(-len(input_list) // slice_size))]  # Ceiling division
    
def load_config_values(config_file_path='config/secrets.toml'):
    try:
        with open(config_file_path, 'r') as file:
            data = toml.load(file)
    except FileNotFoundError:
        print(f"File not found: {config_file_path}", file=sys.stderr)
        return
    except toml.TomlDecodeError:
        print(f"Error decoding TOML file: {config_file_path}", file=sys.stderr)
        return
    # Set environment variables
    for key, value in data.items():
        os.environ[key] = str(value)