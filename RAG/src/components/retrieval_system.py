import numpy as np
import torch
import pandas as pd
from src.utils.helpers import retrieve_relevant_resources

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_embeddings_from_csv(file_path):
    """
    Load embeddings from a CSV file and prepare them for processing.
    """
    # Read embeddings from CSV
    embeddings_df = pd.read_csv(file_path)

    # Convert embedding column back to np.array
    embeddings_df["embedding"] = embeddings_df["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" "))

    # Convert embeddings to torch tensor and send to device
    embeddings = torch.tensor(np.array(embeddings_df["embedding"].tolist()), dtype=torch.float32).to(
        device)

    return embeddings_df, embeddings


def print_top_results(query: str,
                      scores: torch.tensor,
                      indices: torch.tensor,
                      embeddings_df: pd.DataFrame):
    """
    Prints the top results and scores.
    """

    print(f"User Query: {query}\n")

    # Loop through zipped scores and indices
    for score, index in zip(scores, indices):
        index = index.item()  # Convert tensor index to integer
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk
        print(embeddings_df.loc[index, "sentence_chunk"])
        # Print the page number
        print(f"Page number: {embeddings_df.loc[index, 'page_number']}", "\n")
        # print("\n")


def main():
    # Load embeddings from CSV
    embeddings_df, embeddings = load_embeddings_from_csv("sent_chunks_embeddings.csv")

    # Define the query
    # query ="How to measure soil moisture?"
    # query = "How can we add sidechain to Ethereum blockchain?"
    query = "Explain why a tomato might consider itself the king of the salad."
    
    print(f"Query: {query}")

    # Retrieve relevant resources
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings)

    # Print top results and scores
    print_top_results(query=query,
                      scores=scores,
                      indices=indices,
                      embeddings_df=embeddings_df)


if __name__ == "__main__":
    main()
