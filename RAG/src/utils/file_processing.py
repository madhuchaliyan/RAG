import fitz
import pandas as pd
from spacy.lang.en import English
import re
from sentence_transformers import SentenceTransformer
from src.utils.helpers import split_list

nlp = English()

def load_data(file_path):
    doc = fitz.open(file_path)  
    
    file_content = []
    for page_number, page in enumerate(doc): 
        # Extract text from each page
        text = page.get_text().strip()
        
        # Remove the header from every page
        first_line_end = text.find("\n")
        if first_line_end != -1:
            text = text[first_line_end:].strip()
        
        # Replace "\n" with " " 
        text = text.replace("\n", " ")
        
        # Store page number and text content
        file_content.append({
            "page_number": page_number - 11,  # Adjust page numbers if needed
            "page_word_count": len(text.split(" ")),
            "page_sent_count": len(text.split(". ")),
            "text": text
        })
    return file_content


def make_sent_chunks(file_content, sent_chunk_size=10):
    """
     "all-mpnet-base-v2" model can take input strings can be up to 384 tokens in length 
      (approximately 280 words).The model indeed crashes after 512 tokens length
      Splitting the page content as chunks of 10 sentences(this need to be adjusted
      based on the page content statistics)
    """
    if "sentencizer" not in nlp.pipe_names:
        # Add 'sentencizer' component if not already present
        nlp.add_pipe("sentencizer")
    
    for item in file_content:
        # Tokenize text into sentences
        item["sentences"] = list(nlp(item["text"]).sents)
        # Overwritting the sentence count with more accurate value.
        item['page_sent_count'] = len(item["sentences"])
        # Calculate number of chunks based on sentence chunk size
        item["num_chunks"] = -(-len(item["sentences"]) // sent_chunk_size)  # Ceiling division(e.g. -(-22//10) => 3)
        # Chunk sentences into groups of 'sent_chunk_size'
        item["sentence_chunks"] = [
            "".join(map(str, chunk)).replace("  ", " ").strip()  # Join sentences into chunks
            for chunk in split_list(item["sentences"], sent_chunk_size)
        ]
        # Remove spaCy objects to avoid pickling errors
        del item["sentences"]
    return file_content

def split_page_chunks(dataset):
    pages_and_chunks = []
    for item in dataset:
        # Remove Cover, Title Page, TOC, Preface etc., start from page #1
        if item["page_number"] > 5:
            for i, sentence_chunk in enumerate(item["sentence_chunks"]):
                # Create dictionary with relevant information for each page chunk
                chunk_dict = {
                    "page_number": item["page_number"],
                    "sentence_chunk": re.sub(r'\.([A-Z])', r'. \1', sentence_chunk),  # Fix spacing after periods
                    "chunk_char_count": len(sentence_chunk),
                    "chunk_word_count": len(sentence_chunk.split()),
                    "chunk_token_count": len(sentence_chunk) / 4,  # Approximate token count
                }
                pages_and_chunks.append(chunk_dict)
    return pages_and_chunks

def generate_sent_embeddings(sent_list, save_path="sent_chunks_embeddings.csv"):
    threshold_token_len = 25
    df = pd.DataFrame(sent_list)
    # Filter sentences based on token count threshold
    sent_list = df[df["chunk_token_count"] > threshold_token_len].to_dict(orient="records")

    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cuda")

    for item in sent_list:
        # Generate embeddings for each sentence chunk
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])
    
    sent_embedded_df = pd.DataFrame(sent_list)
    sent_embedded_df.to_csv(save_path, index=False)
    return sent_embedded_df



data = load_data('data/Hands-On IoT Solutions with Blockchain.pdf')

# Check the words per page statistcis
data_df = pd.DataFrame(data)
data_df.describe().round(2)

processed = make_sent_chunks(data)
processed_df = pd.DataFrame(processed)

pages_and_chunks = split_page_chunks(processed)

sentence_chumnks_and_embeddings = generate_sent_embeddings(pages_and_chunks)
 
