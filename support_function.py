
from sentence_transformers import SentenceTransformer
import numpy as np 
import pandas as pd

model = SentenceTransformer('bert-base-nli-mean-tokens')

def encode_query(query):
    return model.encode(query)

def get_df():
    df = pd.read_csv("https://github.com/ZacharySoo01/I320D_TextMining-NLP_FinalProject/blob/main/arxiv_results.csv", names = ["id", "title", "summary"])
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(0, axis=0)
    return df

def cosine_distance_based_similarity (vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    return dot_product / (norm_vector1 * norm_vector2)

def return_ranked_text(query, vector_list):

    similarity_scores = {}
    for i, title_vector in enumerate(vector_list):
        sim = cosine_distance_based_similarity(title_vector, query)
        similarity_scores[i] = sim

    # Assuming similarity_scores is a dictionary of {text: similarity_score}
    ranked_texts = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
   # Rank texts based on the similarity score in ascending order. Print the top 10 most similar texts.
    return ranked_texts[:10]

def get_title_from_top10(ranked_text, df):
    new_df = {"Title": [], "Score": []}  # Initialize empty lists
    
    for index, score in ranked_text:
        new_df["Title"].append(df.iloc[index]["title"])  # Use square brackets for indexing
        new_df["Score"].append(score)
    
    return new_df



