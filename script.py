# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
import support_function as sf
import json
import streamlit as st

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def local_js(file_name):
    with open(file_name) as f:
        st.markdown(f'<script>{f.read()}</script>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)



def main():
    df = sf.get_df()

        # Open the JSON file
    with open("embedding.json", "r") as json_file:
        # Read the contents of the file
        embedding_data = json.load(json_file)
    # App title
    st.title('Better Arxiv: The Quest to Build a Better Search Engine')
    st.caption("Authors: Thang Truong, Zachary Soo, Rory James, Nicola Rowe")

    st.markdown('[Github Link](https://github.com/ZacharySoo01/I320D_TextMining-NLP_FinalProject)')
    local_css("style.css")
    local_js("script.js")

    p = """This project is about using word embedding to make academic paper searches in Computational Linguistics more relevant. 
        By learning the semantic meaning of words in sentences, word embedding allows for results that are semantically matched to the search queries. So, you can expect more accurate and on-point results when searching for papers in this field. 
    """
    st.markdown(p, unsafe_allow_html=False)
        
    st.write('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css"/>', unsafe_allow_html=True)
    search_query = st.text_input("Enter your search query:", "")

    # Check if the search query is not empty
    if search_query:
        query_embeddings = sf.encode_query(search_query)

        ranked_text = sf.return_ranked_text(query_embeddings, embedding_data)

        new_df = sf.get_title_from_top10(ranked_text,df)
        # Display the table
        st.table(new_df)


if __name__ == "__main__":
    main()
