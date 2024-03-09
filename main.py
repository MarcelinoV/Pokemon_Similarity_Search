import streamlit as st
import pandas as pd
import numpy as np
import os
from azure.storage.blob import BlobServiceClient
from utils.functions import rec_similar_pokemon, read_blob_content
from PIL import Image

account_name = st.secrets["account_name"]
account_key = st.secrets["account_key"]
container_name = st.secrets["container_name"]

custom_blob_name = st.secrets["custom_blob_name"]
native_blob_name = st.secrets["native_blob_name"]
updated_blob_name = st.secrets["updated_blob_name"]

# """
# FOR WHEN IN PRODUCTION

# account_name = os.environ['account_name']
# account_key = os.environ['account_key']
# container_name = os.environ['container_name']

# custom_blob_name = os.environ['custom_blob_name']
# native_blob_name = os.environ['native_blob_name']
# """

# print(read_blob_content(account_name, 
#                               account_key,
#                               container_name,
#                               custom_blob_name))

v3_embedding = np.load(read_blob_content(account_name, 
                              account_key,
                              container_name,
                              native_blob_name))

v4_embedding = np.load(read_blob_content(account_name, 
                              account_key,
                              container_name,
                              updated_blob_name))

other = np.load(read_blob_content(account_name, 
                              account_key,
                              container_name,
                              'cosine_sim_v3_pokemon_custom.npy'))

embedding_list = [v3_embedding, v4_embedding, other]

embedding_label = ["Type-Focused Embedding", "Physical-Focused Embedding"]

blob_list = [native_blob_name, updated_blob_name]

embedding_map = {label:[name, matrix] for label,name,matrix in zip(embedding_label, blob_list, embedding_list)}

#print(v3_embedding)

#print(v4_embedding)
# embedding = np.load(custom_blob_name)

pokemon_names = pd.read_csv("app/pokemon_names.csv", index_col="id")

poke_dict = {"index":pokemon_names.index, "url_name":pokemon_names['url_name'].values}

indices = pd.DataFrame(data=poke_dict, index=pokemon_names['name'])

# streamlit app

icon = Image.open("app\img\pokemon-3418266-640.png")

st.set_page_config(page_title="Pokemon Similarity Search",
                    page_icon=icon,
                    layout="centered",
                    initial_sidebar_state="auto",)

st.title("Pokemon Similarity Search")

st.image(Image.open("app\img\pokemon-1624022_640.jpg"), caption="Image by PIRO4D on Pixabay",)

st.subheader("")

embedding_input = st.selectbox("Select Embedding", embedding_map.keys())

if embedding_input == list(embedding_map.keys())[0]:
    st.text('''This cosine similarity matrix uses the following features (6):
    pokedex description, 
    species, 
    ability, 
    type(s), 
    egg group(s), 
    if dual type, 
    if dual egg group.''')
    
else:
    st.text('''This cosine similarity matrix uses the following features (10):
    height, 
    weight, 
    gender distribution, 
    legendary flag, 
    pokedex description, 
    species, 
    ability, 
    type(s), 
    egg group(s), 
    if dual type, 
    if dual egg group.''')

pokemon_input = st.selectbox("Enter Pokemon here", pokemon_names["name"], placeholder="Pokemon")

limit_input = int(st.text_input("Top X Similar Pokemon", value=10, max_chars=2))

#pokemon_input = st.text_input(label="Enter Pokemon here", placeholder="Pokemon", value="Pikachu")

st.dataframe(
    rec_similar_pokemon(pokemon_input, 
    embedding_map[embedding_input][1], 
    indices, 
    pokemon_names,
    limit_input).style.format(subset=['similarity_score'], precision=3),
    column_config={"link": st.column_config.LinkColumn(
        "Pokedex Link",
        display_text=indices.url_name.iloc[0]
    )},
    height=limit_input*38 if limit_input < 25 else limit_input*25 #- round((limit_input*.075*100))
    )