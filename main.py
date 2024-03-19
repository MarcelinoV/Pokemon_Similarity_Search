import streamlit as st
import pandas as pd
import numpy as np
import os
from azure.storage.blob import BlobServiceClient
from utils.functions import rec_similar_pokemon, read_blob_content
from PIL import Image
from io import BytesIO

account_name = os.environ["account_name"]
account_key = os.environ["account_key"]
container_name = os.environ["container_name"]

custom_blob_name = os.environ["custom_blob_name"]
native_blob_name = os.environ["native_blob_name"]
updated_blob_name = os.environ["updated_blob_name"]

icon_path = os.path.abspath(os.path.join("img", "pokemon-3418266-640.png"))

main_img_path = os.path.abspath(os.path.join("img", "pokemon-1624022_640.jpg"))

v3_content = read_blob_content(account_name, 
                              account_key,
                              container_name,
                              custom_blob_name)

v4_content = read_blob_content(account_name, 
                              account_key,
                              container_name,
                              updated_blob_name)

with BytesIO(v3_content) as v3:
    v3_vectors = np.load(v3)

with  BytesIO(v4_content) as v4:
    v4_vectors = np.load(v4)

# other = np.load(BytesIO(read_blob_content(account_name, 
#                               account_key,
#                               container_name,
#                               native_blob_name)), allow_pickle=True)

vector_list = [v3_vectors, v4_vectors]

vector_label = ["Type-Focused Vectors", "Physical-Focused Vectors"]

blob_list = [custom_blob_name, updated_blob_name]

vector_map = {label:[name, matrix] for label,name,matrix in zip(vector_label, blob_list, vector_list)}

pokemon_names = pd.read_csv("pokemon_names.csv", index_col="id")

poke_dict = {"index":pokemon_names.index, "url_name":pokemon_names['url_name'].values}

indices = pd.DataFrame(data=poke_dict, index=pokemon_names['name'])

# streamlit app

icon = Image.open(icon_path)

st.set_page_config(page_title="Pokemon Similarity Search",
                    page_icon=icon,
                    layout="centered",
                    initial_sidebar_state="auto",)

st.title("Pokemon Similarity Search")

st.image(Image.open(main_img_path), caption="Image by PIRO4D on Pixabay")

st.subheader("")

vector_input = st.selectbox("Select Vectors", vector_map.keys())

pokemon_input = st.selectbox("Enter Pokemon here", pokemon_names["name"], placeholder="Pokemon")

if vector_input == list(vector_map.keys())[0]:
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

try:
    limit_input = int(st.text_input("Top X Similar Pokemon", value=10, max_chars=2))
    
    st.dataframe(
    rec_similar_pokemon(pokemon_input, 
    vector_map[vector_input][1], 
    indices, 
    pokemon_names,
    limit_input).style.format(subset=['similarity_score'], precision=3),
    column_config={"link": st.column_config.LinkColumn(
        "Pokedex Link",
        display_text=indices.url_name.iloc[0]
    )},
    height=limit_input*38 if limit_input < 25 else limit_input*25
    )

except ValueError:
    st.write("Please enter a number.")
