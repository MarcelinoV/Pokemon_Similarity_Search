# Pokemon_Similarity_Search

![alt text](https://github.com/MarcelinoV/kaggle_commonlit/blob/main/images/commonlit.JPG "CommonLit Readability Competition Banner")

Pokemon similarity search engine powered by cosine similarity embeddings stored on Azure Blob, serving as a vector database, with the index on the client side

## Code and Resources Used

**Python Version**: 3.12.2

**Packages**: pandas, numpy, nltk, string, re, scipy, sklearn, requests, BeautifulSoup, streamlit, azure-storage-blob

**Data Source**: [Pokemon Database](https://pokemondb.net/)

## Data Collection

I gathered the data for this project from the Pokemon Database website, listed as the data source in the previous section. This website is a dedicated resource for Pokemon facts, from the games in the series to individual Pokemon stats, types, traits, and more. For this project, I was interested in attributes pertaining to each Pokemon in the database, such as type, species, and ability. Here is an image of what the part of the page for each pokemon looks like:

![alt text](https://github.com/MarcelinoV/Pokemon_Similarity_Search/blob/main/readme_images/sample_pokemon_db_img.jpg "Example view of webpage with features of interest highlighted")

Using requests and BeautifulSoup, I developed code to loop through the web page of each Pokemon in the database, collect the relevant features, and store them as a pandas dataframe. The resulting dataset had the following schema:

- name STR
- pokedex_desc (pokedex descriptions) STR
- species STR
- ability STR
- type STR
- egg_group STR
- height FLOAT
- weight FLOAT
- male_dist (male distribution) FLOAT
- female_dist (female distribution) FLOAT

From these features, I would create others such as if_dual_type BOOL or if_dual_egg_group BOOL.

To see the code for this, please refer to the Jupyter notebook in the repo.

## Data Processing

In cleaning the data, I mostly used string operations since most features collected were text. This included removing punctuation, converting letters to lowercase, and separating columns with 2 values into 2 separate columns. For example, the type column as scraped included 2 types for some pokemon, so I separated that into *type_1* and *type_2* columns for easier interpretation (both for me and the algorithms). 

The most involved data cleaning was with the pokedex descriptions, as I planned to use this column to create TF-IDF vectors for each pokemon. Given that these were represented as a list of pokedex descriptions from games where a given pokemon was present, more processing had to be done such as removing stop words, stemming and lemmatization, and type conversion.

The last feature I added was a is_legendary flag to label if a pokemon was a legendary pokemon or not. My final set of features looked like the following:

![alt text](https://github.com/MarcelinoV/Pokemon_Similarity_Search/blob/main/readme_images/sample_view_of_final_dataset.jpg "final features for creating embeddings")

## Constructing the Embeddings & Cosine Similarity Scores



## Azure Blob Storage


## Streamlit Deployment


## Conclusion and Recommendations

