# Pokemon_Similarity_Search

![alt text](https://github.com/MarcelinoV/Pokemon_Similarity_Search/blob/main/readme_images/app_flowchart.jpg "Flowchart of Application Components")

Pokemon similarity search engine powered by cosine similarity embeddings stored on Azure Blob, serving as a vector database, with the index on the client side. Site link here:

https://pokemonsimilaritysearch-mv.streamlit.app/

## Code and Resources Used

**Python Version**: 3.12.2

**Packages**: pandas, numpy, nltk, string, re, scipy, sklearn, requests, BeautifulSoup, streamlit, azure-storage-blob

**Data Sources**: [Pokemon Database](https://pokemondb.net/), [Serebii.net](https://www.serebii.net/pokemon/legendary.shtml)

## Data Collection

I gathered the data for this project from the Pokemon Database website, listed as the data source in the previous section. This website is a dedicated resource for Pokemon facts, from the games in the series to individual Pokemon stats, types, traits, and more. For this project, I was interested in attributes pertaining to each Pokemon in the database, such as type, species, and ability. Here is an image of what the part of the page for each pokemon looks like:

![alt text](https://github.com/MarcelinoV/Pokemon_Similarity_Search/blob/main/readme_images/sample_pokemon_db_img.jpg "Example view of webpage with features of interest highlighted")

I also collected the pokedex entries from the games each pokemon appeared in. This array of descriptions would be used to construct each pokemon's TF-IDF feature matrix.

![alt text](https://github.com/MarcelinoV/Pokemon_Similarity_Search/blob/main/readme_images/sample_pokedex_desc_img.jpg "Sample of pokedex entries for Bulbasaur")

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

In cleaning the data, I mostly used string operations since most features collected were text. This included removing punctuation, converting letters to lowercase, removing suffixes or prefixes, and separating columns with 2 values into 2 separate columns. For example, the type column as scraped included 2 types for some pokemon, so I separated that into *type_1* and *type_2* columns for easier interpretation (both for me and the algorithms). 

The most involved data cleaning was with the pokedex descriptions, as I planned to use this column to create TF-IDF vectors for each pokemon. Given that these were represented as a list of pokedex descriptions from games where a given pokemon was present, more processing had to be done such as removing stop words, stemming and lemmatization, and type conversion.

The last feature I added was an *is_legendary* flag to label if a pokemon was a legendary pokemon or not. I created this list with data from serebii.net, a website dedicated to pokemon facts, including a list of legendary pokemon. I then used this list to label each pokemon *is_legendary* = True/False. My final set of features looked like the following:

![alt text](https://github.com/MarcelinoV/Pokemon_Similarity_Search/blob/main/readme_images/sample_view_of_final_dataset.jpg "final features for creating embeddings")

## Feature Extraction & Cosine Similarity Scores

For this project, I decided to make two types of cosine similarity matrices: one that focuses more on similarity based on pokemon types and abilities, and another that incorporates physical traits like height, weight, and gender distribution into its similarity scores.

To construct the matrices, I used scikit learn's TfidfVectorizer() and OnehotEncoder() functions to convert the features into numeric representations. From there, the cosine similarity would be applied. I first converted the pokedex descriptions into a matrix of TF-IDF features, and then categorically encoded the rest of the non-numeric features such as species, ability, and if_dual_type. Numeric features like height and weight were left as is.

![alt text](https://github.com/MarcelinoV/Pokemon_Similarity_Search/blob/main/readme_images/tfidf-formula.jpg "TF-IDF formula")

Each encoded feature was then reshaped and concatenated along the rows using numpy. This resulted in feature matrices that were numeric representations of each pokemon in the database. With these, I could apply the consine similarity formula, which would measure the similarity between each vector in the embeddings by taking the normalized dot product of X and Y. In this case, X and Y are both the same embedding, so we're finding the dot product of itself.

This results in a 1024 x 1024 matrix of cosine similarity scores, where each row represents the similarity scores of that row's pokemon to all the other pokemon in the database. For example, the first row in this matrrix would represent Bulbasaur's similarity scores to all other pokemon.

## Vector Database: Azure Blob Storage & Client-side Python Function

These cosine similarity matrices would be used as the data behind the similarity search engine I would build in Python. To build on cloud skills learned from my Microsoft Azure Data Fundamentals Certification, I used Azure Blob Storage as a service to store my created matrices. I then coded a function that could serve as an index for this makeshift vector database, where the function takes a pokemon name as an input and returns the top similar pokemon. This is done by creating a reverse mapping of pokemon names and dataframe indices, and using that mapping to query the similarity matrices based on highest scores.

## Streamlit Deployment

This application was deployed on Streamlit Community Cloud, which is a free hosting service for streamlit apps where developers can share their creations. 

## Conclusion and Recommendations

New matrices can be continuously added to this application's storage, which leaves room for building matrices that use different features and/or algorithms to generate similarity scores between pokemon. I plan on implementing the following updatesin the future:

- Add Euclidean Distance matrix to Azure Blob Storage
- Add Hamming Distance matrix to Azure Blob Storage
- Use a word embedding created from GloVe or FastText for feature extraction of Pokedex descriptions instead of TF-IDF
- Evaluate alignment rate of outputs between different similarity search algorithms


