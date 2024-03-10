import pandas as pd
from azure.storage.blob import BlobServiceClient

def rec_similar_pokemon(pokemon, cosine_sim_, indices, pokemon_file, max=10):

    '''Index for Vector Data on Azure Blob Storage.
       Returns top x similar pokemon based on input, a pokemon name.'''
    
    # get index of pokemon that matches input
    # minus 1 since tabular data is indexed by Pokemon Pokedex ID 
    # since idx will be used to index the embedding, correct idx for Bulbasaur would be 0, not 1
    

    idx = indices.loc[pokemon]['index'] - 1
    
    # get pairwise similarity scores of all pokemon with the given name
    
    sim_scores = list(enumerate(cosine_sim_[idx]))
    
    # sort pokemon based on similarity scores
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # get scores of 10 most similar pokemon

    # if max > 50:
    #     raise("Maximum display is 50")
    
    sim_scores = sim_scores[0:max+1]
    
    # get pokemon indices & scores

    poke_indices = [i[0] for i in sim_scores] 

    poke_scores = [float(i[1]) for i in sim_scores] 
    
    # return top 10 most similar pokemon

    recs = pokemon_file['name'].iloc[poke_indices]

    links = pokemon_file['url_name'].iloc[poke_indices]

    poke_dict = {"name":recs.values, "similarity_score":poke_scores, "link":[f"https://pokemondb.net/pokedex/{link}" for link in links]}

    recs = pd.DataFrame(poke_dict, index=[i+1 for i in poke_indices])

    recs.index.name = 'National Pokedex No.'
    
    # create mask to exclude input pokemon
    
    #mask = recs == pokemon
    
    return recs.iloc[1:, 0:3]

def read_blob_content(storage_account_name, storage_account_key, container_name, blob_name):

    '''Reads files stored on Azure blob storage container'''

   # Create a BlobServiceClient
    blob_service_client = BlobServiceClient(account_url=f"https://{storage_account_name}.blob.core.windows.net", credential=storage_account_key)
  
    # Get the blob content as a stream
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
 
    # Get the blob content as a stream
    blob_stream = blob_client.download_blob()
   
    # Read the content from the stream
    blob_content = blob_stream.readall()
    
    return blob_content

def upload_to_azure_blob(blob_name, account_name, account_key, container_name):

    '''Uploads local files to azure blob storage container'''

    # Create a BlobServiceClient
    blob_service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)

    # Create a ContainerClient
    container_client = blob_service_client.get_container_client(container_name)

    # Upload the text embedding data to a blob
    custom_blob_client = container_client.get_blob_client(blob_name)

    custom_blob_client.upload_blob(blob_name, overwrite=True)

    print("Upload Complete")
