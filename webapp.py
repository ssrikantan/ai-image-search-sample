import datetime
import io
import json
import os
import requests
import sys
import gradio as gr

from dotenv import load_dotenv
import PIL.Image

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient  
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery,
    VectorFilterMode,    
)
from azure.search.documents.indexes.models import (  
 
    ExhaustiveKnnParameters,  
    ExhaustiveKnnAlgorithmConfiguration,
    FieldMapping,  
    HnswParameters,  
    HnswAlgorithmConfiguration,  
    InputFieldMappingEntry,  
    OutputFieldMappingEntry,  
    SimpleField,
    SearchField,  
    SearchFieldDataType,  
    SearchIndex,  
    SearchIndexer,  
    SearchIndexerDataContainer,  
    SearchIndexerDataSourceConnection,  
    SearchIndexerSkillset,  
    VectorSearch,  
    VectorSearchAlgorithmKind,  
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,  
    WebApiSkill
)  
from IPython.display import Image, display 
from azure.storage.blob import BlobServiceClient  

topn = 5
imgsize = 360

footnote = "Powered by Azure Computer Vision & Azure Cognitive Search"
header_prompt = "Visual Search with Azure AI using a prompt"
header_images = "Visual Search with Azure AI using a prompt"

logo = "https://github.com/retkowsky/visual-search-azureAI/blob/main/logo.jpg?raw=true"
image = "<center> <img src= {} width=512px></center>".format(logo)

# Themes: https://huggingface.co/spaces/gradio/theme-gallery
theme = "snehilsanyal/scikit-learn"




load_dotenv()

# Azure Computer Vision 4
acv_key = os.getenv("acv_key")
acv_endpoint = os.getenv("acv_endpoint")

print('acv endpoint',acv_endpoint)

# Azure Cognitive Search
acs_endpoint = os.getenv("acs_endpoint")
acs_key = os.getenv("acs_key")

# Azure Cognitive Search index name to create
index_name = os.getenv("index_name")

# Azure Cognitive Search api version
api_version = os.getenv("aisearch_api_version")





def text_embedding(prompt):
    """
    Text embedding using Azure Computer Vision 4.0
    """
    version = "?api-version=" + api_version + "&modelVersion=latest"
    vec_txt_url = f"{acv_endpoint}/computervision/retrieval:vectorizeText{version}"
    headers = {"Content-type": "application/json", "Ocp-Apim-Subscription-Key": acv_key}

    payload = {"text": prompt}
    response = requests.post(vec_txt_url, json=payload, headers=headers)

    if response.status_code == 200:
        text_emb = response.json().get("vector")
        return text_emb

    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def prompt_search_gradio(prompt, topn=5):
    """
    Prompt search for the gradio webapp
    """
    results_list = []
    images_list = []

    # Initialize the Azure Cognitive Search client
    search_client = SearchClient(acs_endpoint, index_name, AzureKeyCredential(acs_key))
    # Perform vector search
    # vector = Vector(value=text_embedding(prompt), k=topn, fields="imagevector")

    vector = VectorizedQuery(vector=text_embedding(prompt), k_nearest_neighbors=5, fields="imageVector")  

    response = search_client.search(
        search_text=prompt, vector_queries=[vector], select=["title", "imageUrl"]
    )

    for result in response:
        image_file = result["imageUrl"]
        print('image url:',image_file)
        results_list.append(image_file)

    #img = Image.open(io.BytesIO(blob_image)).resize((imgsize, imgsize))
    # img = Image().open(io.BytesIO(requests.get(image_file).content)).resize((imgsize, imgsize))
    for image_file in results_list:
        img = PIL.Image.open(io.BytesIO(requests.get(image_file).content))
        images_list.append(img)

    return images_list

prompt_examples = [
    "a red dress",
    "a red dress with long sleeves",
    "Articles with birds",
    "Hair products",
    "Ray-Ban",
    "NYC cap",
    "Camo products",
    "A blue shirt with stripped lines",
    "Ray ban with leopard frames",
    "Show me some articles with Italian names",
    "Show me some articles with Japanese on it",
    "Show me some tie-dye articles",
]

prompt = gr.components.Textbox(
    lines=2,
    label="What do you want to search?",
    placeholder="Enter your prompt for the visual search...",
)

topn_list_prompt = [""] * topn

list_img_results_image = [
    gr.components.Image(label=f"Top {i+1}: {topn_list_prompt[i]}", type="filepath")
    for i in range(topn)
]

webapp_prompt = gr.Interface(
    prompt_search_gradio,
    prompt,
    list_img_results_image,
    title=header_prompt,
    examples=prompt_examples,
    theme=theme,
    description=image,
    article=footnote,
)


def image_embedding(imagefile):
    """
    Image embedding using Azure Computer Vision 4.0
    """
    session = requests.Session()

    version = "?api-version=" + api_version + "&modelVersion=latest"
    vec_img_url = acv_endpoint + "/computervision/retrieval:vectorizeImage" + version
    headers = {
        "Content-type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": acv_key,
    }

    try:
        with open(imagefile, "rb") as f:
            data = f.read()
        response = session.post(vec_img_url, data=data, headers=headers)
        response.raise_for_status()

        image_emb = response.json()["vector"]
        return image_emb

    except requests.exceptions.RequestException as e:
        print(f"Request Exception: {e}")
    except Exception as ex:
        print(f"Error: {ex}")

    return None

def image_search_gradio(imagefile, topn=5):
    """
    Image search for the gradio webapp
    """
    results_list = []
    images_list = []

    # Initialize the Azure Cognitive Search client
    search_client = SearchClient(acs_endpoint, index_name, AzureKeyCredential(acs_key))

    # Perform vector search
    #vector = VectorizedQuery(vector=text_embedding(prompt), k_nearest_neighbors=3, fields="imageVector")  
    response = search_client.search(
        search_text="",
        vector_queries=[VectorizedQuery(vector=image_embedding(imagefile), k_nearest_neighbors=5, fields="imageVector")],
        select=["title", "imageUrl"],
    )

    for result in response:
        image_file = result["imageUrl"]
        print('image url:',image_file)
        results_list.append(image_file)

    for image_file in results_list:
        img = PIL.Image.open(io.BytesIO(requests.get(image_file).content))
        images_list.append(img)

    return images_list

images_examples = [
    "images/image1.jpg",
    "images/image2.jpg",
    "images/image3.jpg",
    "images/image4.jpg",
    "images/image5.jpg",
    "images/image6.jpg",
    "images/image7.jpg",
    "images/image8.jpg",
]

refimage = gr.components.Image(label="Your image:", type="filepath", height=imgsize, width=imgsize)
topn_list_prompt = [""] * topn

list_img_results_image = [
    gr.components.Image(label=f"Top {i+1} : {topn_list_prompt[i]}", type="filepath")
    for i in range(topn)
]

webapp_image = gr.Interface(
    image_search_gradio,
    refimage,
    list_img_results_image,
    title=header_images,
    examples=images_examples,
    theme=theme,
    description=image,
    article=footnote,
)

visual_search_webapp = gr.TabbedInterface(
    [webapp_prompt, webapp_image],
    [
        "1) Visual search from a prompt",
        "2) Visual search from an image"
    ],
    css="body {background-color: black}",
    theme=theme,
)

visual_search_webapp.launch(share=True)