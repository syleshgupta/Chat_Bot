import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from getpass import getpass
import os

#API Initialization
load_dotenv() 
API_KEY = os.getenv("Mistral_Key")

#Header Info

st.header("PDF Reader")

with st.sidebar:
    st.title("Documents")
    file = st.file_uploader("Upload your PDF File",type="pdf")

# Extract the text from PDF
if file is not None:
    pdf_data = PdfReader(file)
    data = ""
    for page in pdf_data.pages:
        data += page.extract_text() #Reading page by page and append the data to a text variable

# Chunk Preparation

    # Recursive Text Splitter ensures that text is splitted based either on paragraph, sentence, word, etc.
    # It recursively tries to split in order to find the one that keeps the chunk under specified size
    # why not just split by fixed characters ? - Blindly cutting the sentences mid way may cause words in half, loosing context, etc.
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n", #To make it recursive breaking - ["\n\n", "\n", ".", " ", ""].
        chunk_size=500, # For every 500 characters, it will be considered as chunk.
        chunk_overlap=150, #Overlap of last 150 characters into the next chunk such that context will stay intact. 
        length_function=len
        )
    chunks = text_splitter.split_text(data)

# Embedding Generation



# Vector Storage     