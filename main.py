import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_mistralai import ChatMistralAI
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

## Phase 1 (Refer Architecture Diagram)

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
    embeddings = MistralAIEmbeddings(mistral_api_key= API_KEY)

# Vector Storage (From Meta)     
    # Generates Embeddings with the help of Mistral and then creates a index based vectors to FAISS
    vector_info = FAISS.from_texts(chunks, embeddings) 

## Phase 2 (Refer Architecture Diagram)
    # User Query
    user_query = st.text_input("Type your question...")

    # Similarity Search with Vector Store
    if user_query:
        # similarity_search has embed_vector module to generate vector for the user query such that vector can be compared.
        matched_data = vector_info.similarity_search(user_query)

        llm = ChatMistralAI(model="mistral-small-latest", 
                            mistral_api_key=API_KEY,
                            temperature = 0.3, # 0 - Specific, 5 - Creative (Randomness) 
                            max_tokens = 200) 
    # Final Results
        chain = load_qa_chain(llm,chain_type="stuff") 
        # stuff indicates that it will stuff all documents into a single prompt.
        # we can use chain_type as map_reduce, refine, map_rerank

        llm_response = chain.run(question = user_query, 
                  input_documents = matched_data)

        st.write(llm_response)