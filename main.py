import streamlit as st
from pyPDF2 import PDFReader

#Header Info

st.header("PDF Reader")

with st.sidebar:
    st.title("Documents")
    file = st.file_uploader("Upload your PDF File",type="pdf")

# Extract the text from PDF
if file is not None:
    pdf_data = PDFReader(file)
    data = ""
    for page in pdf_data.pages:
        text += page.extract_text() #Reading page by page and append the data to a text variable
        st.write(text)
