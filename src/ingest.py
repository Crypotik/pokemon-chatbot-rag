from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
import os

load_dotenv()

# load data/docs
filepath = os.getenv("DATAFILE_PATH")
loader = CSVLoader(
    file_path=filepath,
    metadata_columns=["Total","HP","Attack","Defense","Sp. Atk","Sp. Def","Speed","Generation"],
    content_columns=["Name","Form","Type1","Type2"],
    source_column="ID"
    )
data = loader.load()

# split data (only when CSV rows are above chunk_size)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
# chunks = text_splitter.split_documents(data)

# add nomic prefix to chunks
for chunk in data:
    chunk.page_content = f"search document: {chunk.page_content}"

# init Nomic
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# create/save to vector db
vectordb_filepath = os.getenv("CHROMA_DATABASE")
vector_store = Chroma.from_documents(
    documents=data, 
    embedding=embeddings,
    persist_directory=vectordb_filepath)

print(f"successfully ingested data to {vectordb_filepath}")