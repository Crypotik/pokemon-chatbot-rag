from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# load data/docs
filepath = "/home/crypotik/learning_rag/pokemon_chatbot/data/poke_corpus.csv"
loader = CSVLoader(file_path=filepath)
data = loader.load()

# split data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
chunks = text_splitter.split_documents(data)

# add nomic prefix to chunks
for chunk in chunks:
    chunk.page_content = f"search document: {chunk.page_content}"

# init Nomic
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# create/save to vector db
vectordb_filepath = "/home/crypotik/learning_rag/pokemon_chatbot/chroma"
vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory=vectordb_filepath)

print(f"successfully ingested data to {vectordb_filepath}")