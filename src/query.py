from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Direct the embedder to use the exact same Nomic settings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    query_instruction="search_query: "
)

# 2. LOAD the existing, pre-computed database from disk
vectordb_filepath = "/home/crypotik/learning_rag/pokemon_chatbot/chroma"
vector_store = Chroma(
    persist_directory=vectordb_filepath, 
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 3. Set up your local Llama 3.1 1B generator
llm = Ollama(model="llama3.1:8b", temperature=0.2)

system_prompt = "Answer the user question using only this context:\n\n{context}"
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# 4. Interactive user loop
while True:
    user_input = input("\nAsk your data a question (or type 'exit'): ")
    if user_input.lower() == 'exit':
        break
        
    response = rag_chain.invoke({"input": user_input})
    print(f"\nAI: {response['answer']}")
