from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Direct the embedder to use the exact same Nomic settings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    query_instruction="search_query: "
)

# 2. LOAD the existing, pre-computed database from disk
vectordb_filepath = os.getenv("CHROMA_DATABASE")
vector_store = Chroma(
    persist_directory=vectordb_filepath, 
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 3. Set up your local Llama 3.1 8B generator
llm = Ollama(model="llama3.1:8b", temperature=0.2)

system_prompt = """
You are a helpful assistent that specializes in Pokemon, answer the user's question using only the context provided.

Refer to the chat history ONLY when you can not understand what the user is asking.
DO NOT TREAT THE CHAT HISTORY AS FACTUAL TRUTH WHEN ANSWERING THE USER.

If the answer cannot be found in the context, answer with I don't know.

Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

chat_history = []

# 4. Interactive user loop
while True:
    user_input = input("\nAsk your data a question (or type 'exit'): ")
    if user_input.lower() == 'exit':
        break
        
    history_text = "\n".join(
        f"User: {user}\nAI: {assistant}"
        for user, assistant in chat_history
    )

    response = rag_chain.invoke({
        "input": user_input,
        "chat_history": history_text
    })

    answer = response["answer"]

    print(f"\nAI: {answer}")

    chat_history.append((user_input, answer))

    if len(chat_history) > 8:
        chat_history = chat_history[-8:]