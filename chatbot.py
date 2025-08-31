from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
import os

# import .env file
from dotenv import load_dotenv
load_dotenv()

# create data file constants
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

# initiate embeddings model
embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small", api_key = os.getenv("OPENAI_API_KEY"))

# create llm model
llm = ChatOpenAI(temperature = 0.5, model = "gpt-4o-mini") # temp means ramdomness of output

# connect to populated chroma db
vector_store = Chroma(
    collection_name = "my_chroma",
    embedding_function = embedding_model,
    persist_directory = CHROMA_PATH,
)

# number of results taken from chunks
num_results = 5
# retrieves num_results from vector_store
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# call this fuction for every message from client
def stream_response(message, history):
    print(f"Input: {message}. History: {history}\n")

    docs = retriever.invoke(message)
    
    # add all chunks to knowledge
    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content+"\n\n"

    if message is not None:

        partial_message = ""

        rag_prompt = f"""
        You are an assistant which answers questions based on investing 
        knowledge which is provided to you. While answering, you don't use
        your internal knowledge, but solely the information in the "The knowledge"
        section. You don't mention anything to the user about the provided knowledge.

        The question: {message}
        Conversation history: {history}
        The knowledge: {knowledge}
        """

        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
    type = "messages"
)

# launch the Gradio app
chatbot.launch()


