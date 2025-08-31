from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
import os

from dotenv import load_dotenv
load_dotenv()

##### WHAT THIS DOES #####
# 1. Load pdf's into document
# 2. Split document into chunks
# 3. Create vector embeddings using LLM
# 4. Store vector embeddings into Chroma
##########################



# data variables
DATA_PATH = "data"
CHROMA_PATH = "embeddings.db"

# load pdf's into document
def load_documents():
    loader = DirectoryLoader(path = DATA_PATH, glob = "*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()
    return documents
docs = load_documents()

# split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 500,
    length_function = len,
    add_start_index = True,
)
chunks = text_splitter.split_documents(docs)

print(f"Number of chunks: ", len(chunks))

# create unique ID's
uuids = [str(uuid4()) for _ in range(len(chunks))]


# initiate embeddings model
embeddings_model = OpenAIEmbeddings(model = "text-embedding-3-small", api_key = os.getenv("OPENAI_API_KEY"))

# add chunks into vector store
vector_store = Chroma(
    collection_name = "my_chroma",
    embedding_function = embeddings_model,
    persist_directory = CHROMA_PATH,
)
vector_store.add_documents(documents = chunks, ids = uuids)