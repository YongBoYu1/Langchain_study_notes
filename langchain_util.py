from IPython.display import Image
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def visulize_graph(graph_model):
    """Generate a visual representation of the graph model

    Args:
        graph_model (langgraph): a compiled graph model

    Returns:
        IMG: The visual representation of the graph model
    """
    graph_img = Image(graph_model.get_graph().draw_png())
    return graph_img


def build_db(doc_url: str  = 'https://python.langchain.com/docs/modules/chains'):
    #doc_url = 'https://python.langchain.com/docs/modules/chains'
    #load the documents
    loader = WebBaseLoader(doc_url)
    docs = loader.load()

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Split the documents into chunks then save them in the vector database.
    text_splitter = RecursiveCharacterTextSplitter()
    doc_chunks = text_splitter.split_documents(docs)

    vector_db = Chroma.from_documents(
        documents=doc_chunks,
        embedding=embeddings)

    return vector_db