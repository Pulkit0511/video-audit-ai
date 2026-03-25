import os
import glob
import logging
import uuid
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch

from backend.src.utils.vector_store_utils import safe_close_azuresearch

logging.basicConfig(
    level= logging.INFO,
    format= '%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('indexer')

def index_docs():
    '''
    Reads the PDFs, chunks them, and uploads them to Azure AI Search
    '''

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, '../../backend/data')

    logger.info("="*60)
    logger.info('Environment Configuration Check: ')
    logger.info(f"AZURE_OPENAI_EMBEDDING_ENDPOINT : {os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT')}")
    logger.info(f"AZURE_OPENAI_CHAT_API_VERSION : {os.getenv('AZURE_OPENAI_CHAT_API_VERSION')}")
    logger.info(f"AZURE_OPENAI_EMBEDDING_API_VERSION : {os.getenv('AZURE_OPENAI_EMBEDDING_API_VERSION')}")
    logger.info(f"EMBEDDING_DEPLOYMENT : {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-3-small')}")
    logger.info(f"AZURE_SEARCH_ENDPOINT : {os.getenv('AZURE_SEARCH_ENDPOINT')}")
    logger.info(f"AZURE_SEARCH_INDEX_NAME : {os.getenv('AZURE_SEARCH_INDEX_NAME')}")
    logger.info("="*60)

    required_vars=[
        'AZURE_OPENAI_EMBEDDING_ENDPOINT',
        'AZURE_OPENAI_EMBEDDING_API_KEY',
        'AZURE_SEARCH_ENDPOINT',
        'AZURE_SEARCH_API_KEY',
        'AZURE_SEARCH_INDEX_NAME'
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f'Missing required environment variables : {missing_vars}')
        logger.error(f'Please check your .env file and ensure all the variables are set')
        return
    
    try:
        logger.info('Initializing Azure Open AI Embeddings ......')
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment= os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-3-small'),
            azure_endpoint= os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT'),
            api_key= os.getenv('AZURE_OPENAI_EMBEDDING_API_KEY'),
            openai_api_version = os.getenv('AZURE_OPENAI_EMBEDDING_API_VERSION', '2024-02-01')
        )
        logger.info('Embeddings model initialized successfully')
    except Exception as e:
        logger.error(f'Failed to initialize embeddings : {e}')
        logger.error('Please verify your Azure OpenAI deployment name and endpoint')
        return

    index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')

    try:
        logger.info('Initializing Azure AI Search vector store ......')
        vector_store = AzureSearch(
            azure_search_endpoint= os.getenv('AZURE_SEARCH_ENDPOINT'),
            azure_search_key= os.getenv('AZURE_SEARCH_API_KEY'),
            index_name= index_name,
            embedding_function = embeddings.embed_query
        )
        logger.info('Vector Store initialized for index: %s', index_name)
    except Exception as e:
        logger.error(f'Failed to initialize Azure Search : {e}')
        logger.error('Please verify your Azure Search API key, index name and endpoint')
        return
    
    pdf_files = glob.glob(os.path.join(data_folder, '*.pdf'))
    if not pdf_files:
        logger.warning(f'No PDFs found in {data_folder}. Please add files.')
    logger.info(f'Found {len(pdf_files)} PDFs to process: {[os.path.basename(f) for f in pdf_files]}')

    all_splits = []

    for pdf_path in pdf_files:
        try:
            logger.info(f'Loading:{os.path.basename(pdf_path)}.......')
            loader = PyPDFLoader(pdf_path)
            raw_docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200
            )
            splits = text_splitter.split_documents(raw_docs)
            for split in splits:
                split.metadata['source'] = os.path.basename(pdf_path)
                split.metadata["chunk_id"] = str(uuid.uuid4())

            all_splits.extend(splits)
            logger.info(f'Split into {len(splits)} chunks.')
        except Exception as e:
            logger.error(f'Failed to process {pdf_path} : {e}')
        
    try:
        if all_splits:
            logger.info(
                f'Uploading {len(all_splits)} chunks to Azure AI Search Index "{index_name}"'
            )

            vector_store.add_documents(documents=all_splits)

            logger.info('=' * 60)
            logger.info('Indexing Complete! Knowledge Base is ready....')
            logger.info(f'Total chunks indexed : {len(all_splits)}')
            logger.info('=' * 60)
        else:
            logger.warning('No documents were processed.')

    finally:
        if 'vector_store' in locals():
            safe_close_azuresearch(vector_store)
            del vector_store

if __name__ == '__main__':
    index_docs()