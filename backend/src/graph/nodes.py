import os
import logging
import atexit
from typing import Dict, Any, List

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.messages import SystemMessage, HumanMessage

from backend.src.graph.state import (
    VideoAuditState,
    ComplianceIssue,
    AuditResult
)

from backend.src.services.video_indexer import VideoIndexerService

from backend.src.utils.vector_store_utils import safe_close_azuresearch

logger = logging.getLogger('video-audit-ai')

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_CHAT_API_VERSION"),
    temperature=0.0,
    api_key=os.getenv('AZURE_OPENAI_CHAT_API_KEY')
)

structured_llm = llm.with_structured_output(AuditResult, strict=True)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
    api_key=os.getenv('AZURE_OPENAI_EMBEDDING_API_KEY')
)

vector_store = AzureSearch(
    azure_search_endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
    azure_search_key=os.getenv('AZURE_SEARCH_API_KEY'),
    index_name=os.getenv('AZURE_SEARCH_INDEX_NAME'),
    embedding_function=embeddings.embed_query
)

def _cleanup_vector_store():
    global vector_store
    safe_close_azuresearch(vector_store)
    vector_store = None

atexit.register(_cleanup_vector_store)

# NODE 1: Indexer (converting vid to text)
def index_video_node(state: VideoAuditState) -> Dict[str, Any]:
    '''
    Downloads the yt video from the url
    Uploads to the Azure Video Indexer
    extracts the insights
    '''
    video_url = state.get('video_url')
    video_id_input = state.get('video_id', 'vid_demo')

    logger.info(f'----[Node:Indexer] Processing: {video_url}')

    local_filename= 'temp_audit_video.mp4'

    try:
        vi_service = VideoIndexerService()

        if not video_url:
            return {
                "errors": ["video_url missing"],
                "audit_result": AuditResult(
                    compliance_results=[],
                    status="FAIL",
                    final_report="No video URL provided"
                )
            }

        if 'youtube.com' in video_url or 'youtu.be' in video_url:
            local_path= vi_service.download_youtube_video(video_url, output_path=local_filename)
        else:
            raise Exception('Please provide a valid YouTube URL for this test.')
        
        azure_video_id= vi_service.upload_video(local_path, video_name= video_id_input)
        logger.info(f'Upload Success. Azure ID: {azure_video_id}')

        if(os.path.exists(local_path)):
            os.remove(local_path)

        raw_insights = vi_service.wait_for_processing(azure_video_id)

        clean_data = vi_service.extract_data(raw_insights)
        logger.info('---[NODE: Indexer] Extraction Complete -----')

        return {
            "video_metadata": clean_data.get("video_metadata"),
            "transcript": clean_data.get("transcript"),
            "ocr_text": clean_data.get("ocr_text", [])
        }

    except Exception as e:
        logger.error(f'Video Indexer Failed : {e}')
        return {
            "errors": [str(e)],
            "audit_result": AuditResult(
                compliance_results=[],
                status="FAIL",
                final_report="Video indexing failed"
            )
        }

# NODE 2: Compliance Auditor

def audio_content_node(state: VideoAuditState) -> Dict[str, Any]:
    logger.info('-----[Node: Auditor] querying Knowledge base & LLM')

    transcript = state.get('transcript')
    
    if not transcript:
        return {
            "errors": ["Transcript missing"],
            "audit_result": AuditResult(
                compliance_results=[],
                status="FAIL",
                final_report="Audit skipped: no transcript available"
            )
        }

    metadata = state.get("video_metadata") or {}
    ocr_text = state.get('ocr_text', [])

    query_text = f"{transcript} {''.join(ocr_text)}"
    docs = vector_store.similarity_search(query_text, k=3)
    retrieved_rules = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""
        You are a senior brand compliance auditor.

        OFFICIAL REGULATORY RULES:
        {retrieved_rules}

        Analyze transcript + OCR text and produce compliance findings.
        Return structured output matching the required schema.
    """
    
    user_message = f"""
        VIDEO_METADATA: {metadata}
        TRANSCRIPT: {transcript}
        OCR_TEXT: {ocr_text}
    """

    try:
        result: AuditResult = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])

        return {
            "audit_result": result
        }
    
    except Exception as e:
        logger.exception("System Error in Auditor Node")

        return {
            "errors": [str(e)],
            "audit_result": AuditResult(
                compliance_results=[],
                status="FAIL",
                final_report="Audit failed due to system error"
            )
        }