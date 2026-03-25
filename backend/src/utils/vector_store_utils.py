import logging
from typing import Optional

logger = logging.getLogger("vectorstore-utils")


def safe_close_azuresearch(vector_store: Optional[object]) -> None:
    """
    Safely close a LangChain AzureSearch vector store.

    Prevents:
    ImportError: sys.meta_path is None (Python shutdown issue)

    Works for both global and local vector stores.
    """
    if vector_store is None:
        return

    try:
        client = getattr(vector_store, "client", None)

        if client:
            logger.info("Closing AzureSearch client...")
            try:
                client.close()
            except Exception as e:
                logger.debug(f"AzureSearch client close error: {e}")

            # IMPORTANT:
            # detach client so __del__ doesn't run cleanup later
            try:
                vector_store.client = None
            except Exception:
                pass

    except Exception as e:
        logger.debug(f"Vector store cleanup skipped: {e}")