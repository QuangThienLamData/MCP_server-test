import json
import logging
import os
import tempfile
import threading

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pymupdf
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

load_dotenv()

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_mcp"
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")

mcp = FastMCP(
    name="RAG MCP Server",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

_chroma_client = chromadb.PersistentClient(path="./chroma_db")
_embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def get_drive_service():
    creds_dict = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)


_ingest_status = {"running": False, "total": 0, "done": 0, "current": "", "files_done": [], "error": ""}


def list_drive_files(folder_id: str, service=None, recursive: bool = True) -> list[dict]:
    """List all downloadable files in a Google Drive folder. Returns list of {id, name, mimeType}."""
    if service is None:
        service = get_drive_service()

    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    files = results.get("files", [])

    file_list = []
    for file in files:
        if file["mimeType"] == "application/vnd.google-apps.folder":
            if recursive:
                file_list.extend(list_drive_files(file["id"], service, recursive))
            continue
        if file["mimeType"].startswith("application/vnd.google-apps."):
            continue
        file_list.append(file)

    return file_list


def download_single_file(file_id: str, file_name: str, dest_dir: str, service=None) -> str:
    """Download a single file from Google Drive. Returns the local file path."""
    if service is None:
        service = get_drive_service()

    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join(dest_dir, file_name)
    with open(file_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return file_path


def _ingest_files_background(folder_id: str, recursive: bool):
    """Background worker: download and ingest files one by one."""
    try:
        service = get_drive_service()
        file_list = list_drive_files(folder_id, service, recursive)

        _ingest_status["total"] = len(file_list)
        if not file_list:
            _ingest_status["running"] = False
            return

        collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=_embedding_fn)
        chunk_id = collection.count()

        for i, file_info in enumerate(file_list):
            file_name = file_info["name"]
            _ingest_status["current"] = file_name
            _ingest_status["done"] = i

            logger.info(f"Ingesting ({i + 1}/{len(file_list)}): {file_name}")

            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_path = download_single_file(file_info["id"], file_name, tmp_dir, service)
                    for text, metadata in _extract_text(local_path):
                        collection.add(
                            documents=[text],
                            metadatas=[metadata],
                            ids=[f"{file_name}_{chunk_id}"],
                        )
                        chunk_id += 1
                _ingest_status["files_done"].append(file_name)
            except Exception:
                logger.exception(f"Failed to ingest {file_name}, skipping")

        _ingest_status["done"] = len(file_list)
        _ingest_status["current"] = ""
        logger.info(f"Ingestion complete: {collection.count()} chunks from {len(_ingest_status['files_done'])} files")
    except Exception as e:
        logger.exception("Background ingestion failed")
        _ingest_status["error"] = str(e)
    finally:
        _ingest_status["running"] = False


@mcp.tool()
def list_drive_folders(folder_id: str = "") -> str:
    """
    List subfolders and files in a Google Drive folder.
    Use this to find folder IDs before ingesting. Returns folder names with their IDs.
    Uses the configured GOOGLE_DRIVE_FOLDER_ID if no folder_id is provided.
    """
    target_folder = folder_id or GOOGLE_DRIVE_FOLDER_ID
    if not target_folder:
        return "Error: No folder ID provided."

    service = get_drive_service()
    query = f"'{target_folder}' in parents and trashed = false"
    results = service.files().list(
        q=query, fields="files(id, name, mimeType, size)", pageSize=100
    ).execute()
    files = results.get("files", [])

    if not files:
        return "Folder is empty."

    lines = []
    folders = [f for f in files if f["mimeType"] == "application/vnd.google-apps.folder"]
    docs = [f for f in files if not f["mimeType"].startswith("application/vnd.google-apps.")]

    if folders:
        lines.append("Subfolders:")
        for f in folders:
            lines.append(f"  [folder] {f['name']} (id: {f['id']})")

    if docs:
        lines.append(f"\nFiles ({len(docs)}):")
        for f in docs:
            size_kb = int(f.get("size", 0)) // 1024
            lines.append(f"  [file] {f['name']} ({size_kb}KB)")

    return "\n".join(lines)


def _extract_text(file_path: str) -> list[tuple[str, dict]]:
    """Extract text from a file. Returns list of (text, metadata) tuples per page/chunk."""
    ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    chunks = []

    if ext == ".pdf":
        doc = pymupdf.open(file_path)
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                chunks.append((text, {"file_name": file_name, "page": page_num + 1}))
        doc.close()
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text:
            chunks.append((text, {"file_name": file_name}))

    return chunks


def _start_ingest(folder_id: str, recursive: bool = False) -> str:
    """Start background ingestion. Returns status message."""
    if _ingest_status["running"]:
        return f"Ingestion already in progress: {_ingest_status['done']}/{_ingest_status['total']} files done. Currently processing: {_ingest_status['current']}"

    _ingest_status.update({"running": True, "total": 0, "done": 0, "current": "scanning...", "files_done": [], "error": ""})
    thread = threading.Thread(target=_ingest_files_background, args=(folder_id, recursive), daemon=True)
    thread.start()
    return "Ingestion started in background. Use ingest_status tool to check progress."


def auto_ingest_on_startup():
    """Auto-ingest from Google Drive in background if the knowledge base is empty."""
    if not GOOGLE_DRIVE_FOLDER_ID or not GOOGLE_SERVICE_ACCOUNT_JSON:
        logger.info("Skipping auto-ingest: Google Drive env vars not set.")
        return

    collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=_embedding_fn)
    if collection.count() > 0:
        logger.info(f"Knowledge base already has {collection.count()} chunks, skipping auto-ingest.")
        return

    logger.info(f"Knowledge base is empty. Starting background auto-ingest from: {GOOGLE_DRIVE_FOLDER_ID}")
    _start_ingest(GOOGLE_DRIVE_FOLDER_ID, recursive=True)


@mcp.tool()
def ingest_from_google_drive(folder_id: str = "", recursive: bool = False) -> str:
    """
    Start ingesting documents from a Google Drive folder into the knowledge base.
    Documents are ingested one by one in the background so the server stays responsive.
    Use ingest_status to check progress.

    Args:
        folder_id: Google Drive folder ID. Use list_drive_folders to find subfolder IDs.
        recursive: If True, also download files from subfolders. Default False.
    """
    target_folder = folder_id or GOOGLE_DRIVE_FOLDER_ID
    if not target_folder:
        return "Error: No Google Drive folder ID provided."

    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        return "Error: GOOGLE_SERVICE_ACCOUNT_JSON env var not set."

    return _start_ingest(target_folder, recursive)


@mcp.tool()
def ingest_status() -> str:
    """Check the current status of background document ingestion."""
    if not _ingest_status["running"] and _ingest_status["total"] == 0:
        return "No ingestion has been started."

    collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=_embedding_fn)
    lines = [f"Total chunks in knowledge base: {collection.count()}"]

    if _ingest_status["running"]:
        lines.append(f"Status: IN PROGRESS ({_ingest_status['done']}/{_ingest_status['total']} files)")
        lines.append(f"Currently processing: {_ingest_status['current']}")
    else:
        lines.append(f"Status: COMPLETE ({_ingest_status['done']}/{_ingest_status['total']} files)")

    if _ingest_status["error"]:
        lines.append(f"Error: {_ingest_status['error']}")

    if _ingest_status["files_done"]:
        lines.append(f"Files ingested: {', '.join(_ingest_status['files_done'])}")

    return "\n".join(lines)


@mcp.tool()
def clear_knowledge_base() -> str:
    """Clear all documents from the knowledge base."""
    try:
        _chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    _chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=_embedding_fn)
    return "Knowledge base cleared."


@mcp.tool()
def query_documents(query: str, n_results: int = 5) -> str:
    """
    Query the knowledge base for documents relevant to the provided query.
    """
    try:
        collection = _chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=_embedding_fn)
    except Exception:
        return "No documents ingested yet. Call ingest_from_google_drive first."

    if collection.count() == 0:
        return "Knowledge base is empty. Call ingest_from_google_drive first."

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["metadatas", "documents", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        return "No documents found for the given query."

    formatted_results = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
    distances = results["distances"][0] if results["distances"] else [0] * len(documents)

    for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
        result_text = f"\n--- Result {i + 1} ---\n"
        result_text += f"Content: {doc}\n"
        result_text += f"Source: {metadata.get('file_name', 'Unknown')}\n"
        result_text += f"Similarity Score: {1 - distance:.3f}\n"
        formatted_results.append(result_text)

    response = f"Found {len(documents)} relevant documents for query: '{query}'\n"
    response += "\n".join(formatted_results)
    return response


if __name__ == "__main__":
    mcp.run(transport="stdio")
