"""BIS PDF extraction + Chroma Cloud upsert service.

Matches the n8n workflow "BIS PDF to ChromaDB Ingestion":
  POST /extract-pdf    <- multipart file field "pdf"   -> {"pages": [{"page": 1, "text": "..."}]}
  POST /chroma/upsert  <- JSON {ids, embeddings, documents, metadatas} -> {"upserted": N, "collection": "..."}

Required env vars (Chroma Cloud):
  CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE
Optional:
  CHROMA_COLLECTION (default "bis_standards")
  PORT (default 8000)
"""

import os

import chromadb
import fitz  # PyMuPDF
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

app = FastAPI(title="BIS PDF Service")

COLLECTION = os.getenv("CHROMA_COLLECTION", "bis_standards")


def get_collection():
    missing = [v for v in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE") if not os.getenv(v)]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing env vars: {', '.join(missing)}")
    client = chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )
    return client.get_or_create_collection(COLLECTION)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract-pdf")
async def extract_pdf(pdf: UploadFile = File(...)):
    data = await pdf.read()
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open PDF: {e}")
    pages = [{"page": i + 1, "text": doc[i].get_text("text")} for i in range(doc.page_count)]
    doc.close()
    return {"pages": pages}


class UpsertBody(BaseModel):
    ids: list[str]
    embeddings: list[list[float]]
    documents: list[str]
    metadatas: list[dict]


@app.post("/chroma/upsert")
def chroma_upsert(body: UpsertBody):
    collection = get_collection()
    collection.upsert(
        ids=body.ids,
        embeddings=body.embeddings,
        documents=body.documents,
        metadatas=body.metadatas,
    )
    return {"upserted": len(body.ids), "collection": COLLECTION}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
requirements.txt

fastapi
uvicorn[standard]
pymupdf
chromadb
python-multipart
pydantic
