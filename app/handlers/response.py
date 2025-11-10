from pydantic import BaseModel


class UploadChunkResponse(BaseModel):
    filename: str
    status: str
    total_chunks: int
    chunk_size: int
