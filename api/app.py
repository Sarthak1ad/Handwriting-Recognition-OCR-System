from fastapi import FastAPI

app = FastAPI(title="Handwritten OCR API")

@app.get("/")
def read_root():
    return {"message": "Handwritten OCR API is running"}
