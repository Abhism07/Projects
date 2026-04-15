from fastapi import FastAPI, UploadFile, File
import shutil, os

from core.file_loader import load_file
from core.cleaner import clean_data
from core.schema_profiler import generate_reports
from core.intelligence import analyze_columns
import uvicorn


app = FastAPI(title="AI Data Intelligence Engine")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    df = load_file(file_path)
    df = clean_data(df)

    reports = generate_reports(df)
    column_info = analyze_columns(df)

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_intelligence": column_info,
        "reports": reports
    }
if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)
