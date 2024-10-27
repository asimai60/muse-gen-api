from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware
from vae_model_generate import generate_music_api

#global variables

app = FastAPI()

origins = [
    "http://localhost:3000",  # Allow your frontend origin (already present)
    "https://music-generator-website.vercel.app",
    "https://music-generator-website-git-main-asi-mdns-projects.vercel.app",
    "https://music-generator-website-d9tlr2z8y-asi-mdns-projects.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process-midi/", status_code=201, description="Upload a MIDI file to process and generate a new music track.")
async def process_midi_file(file: UploadFile = File(...)):
    """
    Processes an uploaded MIDI file and generates a new music track.

    Args:
        file (UploadFile): The uploaded MIDI file.

    Returns:
        ProcessedFileResponse: A response model containing the URL of the processed file.
    """
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name

    # Process the MIDI file using your AI model
    bucket_name = 'muse-gen-midi-files'
    url = generate_music_api(temp_file_path, bucket_name)
    print("URL: ", url)
    if url is None:
        return {"error": "Failed to process MIDI file"}
    
    # Return the URL of the processed file
    return {"url": url}

@app.get("/download-midi/")
async def download_midi(url: str):
    return FileResponse(url, media_type='audio/midi', filename="generated_track.mid")
