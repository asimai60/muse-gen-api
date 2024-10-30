"""
FastAPI application for music generation using VAE and RNN models.

This module provides REST API endpoints for processing MIDI files and generating music using
either a Variational Autoencoder (VAE) or Recurrent Neural Network (RNN) model. It handles
file uploads, model processing, and cloud storage integration.

Endpoints:
    POST /process-midi/{model_type}/: Process an uploaded MIDI file
    GET /download-midi/: Download a generated MIDI file

Dependencies:
    - fastapi: Web framework for building APIs
    - google-cloud-storage: Google Cloud Storage client
    - python-multipart: For handling file uploads
    - tempfile: For temporary file handling
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware
from vae_model_generate import generate_music_api as generate_music_vae
from RNNGeneration import generate_music_api as generate_music_rnn
from google.cloud import storage
from datetime import timedelta
import uuid
import io
from typing import Dict, Optional, Union
from gpt_generation import generate_music_api as generate_music_gpt

# Initialize FastAPI app
app = FastAPI(
    title="Music Generation API",
    description="API for generating music using VAE and RNN models",
    version="1.0.0"
)

# Configure CORS
origins = [
    "http://localhost:3000",
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

@app.post("/process-midi/{model_type}/", status_code=201, 
         response_model=Dict[str, str],
         responses={
             400: {"description": "Invalid model type or processing failed"},
             500: {"description": "Failed to upload processed files"}
         })
async def process_midi_file(model_type: str, file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Process an uploaded MIDI file using either VAE or RNN model.
    
    Args:
        model_type (str): The type of model to use ('vae' or 'rnn')
        file (UploadFile): The uploaded MIDI file to process
    
    Returns:
        Dict[str, str]: Dictionary containing:
            - midi_url: Signed URL for downloading generated MIDI file
            - wav_url: Signed URL for downloading generated WAV file
            - model_type: Type of model used for generation
            
    Raises:
        HTTPException: If model type is invalid, processing fails, or upload fails
    """
    if model_type not in ['vae', 'rnn', 'gpt']:
        raise HTTPException(status_code=400, detail="Model type must be either 'vae', 'rnn', or 'gpt'")

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name

    try:
        # Process the MIDI file using the selected model
        try:
            if model_type == 'vae':
                midi_content, wav_content = generate_music_vae(temp_file_path)
            elif model_type == 'rnn':
                midi_content, wav_content = generate_music_rnn(temp_file_path)
            else:  # gpt
                midi_content, wav_content = generate_music_gpt(temp_file_path)
        except ValueError as e:
            if "No piano part found in the MIDI file" in str(e):
                raise HTTPException(status_code=400, detail="No piano part found in the MIDI file. Please provide a MIDI file containing piano music.")
            raise e
        
        if midi_content is None or wav_content is None:
            raise HTTPException(status_code=400, detail="Failed to process MIDI file")
        
        # Upload both files to cloud storage
        bucket_name = 'muse-gen-midi-files'
        midi_url = upload_to_cloud_storage(midi_content, bucket_name, '.mid')
        wav_url = upload_to_cloud_storage(wav_content, bucket_name, '.wav')
        
        if midi_url is None or wav_url is None:
            raise HTTPException(status_code=500, detail="Failed to upload processed files")
        
        return {
            "midi_url": midi_url,
            "wav_url": wav_url,
            "model_type": model_type
        }
        
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

@app.get("/download-midi/", response_class=FileResponse)
async def download_midi(url: str) -> FileResponse:
    """
    Download a generated MIDI file.
    
    Args:
        url (str): The signed URL of the MIDI file to download
        
    Returns:
        FileResponse: The MIDI file as a downloadable response
    """
    return FileResponse(url, media_type='audio/midi', filename="generated_track.mid")

def upload_to_cloud_storage(file_content: bytes, bucket_name: str, extension: str = '.mid') -> Optional[str]:
    """
    Upload a file to Google Cloud Storage and generate a signed URL.
    
    Args:
        file_content (bytes): The file content to upload
        bucket_name (str): Name of the Google Cloud Storage bucket
        extension (str, optional): File extension ('.mid' or '.wav'). Defaults to '.mid'
        
    Returns:
        Optional[str]: Signed URL for accessing the uploaded file, or None if upload fails
        
    Note:
        The signed URL is valid for 1 hour after generation
    """
    try:
        # Initialize the Google Cloud Storage client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Generate a unique filename
        unique_filename = f"{uuid.uuid4()}{extension}"
        blob = bucket.blob(unique_filename)

        # Set appropriate content type
        content_type = 'audio/midi' if extension == '.mid' else 'audio/wav'
        blob.upload_from_string(file_content, content_type=content_type)

        # Generate a signed URL valid for 1 hour
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=1),
            method="GET"
        )
        return url
    except Exception as e:
        print(f"Failed to upload to cloud storage: {str(e)}")
        return None
