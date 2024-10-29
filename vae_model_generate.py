"""
VAE-based Music Generation Module

This module implements a variational autoencoder (VAE) based system for generating multi-instrument
musical sequences. It provides functionality to load trained models, process MIDI files, and generate
new musical compositions based on input sequences.

The module uses PyTorch for neural network implementations and handles 5 instrument tracks:
piano, guitar, bass, strings, and drums.

Key Components:
- VAE models for each instrument
- Neural networks for conditional generation
- MIDI file processing and generation utilities
- Audio synthesis capabilities

Requirements:
    - torch
    - numpy
    - pretty_midi
    - pypianoroll
    - scipy
    - google-cloud-storage
"""

import os
import numpy as np
import pretty_midi
import pypianoroll
import torch
import torch.nn as nn
from torch.nn import functional as F
from vae_helpers import *
from conv_vae import *
import random
import uuid
import io
from datetime import timedelta
from google.cloud import storage
import scipy

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "muse-gen-midi-files-keys.json"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_dir = ''  # 'drive/MyDrive/project'

def load_and_parse_midi(midi_path: str) -> torch.Tensor:
    """
    Load and parse a MIDI file into a multi-track piano roll representation.
    
    Args:
        midi_path (str): Path to the input MIDI file
        
    Returns:
        torch.Tensor: Combined piano roll of shape (5, time_steps, 128) containing
                     piano, guitar, bass, strings and drums tracks.
                     Returns None if parsing fails.
    """
    combined_pianoroll = None
    multitrack = pypianoroll.read(midi_path)
    multitrack.set_resolution(2).pad_to_same()

    # Define the parts we're interested in
    parts = ['Piano', 'Guitar', 'Bass', 'Strings','Ensemble', 'Percussive', 'Drums']

    # Initialize a dictionary to store pianorolls
    pianorolls = {part: None for part in parts}

    # Get the shape of the first non-empty track to use for empty tracks
    empty_array_shape = next((track.pianoroll.shape for track in multitrack.tracks if track.pianoroll.shape[0] > 0), None)

    if empty_array_shape is None:
        print("All tracks are empty")
        return None
    else:
        for track in multitrack.tracks:
            track_name = 'Drums' if track.is_drum else pretty_midi.program_to_instrument_class(track.program)   
            if track_name in parts:
                if track_name == 'Ensemble':
                    track_name = 'Strings'
                pianorolls[track_name if track_name != 'Percussive' else 'Drums'] = track.pianoroll.astype(np.float32) if track.pianoroll.shape[0] > 0 else np.zeros(empty_array_shape, dtype=np.float32)

        # Check if all parts are present and non-empty
        if any(pianoroll is not None and pianoroll.size > 0 for pianoroll in pianorolls.values()):
            # Stack all parts together, using zero arrays for missing parts
            combined_pianoroll = torch.stack([
                torch.from_numpy(pianorolls.get(instrument)) if pianorolls.get(instrument) is not None else torch.zeros(empty_array_shape, dtype=torch.float32)
                for instrument in ['Piano', 'Guitar', 'Bass', 'Strings', 'Drums']
            ])
        else:
            print("No valid parts found in the MIDI file")
            return None

    print("Combined pianoroll shape:", combined_pianoroll.shape)
    return combined_pianoroll

def generate_music_vae(sample: torch.Tensor, vae_models: tuple, nn_models: tuple, 
                      noise_sd: float = 0, threshold: float = 0.3, binarize: bool = True) -> torch.Tensor:
    """
    Generate a new 5-instrument sequence based on the previous sequence using VAE models and neural networks.
    
    Args:
        sample (torch.Tensor): Input sequence of shape (5, 32, 128)
        vae_models (tuple): 5-tuple of trained VAE models for each instrument
        nn_models (tuple): 5-tuple of trained neural networks for each instrument
        noise_sd (float): Standard deviation of noise to add to latent representations
        threshold (float): Threshold for note intensity (0-1)
        binarize (bool): If True, set all non-zero intensities to 0.8
    
    Returns:
        torch.Tensor: Generated sequence of shape (5, 32, 128)
    """
    piano_vae, guitar_vae, bass_vae, strings_vae, drums_vae = vae_models
    melody_nn, guitar_nn, bass_nn, strings_nn, drums_nn = nn_models

    # Split the sample into individual instrument tracks
    instruments = torch.split(sample, 1)

    # Check which instruments are present in the original sample
    instrument_present = [torch.any(instr != 0) for instr in instruments]

    latent_vectors = [
        vae.infer(instr.to(device))[:, :-1] if present
        else torch.zeros(1, vae.fc_mean.out_features).to(device)
        for vae, instr, present in zip(vae_models, instruments, instrument_present)
    ]

    # Generate next latent vector for piano (melody)
    piano_next_latent = melody_nn(latent_vectors[0])
    piano_next_latent += torch.randn_like(piano_next_latent) * noise_sd

    # Generate next latent vectors for other instruments
    next_latent_vectors = [piano_next_latent] + [
        nn(latent, piano_next_latent) + torch.randn_like(piano_next_latent) * noise_sd
        for nn, latent in zip(nn_models[1:], latent_vectors[1:])
    ]

    # Generate new samples from latent vectors only for present instruments
    new_samples = []

    for i, (vae, latent, present) in enumerate(zip(vae_models, next_latent_vectors, instrument_present)):
        if present:
            new_sample = vae.generate(latent.unsqueeze(0)).view(1, 32, 128)
        else:
            if torch.rand(1) < 0.15:  # 15% chance to generate based on piano's latent vector if not present
                new_sample = vae.generate(latent.unsqueeze(0)+ torch.randn_like(piano_next_latent) * noise_sd * 2).view(1, 32, 128)
            else:
                new_sample = torch.zeros(1, 32, 128)
        new_samples.append(new_sample)

    # Combine all instrument samples
    creation = torch.cat(new_samples, dim=0)

    # Apply threshold
    creation[creation < threshold] = 0

    if binarize:
        creation[creation > 0] = 0.8

        # Quieten the strings if present
        if instrument_present[3]:
            creation[3, :, :] *= 0.5

    return creation

def load_models() -> tuple:
    """
    Load all trained VAE and neural network models.
    
    Returns:
        tuple: (vae_models, nn_models) where each is a 5-tuple of models for each instrument
    """
    def load_model(model_class, instrument: str, K: int, model_type: str = 'VAE'):
        model_name = f'{model_type}_{instrument}_{K}'
        save_path = './'+ os.path.join(root_dir, 'Saved Models', 'VAE', model_name)

        model = model_class(K=K).to(device)
        model.load_state_dict(torch.load(save_path,map_location=torch.device('cpu')))
        model.eval()
        return model

    # Specify dimensionality of VAEs
    K = 32

    # Load VAEs
    instruments = ['piano', 'guitar', 'bass', 'strings', 'drums']
    vae_models = tuple(load_model(ConvVAE, instr, K) for instr in instruments)

    # Load Melody NN
    melody_nn = load_model(MelodyNN, 'piano', K, model_type='VAE_NN')

    # Load Conditional NNs
    nn_models = (melody_nn,) + tuple(load_model(ConditionalNN, instr, K, model_type='VAE_NN') for instr in instruments[1:])

    return vae_models, nn_models

def create_midi_file(generated_track: torch.Tensor) -> bytes:
    """
    Convert generated track to MIDI file and return bytes.
    
    Args:
        generated_track (torch.Tensor): Generated music of shape (5, time_steps, 128)
        
    Returns:
        bytes: MIDI file content as bytes
    """
    # Scale the generated track to MIDI velocity range (0-127)
    generated_track_out = generated_track * 127

    # Define instrument configurations
    instruments = [
        ('Piano', 0, False),
        ('Guitar', 24, False),
        ('Bass', 32, False),
        ('Strings', 48, False),
        ('Drums', None, True)
    ]

    # Convert predictions into the multitrack pianoroll
    tracks = []
    for i, (name, program, is_drum) in enumerate(instruments):
        pianoroll = generated_track_out[i].cpu().detach().numpy()
        track = pypianoroll.StandardTrack(
            name=name,
            program=program,
            is_drum=is_drum,
            pianoroll=pianoroll
        )
        tracks.append(track)

    # Create the multitrack
    generated_multitrack = pypianoroll.Multitrack(
        name='Generated',
        resolution=2,
        tracks=tracks
    )

    # Convert to pretty_midi
    generated_pm = pypianoroll.to_pretty_midi(generated_multitrack)

    # Save the MIDI file to a bytes buffer
    midi_bytes = io.BytesIO()
    generated_pm.write(midi_bytes)
    midi_bytes.seek(0)
    
    return midi_bytes.getvalue()

def create_wav_file(generated_track: torch.Tensor) -> bytes:
    """
    Convert generated track to WAV file and return bytes.
    
    Args:
        generated_track (torch.Tensor): Generated music of shape (5, time_steps, 128)
        
    Returns:
        bytes: WAV file content as bytes
    """
    # First convert to MIDI
    midi_data = create_midi_file(generated_track)
    
    # Load MIDI data with pretty_midi
    midi_data = io.BytesIO(midi_data)
    pm = pretty_midi.PrettyMIDI(midi_data)
    
    # Synthesize audio
    audio_data = pm.synthesize(fs=44100)
    
    # Normalize audio to prevent clipping
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Convert to 16-bit integer values
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Save as WAV file to bytes buffer
    wav_buffer = io.BytesIO()
    scipy.io.wavfile.write(wav_buffer, 44100, audio_data)
    wav_buffer.seek(0)
    
    return wav_buffer.getvalue()

def generate_music(sample: torch.Tensor, prediction_steps: int = 3) -> torch.Tensor:
    """
    Generate a musical sequence by extending the input sample.
    
    Args:
        sample (torch.Tensor): Input music of shape (5, time_steps, 128)
        prediction_steps (int): Number of 32-step sequences to generate
        
    Returns:
        torch.Tensor: Generated music sequence
    """
    vae_models, nn_models = load_models()

    generated_track = torch.zeros((5, sample.shape[1] + 32 * (prediction_steps + 1), 128), device=device)
    generated_track[:, :sample.shape[1], :] = sample

    sample = sample[:, -32:, :]
    noise_sd = random.uniform(0.3, 0.7)

    for i in range(prediction_steps):        
        sample = generate_music_vae(
            sample,
            vae_models,
            nn_models,
            noise_sd=noise_sd,
            threshold=0.3,
            binarize=True
        )
        generated_track[:, sample.shape[1] + 32*i:sample.shape[1] + 32*(i+1), :] = sample
    return generated_track

def generate_music_api(input_midi_path: str) -> tuple:
    """
    Process MIDI file and return generated MIDI and WAV file content.
    
    Args:
        input_midi_path (str): Path to input MIDI file
        
    Returns:
        tuple: (midi_content, wav_content) as bytes, or (None, None) if generation fails
    """
    sample = load_and_parse_midi(input_midi_path)
    if sample is None:
        return None, None
        
    generated_track = generate_music(sample, prediction_steps=3)
    
    # Convert the generated track to both MIDI and WAV bytes
    midi_content = create_midi_file(generated_track)
    wav_content = create_wav_file(generated_track)
    
    return midi_content, wav_content

if __name__ == "__main__":
    path = "C:/Users/asifm/Desktop/a.mid"
    generate_music_api(path)

