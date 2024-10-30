"""
GPT-based Music Generation Module

This module implements an OpenAI GPT-based system for generating musical sequences.
It converts MIDI data into text prompts, sends them to the GPT API, and converts
the responses back into MIDI sequences.

Key Components:
- MIDI to text conversion
- GPT prompt construction 
- Response parsing
- MIDI generation utilities

The module provides a complete pipeline for:
1. Loading and parsing MIDI files
2. Converting musical data to text format
3. Generating continuations using GPT
4. Converting responses back to MIDI/WAV

Requirements:
    - openai>=1.0.0
    - pretty_midi>=0.2.10 
    - numpy>=1.21.0
    - scipy>=1.7.0
    - python-dotenv>=0.19.0
    - pypianoroll>=1.0.0
    - torch>=1.9.0

Example usage:
    >>> from gpt_generation import generate_music_api
    >>> midi_data, wav_data = generate_music_api("input.mid")
    >>> with open("output.mid", "wb") as f:
    >>>     f.write(midi_data)

Notes:
    - Requires OpenAI API key set in environment variables
    - Supports multi-track MIDI files with piano, guitar, bass, strings and drums
    - Generates both MIDI and WAV output formats
    - Uses GPT-4 for high quality music generation
"""

import os
import numpy as np
import pretty_midi
import pypianoroll
import torch
import io
import scipy.io.wavfile
from typing import Tuple, Optional, List, Dict, Union
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
if not os.getenv('OPENAI_API_KEY'):
    print("Warning: OPENAI_API_KEY not found in environment variables")

def notes_to_text(notes: List[Tuple[int, float, float, str]]) -> str:
    """
    Convert note data to a more efficient text representation.

    Args:
        notes (List[Tuple[int, float, float, str]]): List of note tuples containing:
            - pitch (int): MIDI note number (0-127)
            - start_time (float): Note start time in seconds
            - duration (float): Note duration in seconds  
            - instrument (str): Instrument name

    Returns:
        str: Space-separated string of notes in format "I:pitch:start:duration"
            where I is first letter of instrument name

    Example:
        >>> notes = [(60, 0.0, 0.5, "Piano"), (64, 0.5, 0.5, "Guitar")]
        >>> notes_to_text(notes)
        'P:60:0.00:0.50 G:64:0.50:0.50'
    """
    # Map full instrument names to single letters
    instrument_map = {
        'Piano': 'P',
        'Guitar': 'G', 
        'Bass': 'B',
        'Strings': 'S',
        'Drums': 'D'
    }
    
    text_representation = []
    for pitch, start_time, duration, instrument in notes:
        text_representation.append(
            f"{instrument_map[instrument]}:{pitch}:{start_time:.2f}:{duration:.2f}"
        )
    
    return " ".join(text_representation)

def text_to_notes(text: str) -> List[Tuple[int, float, float, str]]:
    """
    Convert text representation back to note data.

    Args:
        text (str): Space-separated string of notes in format "I:pitch:start:duration"

    Returns:
        List[Tuple[int, float, float, str]]: List of note tuples containing:
            - pitch (int): MIDI note number
            - start_time (float): Note start time in seconds
            - duration (float): Note duration in seconds
            - instrument (str): Full instrument name

    Example:
        >>> text = "P:60:0.00:0.50 G:64:0.50:0.50"
        >>> text_to_notes(text)
        [(60, 0.0, 0.5, "Piano"), (64, 0.5, 0.5, "Guitar")]
    """
    # Map single letters back to full instrument names
    instrument_map = {
        'P': 'Piano',
        'G': 'Guitar',
        'B': 'Bass', 
        'S': 'Strings',
        'D': 'Drums'
    }
    
    notes = []
    for note_text in text.split():
        try:
            instrument, pitch, start, duration = note_text.split(":")
            notes.append((
                int(pitch),
                float(start),
                float(duration),
                instrument_map[instrument]
            ))
        except ValueError:
            continue
    return notes

def extract_notes_from_pianoroll(pianoroll: torch.Tensor) -> List[Tuple[int, float, float, str]]:
    """
    Extract note information from a piano roll representation.
    
    Args:
        pianoroll (torch.Tensor): Tensor of shape (5, time_steps, 128) containing:
            - 5 instrument tracks
            - Variable number of time steps
            - 128 MIDI pitches
            
    Returns:
        List[Tuple[int, float, float, str]]: List of note tuples sorted by start time,
            instrument, and pitch. Each tuple contains:
            - pitch (int): MIDI note number
            - start_time (float): Note start time in seconds
            - duration (float): Note duration in seconds
            - instrument (str): Instrument name

    Note:
        Assumes 2 time steps per second for conversion to real time.
        Uses fixed 0.5s duration for all notes for simplicity.
    """
    instruments = ['Piano', 'Guitar', 'Bass', 'Strings', 'Drums']
    notes = []
    
    for i, instrument in enumerate(instruments):
        track = pianoroll[i].numpy()
        active_notes = np.where(track > 0)
        for time, pitch in zip(*active_notes):
            # Convert time steps to seconds (assuming 2 steps per second)
            start_time = time * 0.5
            duration = 0.5  # Fixed duration for simplicity
            notes.append((int(pitch), start_time, duration, instrument))
    
    return sorted(notes, key=lambda x: (x[1], x[3], x[0]))  # Sort by start time, then instrument, then pitch

def create_gpt_prompt(notes: List[Tuple[int, float, float, str]]) -> str:
    """
    Create a musically-aware prompt for GPT to continue the sequence.

    Args:
        notes (List[Tuple[int, float, float, str]]): List of note tuples containing:
            - pitch (int): MIDI note number
            - start_time (float): Note start time in seconds
            - duration (float): Note duration in seconds
            - instrument (str): Instrument name

    Returns:
        str: Formatted prompt containing:
            - Technical format specification
            - Musical requirements and guidelines
            - Current musical sequence
            - Instructions for continuation

    The prompt is designed to:
        - Maintain musical coherence and style
        - Preserve instrument relationships
        - Allow natural timing variations
        - Generate musically meaningful continuations
    """
    # Analyze musical patterns
    instruments_present = set()
    last_times = {}
    chord_patterns = defaultdict(list)
    
    for pitch, start_time, duration, instrument in notes:
        instruments_present.add(instrument)
        last_times[instrument] = max(last_times.get(instrument, 0), start_time + duration)
        time_key = f"{start_time:.2f}"
        chord_patterns[time_key].append((instrument, pitch))

    # Convert notes to text format
    note_text = notes_to_text(notes)
    
    return (
        "You are a musical AI composer. Continue this piece naturally while maintaining its musical character.\n\n"
        
        "Technical Format:\n"
        "- Instruments: P (Piano), G (Guitar), B (Bass), S (Strings), D (Drums)\n"
        "- Note format: Instrument:pitch:start_time:duration\n"
        "- Times in seconds (0.50 duration)\n\n"
        
        "Musical Requirements:\n"
        f"- Continue each instrument from its last note:\n"
        + "\n".join([f"  * {instr}: from {last_times[instr]:.2f}, generate 32 notes"
                    for instr in sorted(instruments_present)])
        + "\n\n"
        
        "Composition Guidelines:\n"
        "- Make sure you analyze the original piece thoroughly and understand its musical character and style\n"
        "- Maintain natural musical flow and expression\n"
        "- Keep instruments coordinated but not rigidly synchronized\n"
        "- Allow subtle timing variations for musicality\n"
        "- Preserve the harmonic relationships between instruments\n"
        "- Introduce melodic variations while keeping the style\n"
        "- Let the music breathe - not every instrument needs to play at every moment\n\n"
        
        f"Current musical sequence:\n{note_text}\n\n"
        
        "Continue the music naturally. Respond only with the note sequence."
    )

def generate_continuation(notes: List[Tuple[int, float, float, str]]) -> Optional[List[Tuple[int, float, float, str]]]:
    """
    Generate a continuation of the musical sequence using GPT.
    
    Args:
        notes (List[Tuple[int, float, float, str]]): List of note tuples containing:
            - pitch (int): MIDI note number
            - start_time (float): Note start time in seconds
            - duration (float): Note duration in seconds
            - instrument (str): Instrument name
            
    Returns:
        Optional[List[Tuple[int, float, float, str]]]: Generated continuation notes
            in same format as input, or None if generation fails

    Raises:
        Various OpenAI API exceptions handled internally:
            - insufficient_quota: API quota exceeded
            - rate_limit: API rate limit reached
            - Other API errors with appropriate messages
    """
    try:
        prompt = create_gpt_prompt(notes)
        print("\n=== GPT Prompt ===")
        print(prompt)
        print("\n=== End Prompt ===\n")

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a music generation assistant that continues musical sequences."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        continuation_text = response.choices[0].message.content.strip()
        print("\n=== GPT Response ===")
        print(continuation_text)
        print("\n=== End Response ===\n")
        
        return text_to_notes(continuation_text)
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        if "insufficient_quota" in error_msg:
            print(f"OpenAI API quota exceeded. Please check your billing details.")
        elif "rate_limit" in error_msg:
            print(f"OpenAI API rate limit reached. Please try again later.")
        else:
            print(f"GPT generation failed: {error_type} - {error_msg}")
        
        return None

def create_midi_from_notes(notes: List[Tuple[int, float, float, str]]) -> bytes:
    """
    Create a MIDI file from note data.
    
    Args:
        notes (List[Tuple[int, float, float, str]]): List of note tuples containing:
            - pitch (int): MIDI note number
            - start_time (float): Note start time in seconds
            - duration (float): Note duration in seconds
            - instrument (str): Instrument name
            
    Returns:
        bytes: MIDI file content as bytes buffer

    The function:
        - Creates appropriate MIDI instruments with correct programs
        - Sets drum track for percussion
        - Uses consistent velocity (100) for all notes
        - Preserves exact timing from input notes
    """
    midi = pretty_midi.PrettyMIDI()
    instruments = {}
    
    for pitch, start_time, duration, instrument_name in notes:
        if instrument_name not in instruments:
            program = 0 if instrument_name == 'Piano' else (
                25 if instrument_name == 'Guitar' else
                32 if instrument_name == 'Bass' else
                48 if instrument_name == 'Strings' else 0
            )
            is_drum = instrument_name == 'Drums'
            instrument = pretty_midi.Instrument(
                program=program,
                is_drum=is_drum,
                name=instrument_name
            )
            instruments[instrument_name] = instrument
            midi.instruments.append(instrument)
        
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=start_time,  # Use actual start time
            end=start_time + duration
        )
        instruments[instrument_name].notes.append(note)
    
    midi_buffer = io.BytesIO()
    midi.write(midi_buffer)
    return midi_buffer.getvalue()

def generate_music_gpt(input_midi_path: str) -> Tuple[Optional[bytes], Optional[bytes]]:
    """
    Generate music continuation using GPT model.
    
    Args:
        input_midi_path (str): Path to input MIDI file
        
    Returns:
        Tuple[Optional[bytes], Optional[bytes]]: Tuple containing:
            - MIDI file content as bytes
            - WAV file content as bytes
            Returns (None, None) if generation fails

    The function:
        1. Loads and parses input MIDI
        2. Extracts first 64 time steps
        3. Generates continuation using GPT
        4. Combines original and continuation
        5. Creates MIDI and WAV outputs

    Example:
        >>> midi_data, wav_data = generate_music_gpt("input.mid")
        >>> if midi_data:
        >>>     with open("output.mid", "wb") as f:
        >>>         f.write(midi_data)
    """
    try:
        # Load and parse input MIDI
        multitrack = pypianoroll.read(input_midi_path)
        multitrack.set_resolution(2)  # 2 steps per second
        
        # Convert to tensor format - now taking first 64 time steps
        pianoroll = torch.tensor(np.stack([
            track.pianoroll[:64] for track in multitrack.tracks
        ]), dtype=torch.float32)
        
        # Extract notes from first 64 time steps
        initial_notes = extract_notes_from_pianoroll(pianoroll)
        
        # Generate continuation
        continuation = generate_continuation(initial_notes)
        if continuation is None:
            return None, None
        
        # Combine initial and continuation notes
        full_sequence = initial_notes + continuation
        
        # Create MIDI file
        midi_content = create_midi_from_notes(full_sequence)
        
        # Generate WAV audio
        midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_content))
        audio = midi.synthesize(fs=44100)
        normalized_audio = (audio / np.max(np.abs(audio)) * 32767).astype(np.int16)
        
        wav_buffer = io.BytesIO()
        scipy.io.wavfile.write(wav_buffer, 44100, normalized_audio)
        wav_content = wav_buffer.getvalue()
        
        return midi_content, wav_content
        
    except Exception as e:
        print(f"Music generation failed: {str(e)}")
        return None, None

def generate_music_api(input_midi_path: str) -> Tuple[Optional[bytes], Optional[bytes]]:
    """
    API endpoint for GPT music generation.
    
    Args:
        input_midi_path (str): Path to input MIDI file
        
    Returns:
        Tuple[Optional[bytes], Optional[bytes]]: Tuple containing:
            - MIDI file content as bytes
            - WAV file content as bytes
            Returns (None, None) if generation fails

    This is the main API entry point that should be used by external applications.
    It provides the same functionality as generate_music_gpt() but with a simpler
    interface focused on API usage.

    Example:
        >>> from gpt_generation import generate_music_api
        >>> midi_data, wav_data = generate_music_api("input.mid")
        >>> if midi_data and wav_data:
        >>>     with open("output.mid", "wb") as f:
        >>>         f.write(midi_data)
        >>>     with open("output.wav", "wb") as f:
        >>>         f.write(wav_data)
    """
    return generate_music_gpt(input_midi_path)

def test_api_connection() -> bool:
    """
    Test the OpenAI API connection and quota.

    Returns:
        bool: True if connection successful, False otherwise

    This function:
        1. Makes a minimal API call to verify connectivity
        2. Catches and reports any authentication or quota issues
        3. Provides clear error messages for troubleshooting

    Example:
        >>> if test_api_connection():
        >>>     print("API ready to use")
        >>> else:
        >>>     print("Please check API configuration")
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'API is working'"}
            ],
            max_tokens=10
        )
        print("API Connection Test: Success")
        return True
    except Exception as e:
        print(f"API Connection Test Failed: {str(e)}")
        return False

if __name__ == "__main__":
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment variables")
        exit(1)
    
    print(f"Using API key: {os.getenv('OPENAI_API_KEY')[:6]}...")
    
    if test_api_connection():
        path = "inputmidi.mid"
        generate_music_api(path)
    else:
        print("Please verify your API key and billing status at https://platform.openai.com/account/billing")