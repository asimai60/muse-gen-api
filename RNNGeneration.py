"""
RNN-based Music Generation Module

This module implements a recurrent neural network (RNN) for generating musical sequences.
It provides functionality to load trained models, process MIDI files, and generate new musical
compositions based on input sequences.

The module uses PyTorch for the neural network implementation and music21 for MIDI file handling.

Key Components:
- GenerationRNN: The main RNN model class
- Note mapping functions for converting between notes and integers
- MIDI processing and generation utilities
- API endpoints for integration with web services

Requirements:
    - PyTorch
    - music21
    - numpy
    - scipy
    - pretty_midi
"""

import itertools
import music21 as m21
import pickle
import torch
import torch.nn as nn
import os
import io
import numpy as np
import scipy.io.wavfile
import pretty_midi

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GenerationRNN(nn.Module):
    """
    Recurrent Neural Network model for music generation.
    
    This model uses an embedding layer followed by GRU layers and a linear decoder
    to generate musical sequences.
    
    Args:
        input_size (int): Size of the input vocabulary (number of unique notes)
        hidden_size (int): Number of features in the hidden state
        output_size (int): Size of the output vocabulary (same as input_size)
        n_layers (int, optional): Number of GRU layers. Defaults to 1
    """
    
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(GenerationRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size * n_layers, output_size)

    def forward(self, x, hidden):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor
            hidden (torch.Tensor): Hidden state
            
        Returns:
            tuple: (output, hidden_state)
        """
        embedded = self.embedding(x.view(1, -1))
        output, hidden = self.gru(embedded, hidden)
        output = self.decoder(hidden.view(1, -1))
        return output, hidden

    def init_hidden(self):
        """Initialize hidden state with zeros.
        
        Returns:
            torch.Tensor: Initialized hidden state
        """
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)

def load_note_mappings():
    """
    Load note mappings from pickle file and create bidirectional mappings.
    
    Returns:
        tuple: (notes_list, note_to_int_dict, int_to_note_dict)
    
    Raises:
        FileNotFoundError: If the notes pickle file is not found
    """
    notes_path = os.path.join(os.path.dirname(__file__), "notes pickle", "notes.pkl")
    
    with open(notes_path, 'rb') as f:
        notes = pickle.load(f)

    unique_notes = sorted(set(itertools.chain(*notes)))
    note_to_int = {note: i for i, note in enumerate(unique_notes)}
    int_to_note = {i: note for note, i in note_to_int.items()}
    
    return notes, note_to_int, int_to_note

def load_model(input_size=None, hidden_size=96, n_layers=2):
    """
    Initialize and load trained model weights.
    
    Args:
        input_size (int, optional): Size of input vocabulary
        hidden_size (int, optional): Size of hidden layers. Defaults to 96
        n_layers (int, optional): Number of GRU layers. Defaults to 2
        
    Returns:
        GenerationRNN: Loaded model instance
        
    Raises:
        FileNotFoundError: If model weights file is not found
    """
    if input_size is None:
        _, note_mappings, _ = load_note_mappings()
        input_size = len(note_mappings)
    
    model = GenerationRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=input_size,
        n_layers=n_layers
    ).to(device)

    model_path = os.path.join(os.path.dirname(__file__), "Saved Models", "RNN", "model0.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def evaluate_multinomial(model, seed_sequence, sequence_length, temperature=0.8):
    """
    Generate a sequence of notes using multinomial sampling.
    
    Args:
        model (GenerationRNN): The trained model
        seed_sequence (list): Initial sequence to seed the generation
        sequence_length (int): Length of sequence to generate
        temperature (float, optional): Sampling temperature. Defaults to 0.8
        
    Returns:
        list: Generated sequence of note indices
    """
    hidden = model.init_hidden()
    generated = seed_sequence.copy()
    seed_tensor = torch.tensor(seed_sequence, dtype=torch.long, device=device)

    # Initialize hidden state with seed sequence
    for note in seed_tensor[:-1]:
        _, hidden = model(note, hidden)

    current_input = seed_tensor[-1]

    # Generate new sequence
    for _ in range(sequence_length):
        output, hidden = model(current_input, hidden)
        output_dist = output.data.view(-1).div(temperature).exp()
        next_note = torch.multinomial(output_dist, 1)
        generated.append(next_note.item())
        current_input = next_note

    return generated

def create_midi_stream(prediction_output):
    """
    Convert prediction output to a music21 stream.
    
    Args:
        prediction_output (list): List of strings representing notes/chords
        
    Returns:
        music21.stream.Stream: Stream containing the musical notes/chords
    """
    offset = 0
    notes = []
    piano = m21.instrument.Piano()

    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            # Create chord from dot-separated note numbers
            chord_notes = [m21.note.Note(int(pitch), instrument=piano) 
                         for pitch in pattern.split('.')]
            chord = m21.chord.Chord(chord_notes)
            chord.offset = offset
            notes.append(chord)
        else:
            # Create single note
            note = m21.note.Note(pattern)
            note.offset = offset
            note.storedInstrument = piano
            notes.append(note)

        offset += 0.5

    return m21.stream.Stream(notes)

def extract_piano_sequence(midi_data):
    """
    Extract piano notes from MIDI file.
    
    Args:
        midi_data (music21.stream.Score): Parsed MIDI data
        
    Returns:
        list: Sequence of note/chord strings
        
    Raises:
        ValueError: If no piano part is found in the MIDI file
    """
    parts = m21.instrument.partitionByInstrument(midi_data)
    
    piano_part = None
    for part in parts:
        if isinstance(part.getInstrument(), m21.instrument.Piano):
            piano_part = part
            break
    
    if not piano_part:
        raise ValueError("No piano part found in the MIDI file. Please provide a MIDI file containing piano music.")

    sequence = []
    for element in piano_part:
        if isinstance(element, m21.note.Note):
            sequence.append(str(element.pitch))
        elif isinstance(element, m21.chord.Chord):
            sequence.append('.'.join(str(n) for n in element.normalOrder))

    return sequence

def generate_music_rnn(input_midi_path, sequence_length=500, temperature=1.2):
    """
    Generate music using the RNN model.
    
    Args:
        input_midi_path (str): Path to input MIDI file
        sequence_length (int, optional): Length of sequence to generate. Defaults to 500
        temperature (float, optional): Sampling temperature. Defaults to 1.2
        
    Returns:
        tuple: (midi_bytes, wav_bytes) or (None, None) if generation fails
        
    Raises:
        ValueError: If no piano part is found in the input MIDI file
    """
    # Resolve absolute path
    if not os.path.isabs(input_midi_path):
        input_midi_path = os.path.join(os.path.dirname(__file__), input_midi_path)
    
    # Load model and mappings
    _, note_to_int, int_to_note = load_note_mappings()
    model = load_model()
    
    # Process input MIDI
    midi_data = m21.converter.parse(input_midi_path)
    piano_sequence = extract_piano_sequence(midi_data)
    
    # Generate new sequence
    input_sequence = [note_to_int[note] for note in piano_sequence if note in note_to_int]
    generated_sequence = evaluate_multinomial(model, input_sequence, sequence_length, temperature)
    generated_notes = [int_to_note[idx] for idx in generated_sequence[-sequence_length:]]
    
    # Create MIDI stream and convert to bytes
    midi_stream = create_midi_stream(generated_notes)
    temp_midi_path = 'temp_output.mid'
    midi_stream.write('midi', fp=temp_midi_path)
    
    with open(temp_midi_path, 'rb') as f:
        midi_bytes = f.read()
    os.remove(temp_midi_path)
    
    # Generate WAV audio
    midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    audio = midi.synthesize(fs=44100)
    normalized_audio = (audio / np.max(np.abs(audio)) * 32767).astype(np.int16)
    
    wav_buffer = io.BytesIO()
    scipy.io.wavfile.write(wav_buffer, 44100, normalized_audio)
    wav_bytes = wav_buffer.getvalue()
    
    return midi_bytes, wav_bytes

def generate_music_api(input_midi_path):
    """
    API endpoint for RNN music generation.
    
    Args:
        input_midi_path (str): Path to input MIDI file
        
    Returns:
        tuple: (midi_bytes, wav_bytes) containing the generated music
        
    Raises:
        ValueError: If no piano part is found in the input MIDI file
    """
    return generate_music_rnn(input_midi_path)

if __name__ == "__main__":
    path = "inputmidi.mid"
    generate_music_api(path)