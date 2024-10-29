# VAE Music Generator

This project is a FastAPI-based application that processes MIDI files using both Variational Autoencoder (VAE) and Recurrent Neural Network (RNN) models to generate new music tracks. It leverages multiple machine learning approaches to create multi-instrumental music by analyzing and transforming MIDI files.

## Project Overview

The Music Generator supports two distinct approaches to music generation:
1. **VAE-based Generation**: Creates multi-instrumental tracks with piano, guitar, bass, strings, and drums
2. **RNN-based Generation**: Focuses on piano-specific generation with melodic continuity

The application is containerized using Docker and deployed on Google Cloud, with generated files stored in Google Cloud Storage.

## Features

- Dual model support (VAE and RNN) for diverse music generation approaches
- Multi-instrumental track generation (VAE model)
- Piano-focused melodic generation (RNN model)
- Upload and process MIDI files
- Download generated MIDI and WAV files
- Containerized with Docker for easy deployment
- Cloud storage integration for file management
- Real-time audio synthesis

## Technical Architecture

### VAE Components
- **ConvVAE**: Convolutional Variational Autoencoder for learning musical patterns
- **ConditionalNN**: Neural network for harmony generation
- **MelodyNN**: Specialized network for melody sequence generation
- **Multi-track Support**: Separate models for piano, guitar, bass, strings, and drums

### RNN Components
- **GenerationRNN**: GRU-based recurrent neural network for sequential note generation
- **Embedding Layer**: For learning note representations
- **Multinomial Sampling**: Temperature-based sampling for creative generation

## Setup Instructions

### Prerequisites

- Docker
- Python 3.9
- Google Cloud Storage access
- Required Python packages (see requirements.txt)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/asimai60/vae-music-generator.git
   cd vae-music-generator
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Build the Docker image:**
   ```bash
   docker build -t vae-music-generator .
   ```

4. **Configure Google Cloud credentials:**
   - Place your service account key file as `muse-gen-midi-files-keys.json` in the project root

5. **Run the Docker container:**
   ```bash
   docker run -p 8080:8080 vae-music-generator
   ```

## API Documentation

### Process MIDI File
- **Endpoint:** `POST /process-midi/{model_type}/`
- **Parameters:**
  - `model_type`: Either 'vae' or 'rnn'
  - `file`: MIDI file upload
- **Response:**
  ```json
  {
    "midi_url": "https://storage.googleapis.com/bucket/generated_midi.mid",
    "wav_url": "https://storage.googleapis.com/bucket/generated_audio.wav",
    "model_type": "vae"
  }
  ```

### Download Generated File
- **Endpoint:** `GET /download-midi/`
- **Parameters:**
  - `url`: Signed URL from the process endpoint

## Code Structure

### Core Components
- **`vae_app.py`:** FastAPI application and endpoint definitions
- **`conv_vae.py`:** VAE model architectures and neural networks
- **`vae_model_generate.py`:** Music generation logic for VAE approach
- **`RNNGeneration.py`:** RNN-based music generation implementation
- **`vae_helpers.py`:** Utility functions for VAE operations

### Helper Modules
- Note mapping and conversion utilities
- MIDI file processing functions
- Audio synthesis capabilities
- Cloud storage integration

## Model Architecture Details

### VAE Model
- Convolutional layers for both encoder and decoder
- Latent space dimension: 32
- Multiple instrument-specific models
- Conditional generation between instruments

### RNN Model
- GRU-based architecture
- Embedding dimension: 96
- 2-layer configuration
- Temperature-based sampling for generation

## Contribution Guidelines

We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create a new branch for your feature or bugfix
2. **Write clear, concise commit messages**
3. **Follow the project's code style**
4. **Add tests** for new features
5. **Update documentation** as needed
6. **Submit a pull request** with a detailed description

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact [your-email@example.com].
