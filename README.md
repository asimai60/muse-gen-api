# VAE Music Generator

This project is a FastAPI-based application that processes MIDI files using a Variational Autoencoder (VAE) model to generate new music tracks. It leverages machine learning models to create music by analyzing and transforming MIDI files.

## Project Overview

The VAE Music Generator is designed to take an input MIDI file, process it using a trained VAE model, and generate a new music track. The application is containerized using Docker and is deployed on Google Cloud.

## Features

- Upload and process MIDI files to generate new music tracks.
- Download generated MIDI files.
- Containerized with Docker for easy deployment.
- Deployed on Google Cloud for accessibility.

## Setup Instructions

### Prerequisites

- Docker
- Python 3.9
- Access to an AWS S3 bucket for storing generated MIDI files.

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/asimai60/vae-music-generator.git
   cd vae-music-generator
   ```

2. **Build the Docker image:**

   ```bash
   docker build -t vae-music-generator .
   ```

3. **Run the Docker container locally (optional):**

   ```bash
   docker run -p 8080:8080 vae-music-generator
   ```

4. **Access the API:**

   The API is deployed and available at `https://muse-gen-api-883573555203.me-west1.run.app`.

## Usage Guide

### Process MIDI File

- **Endpoint:** `POST /process-midi/`
- **Description:** Upload a MIDI file to process and generate a new music track.
- **Request:**

  ```bash
  curl -X POST "https://muse-gen-api-883573555203.me-west1.run.app/process-midi/" -F "file=@path/to/your/file.mid"
  ```

- **Response:**

  ```json
  {
    "url": "https://your-bucket.s3.amazonaws.com/generated_track.mid"
  }
  ```

## Code Structure

- **`vae_app.py`:** Main application file containing API routes and middleware setup.
- **`vae_model_generate.py`:** Contains functions for loading models, processing MIDI files, and generating music.
- **`conv_vae.py`:** Defines the VAE and neural network models used for music generation.

## Contribution Guidelines

We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create a new branch for your feature or bugfix.

2. **Write clear, concise commit messages.**

3. **Ensure your code follows the project's style guide.**

4. **Write tests** for new features and bugfixes.

5. **Submit a pull request** with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact [your-email@example.com].
