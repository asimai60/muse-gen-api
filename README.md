# Multi-Model Music Generator API

A sophisticated FastAPI-based application that generates music using multiple AI approaches: Variational Autoencoder (VAE), Recurrent Neural Network (RNN), and GPT-based generation. The system processes MIDI files to create new musical compositions with support for multiple instruments and styles.

---

## ✨ Features

### 🎵 Multiple Generation Approaches
- **VAE Generation**: Multi-instrumental tracks (piano, guitar, bass, strings, drums)
- **RNN Generation**: Piano-focused melodic sequences
- **GPT Generation**: Natural language model-based composition
- **Real-time Audio Synthesis**: WAV file generation from MIDI

### 🛠️ Technical Capabilities
- Multi-track instrument support
- Conditional generation between instruments
- Temperature-based sampling for creative control
- Cloud storage integration
- Docker containerization
- Robust error handling and validation

---

## 🏗️ Architecture

### 🧠 VAE Components
- **ConvVAE**: Convolutional VAE for pattern learning
- **ConditionalNN**: Harmony generation network
- **MelodyNN**: Specialized melody sequence generator
- **Multi-track Support**: Separate models per instrument

### 🔄 RNN Components
- **GenerationRNN**: GRU-based sequence generator
- **Embedding Layer**: Note representation learning
- **Multinomial Sampling**: Creative generation control

### 🤖 GPT Components
- **Text-based Generation**: Musical sequence to text conversion
- **Prompt Engineering**: Musically-aware context creation
- **Response Parsing**: Conversion back to MIDI format

---

## 🚀 API Endpoints

### Process MIDI File
```http
POST /process-midi/{model_type}/
```
#### Parameters
- `model_type`: 'vae', 'rnn', or 'gpt'
- `file`: MIDI file upload

#### Response
```json
{
  "midi_url": "https://storage.googleapis.com/bucket/file.mid",
  "wav_url": "https://storage.googleapis.com/bucket/file.wav",
  "model_type": "vae"
}
```

### Download Generated File
```http
GET /download-midi/
```
#### Parameters
- `url`: Signed URL from process endpoint

---

## 🔧 Setup

### Prerequisites
- Python 3.9+
- Docker
- Google Cloud Storage access
- OpenAI API key (for GPT generation)

### Installation

1. **Clone Repository**:
   ```bash
   git clone https://github.com/asimai60/music-generator-api.git
   cd music-generator-api
   ```

2. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

3. **Configuration**:
   ```bash
   # Create .env file
   OPENAI_API_KEY=your_api_key_here
   
   # Place Google Cloud credentials in:
   muse-gen-midi-files-keys.json
   ```

4. **Docker Build**:
   ```bash
   docker build -t music-generator-api .
   ```

5. **Run Container**:
   ```bash
   docker run -p 8080:8080 music-generator-api
   ```

---

## 📁 Project Structure

```
├── vae_app.py           # FastAPI application
├── conv_vae.py          # VAE model architecture
├── RNNGeneration.py     # RNN implementation
├── gpt_generation.py    # GPT-based generation
├── vae_helpers.py       # Utility functions
├── vae_model_generate.py # VAE generation logic
├── Dockerfile           # Container configuration
└── requirements.txt     # Dependencies
```

---

## 🤖 Model Details

### VAE Model
- Latent space: 32 dimensions
- Multiple instrument-specific models
- Conditional generation capabilities
- Convolutional architecture

### RNN Model
- GRU-based architecture
- Embedding dimension: 96
- 2-layer configuration
- Temperature-based sampling

### GPT Model
- GPT-4 based generation
- Musical pattern recognition
- Style-aware continuation
- Multi-instrument coordination

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
