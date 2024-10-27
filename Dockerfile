FROM python:3.9

WORKDIR /app

COPY requirements.txt .
COPY muse-gen-midi-files-keys.json /app/muse-gen-midi-files-keys.json
RUN apt-get update && apt-get install -y fluidsynth
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "vae_app:app", "--host", "0.0.0.0", "--port", "8080"]


