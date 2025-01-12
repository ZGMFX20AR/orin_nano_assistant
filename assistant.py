import whisper, requests, os, sounddevice as sd, numpy as np, tempfile, wave
import faiss
from sentence_transformers import SentenceTransformer
import torch
import yaml  # Import PyYAML

# Optimization: Use a more efficient embedding model for Jetson Orin Nano
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Optimization: Explicitly use CUDA if available, with fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model("base").to(device)

# Configuration for local LLM server
llama_url = "http://127.0.0.1:8080/completion"

# Load personalities from YAML file
def load_personalities(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)['personalities']

personalities = load_personalities('personalities.yml')

# Function to prompt user to select personality via voice
def select_personality():
    print("Please say the name of the desired personality:")
    record_audio('temp_selection.wav')  # Record voice input
    selection = transcribe_audio('temp_selection.wav').strip().lower()
    for key, persona in personalities.items():
        if persona['name'].lower() in selection:
            return persona['prompt']
    return personalities['cheerful']['prompt']  # Default personality

# Set initial_prompt based on user selection
initial_prompt = select_personality()

# Current directory and path for beep sound files (used to indicate recording start and end)
current_dir = os.path.dirname(os.path.abspath(__file__))
bip_sound = os.path.join(current_dir, "assets/bip.wav")
bip2_sound = os.path.join(current_dir, "assets/bip2.wav")

# Documents to be used in Retrieval-Augmented Generation (RAG)
docs = [
    "The Jetson Nano is a compact, powerful computer designed by NVIDIA for AI applications at the edge.",
    "Developers can create AI assistants in under 100 lines of Python code using open-source libraries.",
    "Retrieval Augmented Generation enhances AI responses by combining language models with external knowledge bases.",
]


# Vector Database class to handle document embedding and search using FAISS
class VectorDatabase:
    def __init__(self, dim):
        # Create FAISS index with specified dimension (384 for SentenceTransformer embeddings)
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []
    
    # Add documents and their embeddings to the FAISS index
    def add_documents(self, docs):
        embeddings = embedding_model.encode(docs)  # Get embeddings for the docs
        self.index.add(np.array(embeddings, dtype=np.float32))  # Add them to the FAISS index
        self.documents.extend(docs)
    
    # Search for the top K most relevant documents based on query embedding
    def search(self, query, top_k=3):
        query_embedding = embedding_model.encode([query])[0].astype(np.float32)
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return [self.documents[i] for i in indices[0]]

# Create a VectorDatabase and add documents to it
db = VectorDatabase(dim=384)
db.add_documents(docs)

# Play sound (beep) to signal recording start/stop
def play_sound(sound_file):
    os.system(f"aplay {sound_file}")

# Record audio using sounddevice, save it as a .wav file
def record_audio(filename, duration=5, fs=16000):
    
    play_sound(bip_sound)  # Start beep
    print("5 seconds recording started...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait for the recording to complete
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())
    play_sound(bip2_sound)  # End beep
    print("recording completed")

# Transcribe recorded audio to text using Whisper
def transcribe_audio(filename):
    return whisper_model.transcribe(filename, language="en")['text']

# Send a query and context to LLaMA server for completion
def ask_llama(query, context):
    data = {
        "prompt": f"{initial_prompt}\nContext: {context}\nQuestion: {query}\nAnswer:",
        "max_tokens": 80,  # Limit response length to avoid delays
        "temperature": 0.7  # Adjust temperature for balanced responses
    }
    response = requests.post(llama_url, json=data, headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        return response.json().get('content', '').strip()
    else:
        return f"Error: {response.status_code}"

# Generate a response using Retrieval-Augmented Generation (RAG)
def rag_ask(query):
    context = " ".join(db.search(query))  # Search for related docs in the FAISS index
    return ask_llama(query, context)  # Ask LLaMA using the retrieved context

# Convert text to speech using Piper TTS model
def text_to_speech(text):
    os.system(f'echo "{text}" | /home/orin_nano/piper/build/piper --model /usr/local/share/piper/models/en_US-lessac-medium.onnx --output_file response.wav && aplay response.wav')

# New function: Save code to desktop
def save_code_to_desktop(code, language):
    import os
    from pathlib import Path

    # Get the desktop path
    desktop = Path.home() / "Desktop"
    # Create the file name
    filename = desktop / f"response_code.{language}"
    # Write the code to the file
    with open(filename, "w") as file:
        file.write(code)
    print(f"Code saved to desktop: {filename}")

# Convert text to speech using Piper TTS model or handle code saving
def handle_response(response):
    if response.startswith("```") and response.endswith("```"):
        # Parse the code block
        try:
            language = response.split("\n")[0][3:]
            code = "\n".join(response.split("\n")[1:-1])
            save_code_to_desktop(code, language)
            text_to_speech("Your code has been successfully saved to the desktop.")
        except Exception as e:
            text_to_speech("An error occurred while saving the code.")
            print(f"Error saving code: {e}")
    else:
        text_to_speech(response)

# Main loop for the assistant
def main():
    while True:
        # Create a temporary .wav file for the recording
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            record_audio(tmpfile.name)  # Record the audio input
            transcribed_text = transcribe_audio(tmpfile.name)  # Convert speech to text
            print(f"SentryBOT heard: {transcribed_text}")
            response = rag_ask(transcribed_text)  # Generate response using RAG and LLaMA
            print(f"SentryBOT response: {response}")
            if response:
                handle_response(response)  # Handle response (speech or save code)

# Entry point of the script
if __name__ == "__main__":
    main()
