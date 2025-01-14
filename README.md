# Gemma 2 on Jetson Orin Nano - Google Dev Day Tokyo Demos

If you're working with a Jetson Orin Nano and want to explore the power of **Gemma 2**, this repository provides three easy to follow demos showcasing how to run a Small Language Model (SLM) with 2 billion parameters on this device. In my talk at the **Google Gemma 2 Dev Day in Tokyo**, I covered everything step-by-step, from setting up the device to running these demos efficiently. For a complete walkthrough, check out the video below.

## YouTube Video
This video offers a detailed guide on setting up **Gemma 2** on the Jetson Orin Nano, covering installation, configuration, and execution of the demos. Watch it to get a better understanding of how to replicate the steps on your device:

[![Watch the video](https://img.youtube.com/vi/Kd7VJ-TKb8I/maxresdefault.jpg)](https://www.youtube.com/watch?v=Kd7VJ-TKb8I)

## Demos

1. **Home Assistant Demo**  
   This demo showcases a basic home assistant powered by Gemma 2, using Whisper for speech-to-text, FAISS for Retrieval-Augmented Generation (RAG), and Piper for text-to-speech. Gemma 2 runs on port 8080 using `llama.cpp` to process the input and generate responses.

   - Key components:  
     - Whisper: Transcribes voice input into text.
     - FAISS: Uses RAG to find relevant context from a set of predefined documents.
     - Gemma 2: Runs on port 8080, responsible for generating concise and relevant answers based on the input.
     - Piper: Converts the generated text into speech.
   - How it works:  
     When the system detects voice input (indicated by beeps due to the lack of silence detection), it uses Whisper to transcribe the text. FAISS enhances this input with external context, and Gemma 2, running on port 8080 via `llama.cpp`, processes the input and provides a concise response. Finally, Piper converts this response into speech.

2. **Translation Assistant Demo**  
   This demo implements a translation assistant that listens to English input, translates it into Japanese, and speaks the translation using Coqui TTS. Gemma 2 also powers this translation process, running on port 8080.

   - Key components:  
     - Whisper: Transcribes English voice input.
     - Gemma 2: Translates the input from English to Japanese, running on port 8080 with `llama.cpp`.
     - Coqui TTS: Synthesizes the translated Japanese text into speech.
   - How it works:  
     After receiving and transcribing voice input via Whisper, Gemma 2 translates the text using the model running on port 8080. The translated text is then synthesized into spoken Japanese using Coqui TTS.

3. **Multi-Agent System Demo**  
   This demo demonstrates a multi-agent setup where two instances of Gemma 2 communicate with each other using different personalities. One instance of Gemma 2 runs on port 8080 and the other on port 8082. Both models, powered by `llama.cpp`, simulate a conversation between two distinct personalities: one creative and one analytical.

   - Key components:  
     - Gemma 2: Two instances of the model running on ports 8080 and 8082, respectively.
     - FAISS: Optionally used for memory and enhanced context handling.
   - How it works:  
     Two separate instances of Gemma 2 are launched on different ports, with one being more creative and the other more analytical. These instances engage in a simulated conversation, showcasing the multi-agent capabilities of Gemma 2 running on an edge device.

## Execution

The code is very straightforward to run. It's essentially a Python script with the dependencies listed in `requirements.txt`.

1. Install the dependencies with:
   ^^^bash
   pip install -r requirements.txt
   ^^^

2. Here's what each script does:
   - `home_assistant.py`: Runs a basic voice-activated assistant using Whisper for speech-to-text, FAISS for context retrieval, and Gemma 2 (on port 8080) for response generation, with Piper for text-to-speech.
   - `translate.py`: Implements a voice-to-voice translation system from English to Japanese using Whisper for transcription, Gemma 2 (on port 8080) for translation, and Coqui TTS for speech synthesis.
   - `multi_agent.py`: Runs two instances of Gemma 2 (on ports 8080 and 8082) as a multi-agent system, simulating a conversation between two different personalities.

### Assets

There is an `assets` folder containing two `.wav` files (beeps) that signal when you can start and stop speaking. These beeps are used because the home assistant does not implement silence detection. The purpose of this code is to show how a minimal home assistant can be built with just a few lines of code.

### Assets

There is an `assets` folder containing two `.wav` files (beeps) that signal when you can start and stop speaking. These beeps are used because the home assistant does not implement silence detection. The purpose of this code is to show how a minimal home assistant can be built with just a few lines of code.

## Directory Structure

The project is organized as follows:

```
GEMMALLAMA/
│
├── assets/
│   ├── bip.wav      # Sound file to indicate the start of a recording
│   └── bip2.wav     # Sound file to indicate the end of a recording
│
├── npcs/
│   └── conversation_log.html  # Directory where NPC conversations will be stored in HTML format
│
├── assistant.py     # Script for the Home Assistant demo, using Whisper, FAISS, and Piper
├── npcservers.py    # Script for the multi-agent system demo, where two Gemma instances interact
└── translate.py     # Script for the translation assistant demo, using Whisper and Coqui TTS
```

- **assets/**: Contains audio files (`bip.wav`, `bip2.wav`) used to indicate when recording starts and ends.
- **npcs/**: This folder stores conversation logs in HTML format. For example, when two NPCs (Gemma and Gemmo) converse, their dialogue is saved here.
- **assistant.py**: Implements a voice-activated assistant using Whisper for transcription, FAISS for retrieval, and Piper for text-to-speech.
- **npcservers.py**: Simulates conversations between two LLaMA-based agents (Gemma and Gemmo) with different personalities.
- **translate.py**: Translates speech from English to Japanese using Whisper and Coqui TTS for speech synthesis.

