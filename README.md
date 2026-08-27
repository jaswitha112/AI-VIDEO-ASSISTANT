# AI Video Assistant

AI Video Assistant turns a YouTube recording or local audio/video file into a searchable meeting knowledge base. It transcribes the recording, generates a concise summary, extracts action items and decisions, and lets you ask questions about the transcript.

The project includes both a Streamlit web interface and an interactive command-line interface.

## Features

- Process YouTube URLs and local audio/video files.
- Convert audio to mono, 16 kHz WAV and split long recordings into chunks.
- Transcribe English recordings locally with OpenAI Whisper.
- Transcribe Hinglish recordings and translate them to English with Sarvam AI.
- Generate a meeting title and summary with Mistral.
- Extract action items, key decisions, and unresolved questions.
- Build a local Chroma vector store for retrieval-augmented question answering.
- Chat with the processed meeting from the Streamlit UI or CLI.

## Architecture

```text
YouTube URL / local media file
			  |
			  v
	  Audio processing and chunking
			  |
			  v
  Whisper (English) or Sarvam (Hinglish)
			  |
			  v
		  Transcript
	   /      |       \
	  v       v        v
  Mistral  Mistral   Mistral
  summary  extraction title
			  |
			  v
	 Chroma + HuggingFace embeddings
			  |
			  v
	   Retrieval-based meeting chat
```

### Project structure

```text
.
├── app.py                    # Streamlit web application
├── main.py                   # Command-line application and pipeline entry point
├── requirements.txt          # Python dependencies
├── core/
│   ├── extractor.py          # Action items, decisions, and questions
│   ├── rag_engine.py         # Retrieval-augmented generation chain
│   ├── summarizer.py         # Title and summary generation
│   ├── transcriber.py        # Whisper and Sarvam transcription
│   └── vector_store.py       # Chroma persistence and retrieval
└── utils/
	└── audio_processor.py    # Downloading, conversion, and chunking
```

## Requirements

- Python 3.10 or newer
- FFmpeg available on your `PATH`
- A Mistral API key
- A Sarvam API key when using `hinglish` transcription
- A machine with enough disk space and memory for the selected Whisper model

On Windows, FFmpeg can be installed with one of these commands when the package manager is available:

```powershell
winget install Gyan.FFmpeg
# or
choco install ffmpeg
```

## Installation

Clone the repository and create an isolated environment:

```bash
git clone <repository-url>
cd AI-VIDEO-ASSISTANT
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install langchain-text-splitters langchain-chroma
```

The last command installs the two LangChain integration packages imported directly by the current source tree.

## Configuration

Create a `.env` file in the project root:

```dotenv
MISTRAL_API_KEY=your_mistral_api_key
SARVAM_API_KEY=your_sarvam_api_key

# Optional Whisper settings
WHISPER_MODEL=small
SARVAM_STT_MODEL=saaras:v2.5
```

`MISTRAL_API_KEY` is required for title generation, summarisation, extraction, and meeting chat. `SARVAM_API_KEY` is required only when the selected language is `hinglish`. Whisper downloads its model the first time it is used.

Do not commit `.env` or API keys to source control.

## Usage

### Streamlit web app

```bash
streamlit run app.py
```

Enter a YouTube URL or local file path in the sidebar, choose `english` or `hinglish`, and select **Analyse**. After processing, the app displays the transcript, summary, extracted meeting information, and a chat interface.

### Command line

```bash
python main.py
```

Follow the prompts for the input source and language. Type `exit`, `quit`, or `q` to leave the meeting chat.

Supported local inputs are media files readable by FFmpeg, such as MP4, WAV, MP3, and M4A. YouTube inputs must be supplied as an HTTP or HTTPS URL.

## Generated data

Runtime files are written to local directories in the project root:

- `downloades/` stores downloaded YouTube audio.
- `vector_db/` stores the persisted Chroma collection.
- Converted audio and temporary chunk files are created beside their source files.

These files can become large and should generally be excluded from version control.

## Development

Run the available test script with:

```bash
python test.py
```

Contributions are welcome. For a pull request, please describe the change, keep credentials and generated media out of commits, and include a focused test or reproduction when behavior changes.

## License

No license file is currently included. Add a license before distributing or accepting external contributions under an open-source license.
