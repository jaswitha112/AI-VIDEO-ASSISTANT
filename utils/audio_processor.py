import yt_dlp
import ffmpeg
import os

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

        for ext in [".webm", ".m4a", ".mp3"]:
            filename = filename.replace(ext, ".wav")

    return filename


def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to WAV format using FFmpeg.
    """
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    try:
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                ac=1,      # mono
                ar=16000   # 16 kHz
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        return output_path

    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg conversion failed:\n{error_msg}")


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """
    Returns the audio as a single chunk.
    """
    return [wav_path]


def process_input(source: str) -> list:
    """
    Main function used by app.py
    """
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Audio ready.")
    return chunk_audio(wav_path)