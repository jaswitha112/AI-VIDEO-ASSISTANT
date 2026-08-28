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