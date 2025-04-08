import io
import time
from typing import Generator as PyGenerator, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
from model_handler import VoiceGenerator


app = FastAPI()
voice_generator = VoiceGenerator()
voice_generator.initialize()


def _create_error_response(message: str, status_code: int, sample_rate: int) -> StreamingResponse:
    """Create an error response with the appropriate media type."""
    return StreamingResponse(
        iter([b'']),
        status_code=status_code,
        media_type=f'audio/l16; rate={sample_rate}; channels=1'
    )


@app.post('/custom-voice-tts')
async def handle_custom_voice_request(request: Request):
    """Handle TTS requests with streaming or non-streaming output."""
    request_start_time = time.monotonic()
    target_sr = 16000  # target telephony rate

    try:
        data = await request.json()
        message_data = data.get('message', {})
        text_to_speak = message_data.get('text')
        requested_sample_rate = message_data.get('sampleRate', target_sr)
        stream_mode = message_data.get('stream', True)
        call_id = message_data.get('call', {}).get('id', 'N/A')
        speaker_id = 0 

        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} | Call ID: {call_id}")
        print(f"Text: \"{text_to_speak[:150]}{'...' if text_to_speak and len(text_to_speak) > 150 else ''}\"")
        print(f"Mode: {'Streaming' if stream_mode else 'Non-streaming'}")

        if not text_to_speak:
            print("ERROR: No text provided.")
            return _create_error_response("No text provided", 400, requested_sample_rate)

        print(f"Starting generation. Handler setup: {(time.monotonic() - request_start_time) * 1000:.2f} ms.")
        
        if stream_mode:
            pcm_stream = voice_generator.generate_pcm_stream(
                text=text_to_speak,
                speaker_id=speaker_id,
                target_sample_rate=requested_sample_rate,
                max_audio_length_ms=30_000,
                temperature=0.9,
                topk=50
            )
            return StreamingResponse(
                pcm_stream,
                media_type=f'audio/l16; rate={requested_sample_rate}; channels=1'
            )
        else:
            pcm_bytes = voice_generator.generate_pcm_full(
                text=text_to_speak,
                speaker_id=speaker_id,
                target_sample_rate=requested_sample_rate,
                max_audio_length_ms=30_000,
                temperature=0.9,
                topk=50
            )
            return Response(
                content=pcm_bytes,
                media_type=f'audio/l16; rate={requested_sample_rate}; channels=1'
            )

    except RuntimeError as e:
        if str(e) == "Generator not initialized.":
            print("ERROR: Generator not initialized.")
            return _create_error_response("Service unavailable", 503, requested_sample_rate)
        print(f"Error generating audio: {e}")
        return _create_error_response("Internal server error", 500, requested_sample_rate)
    except Exception as e:
        print(f"Server error: {e}")
        return _create_error_response("Internal server error", 500, requested_sample_rate)


@app.get('/ping')
async def ping():
    """Health check endpoint indicating model status."""
    if voice_generator.generator is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "message": "Model ready"}