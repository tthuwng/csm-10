import torch
import torchaudio
import time
from typing import Generator as PyGenerator, Optional, List
import traceback

from generator import load_csm_1b, Segment, Generator
MODEL_AVAILABLE = True


class VoiceGenerator:
    def __init__(self) -> None:
        """Initialize device and basic attributes."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                try:
                    torch.backends.cuda.enable_flash_sdp(True)
                    print("Flash Attention enabled.")
                except Exception as e:
                    print(f"Warning: Could not enable Flash Attention: {e}")
        
        self.generator: Optional[Generator] = None
        self.sample_rate: int = 24000
        self.conversation_history: List[Segment] = []
        self.max_context_segments: int = 5

    def initialize(self) -> None:
        """Load model and warm up."""
        if not MODEL_AVAILABLE:
            print("Skipping model initialization due to import failure.")
            return
        
        try:
            self.generator = load_csm_1b(self.device)
            self.sample_rate = self.generator.sample_rate
            print(f"Model loaded on {self.device}. Sample rate: {self.sample_rate}")
            self.warm_up()
        except Exception as e:
            print(f"Failed to initialize model: {e}")
            self.generator = None

    def warm_up(self) -> None:
        """Warm up the model to reduce time to first chunk."""
        if self.generator is None:
            return
        
        print("Warming up model...")
        dummy_text = "This is a warm-up text to prepare the model for fast responses."
        speaker_id = 0
        
        try:
            _ = self.generator.generate(
                text=dummy_text,
                speaker=speaker_id,
                context=[],
                max_audio_length_ms=5000,
                temperature=0.9,
                topk=50
            )
            print("Model warm-up complete.")
        except Exception as e:
            print(f"Warning: Model warm-up failed: {e}")

    def _resample_audio(self, audio: torch.Tensor, target_sample_rate: int) -> torch.Tensor:
        """Resample audio to target sample rate if needed."""
        if self.sample_rate != target_sample_rate:
            return torchaudio.functional.resample(
                audio.cpu(),
                orig_freq=self.sample_rate,
                new_freq=target_sample_rate
            )
        return audio.cpu()

    def _convert_to_pcm(self, audio: torch.Tensor) -> bytes:
        """Convert audio tensor to 16-bit PCM bytes."""
        return (audio * 32767).short().numpy().tobytes()

    def generate_pcm_stream(
        self,
        text: str,
        speaker_id: int,
        target_sample_rate: int,
        max_audio_length_ms: int = 30_000,
        temperature: float = 0.9,
        topk: int = 50,
        buffer_frames: int = 3
    ) -> PyGenerator[bytes, None, None]:
        """Generate a stream of 16-bit PCM bytes, buffering slightly for smoothness."""
        if self.generator is None:
            raise RuntimeError("Generator not initialized.")
        if buffer_frames <= 0:
            raise ValueError("buffer_frames must be positive")

        print(f"\n--- Starting PCM Stream Generation (Buffering {buffer_frames} frames) ---")
        print(f"Input text: '{text}'")
        print(f"Speaker ID: {speaker_id}")
        stream_start_time = time.monotonic()
        first_frame_time_logged = False
        current_audio_frames: List[torch.Tensor] = [] # For history

        try:
            context = self.get_context()
            print(f"Using context with {len(context)} segments")

            for frame in self.generator.generate_stream(
                text=text,
                speaker=speaker_id,
                context=context,
                max_audio_length_ms=max_audio_length_ms,
                temperature=temperature,
                topk=topk
            ):

                # Process frame: Ensure 1D, keep for history, resample, convert to PCM
                processed_frame = frame.squeeze(-1) if frame.ndim > 1 else frame
                current_audio_frames.append(processed_frame) # For history

                resampled_frame = self._resample_audio(processed_frame, target_sample_rate)
                pcm_bytes = self._convert_to_pcm(resampled_frame)
                yield pcm_bytes 

            if current_audio_frames:
                full_audio = torch.cat(current_audio_frames)
                self.add_to_history(text, speaker_id, full_audio)
            else:
                print("Warning: No audio frames generated, skipping history update.")

        except Exception as e:
            print(f"!! ERROR during PCM stream generation: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            raise 

    def generate_pcm_full(
        self,
        text: str,
        speaker_id: int,
        target_sample_rate: int,
        max_audio_length_ms: int = 30_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> bytes:
        """Generate complete 16-bit PCM audio, save to WAV, and return bytes."""
        if self.generator is None:
            raise RuntimeError("Generator not initialized.")

        print(f"\n--- Starting Full PCM Generation ---")
        print(f"Input text: '{text}'")
        print(f"Speaker ID: {speaker_id}")
        generation_start_time = time.monotonic()

        try:
            context = self.get_context()
            print(f"Using context with {len(context)} segments")

            print("Generating audio...")
            audio_data = self.generator.generate(
                text=text,
                speaker=speaker_id,
                context=context,
                max_audio_length_ms=max_audio_length_ms,
                temperature=temperature,
                topk=topk
            )

            print(f"Audio tensor generated - shape: {audio_data.shape}")
            if audio_data.ndim > 1:
                audio_data = audio_data.squeeze(-1)
                print(f"Audio tensor squeezed - new shape: {audio_data.shape}")
            self.add_to_history(text, speaker_id, audio_data)
            resampled_audio = self._resample_audio(audio_data, target_sample_rate)
            pcm_bytes = self._convert_to_pcm(resampled_audio)

            generation_time = time.monotonic() - generation_start_time
            print(f"Full PCM generation completed in {generation_time:.3f}s @ {target_sample_rate}Hz")
            print(f"Generated {len(pcm_bytes)} bytes of audio data")

            return pcm_bytes

        except Exception as e:
            print(f"!! ERROR during full PCM generation: {e}")
            print(f"Error type: {type(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            raise

    def add_to_history(self, text: str, speaker_id: int, audio: torch.Tensor) -> None:
        """Add a new segment to conversation history."""
        segment = Segment(text=text, speaker=speaker_id, audio=audio)
        self.conversation_history.append(segment)
        if len(self.conversation_history) > self.max_context_segments:
            self.conversation_history = self.conversation_history[-self.max_context_segments:]
        if self.generator:
            self.generator.update_ctx_tokens(self.conversation_history)

    def get_context(self) -> List[Segment]:
        """Get current conversation context."""
        return self.conversation_history 