import os
import time
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment  # Uses the modified generator
from dataclasses import dataclass

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

# Default prompts are available at https://hf.co/sesame/csm-1b
prompt_filepath_conversational_a = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_a.wav"
)
prompt_filepath_conversational_b = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_b.wav"
)

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        ),
        "audio": prompt_filepath_conversational_a
    },
    "conversational_b": {
        "text": (
            "like a super Mario level. Like it's very like high detail. And like, once you get "
            "into the park, it just like, everything looks like a computer game and they have all "
            "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
            "will have like a question block. And if you like, you know, punch it, a coin will "
            "come out. So like everyone, when they come into the park, they get like this little "
            "bracelet and then you can go punching question blocks around."
        ),
        "audio": prompt_filepath_conversational_b
    }
}

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample is lazy so we can always call it
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def main():
    # Select the best available device, skipping MPS due to float64 limitations
    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     # Temporary check to work around float64 limitations on MPS
    #     if torch.float64 not in torch.testing.get_all_dtypes():
    #         print("Warning: float64 not supported on MPS, using CPU instead.")
    #         device = "cpu"
    #     else:
    #         # NOTE: MPS performance may be suboptimal
    #         print("Using MPS device (performance may be suboptimal).")
    #         device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # --- Configuration ---
    buffer_size_frames = 10 # Number of token frames to buffer before decoding
    max_audio_length_ms_per_utterance = 15_000 # Max duration for each generated part
    # --- End Configuration ---

    # Load model (uses the modified Generator)
    print("Loading model...")
    generator = load_csm_1b(device)
    model_sample_rate = generator.sample_rate # Get sample rate from generator
    print("Model loaded.")

    # Prepare prompts
    print("Preparing prompts...")
    prompt_a = prepare_prompt(
        SPEAKER_PROMPTS["conversational_a"]["text"],
        0,
        SPEAKER_PROMPTS["conversational_a"]["audio"],
        model_sample_rate
    )

    prompt_b = prepare_prompt(
        SPEAKER_PROMPTS["conversational_b"]["text"],
        1,
        SPEAKER_PROMPTS["conversational_b"]["audio"],
        model_sample_rate
    )
    print("Prompts prepared.")

    # Define conversation
    conversation = [
        {"text": "Hey how are you doing?", "speaker_id": 0},
        {"text": "Pretty good, pretty good. How about you?", "speaker_id": 1},
        {"text": "I'm great! So happy to be speaking with you today.", "speaker_id": 0},
        {"text": "Me too! This is some cool stuff, isn't it?", "speaker_id": 1}
    ]

    generated_segments_for_context = [] # Still need to store full audio for context
    prompt_segments = [prompt_a, prompt_b]

    total_conversation_audio_tokens = [] # Store tokens for final full audio if needed

    for i, utterance in enumerate(conversation):
        print(f"\n--- Generating Utterance {i+1} (Buffered Streaming) ---")
        print(f"Speaker {utterance['speaker_id']}: {utterance['text']}")
        print(f"Using buffer size: {buffer_size_frames} frames")

        start_time_utterance = time.perf_counter()
        first_chunk_received = False
        time_to_first_chunk = 0.0
        chunk_intervals = []
        last_chunk_time = 0.0
        generated_audio_chunks = [] # Collect decoded audio chunks

        # Use the new buffered generator method
        stream_generator = generator.generate_buffered(
            text=utterance['text'],
            speaker=utterance['speaker_id'],
            context=prompt_segments + generated_segments_for_context,
            buffer_size=buffer_size_frames, # Pass the buffer size
            max_audio_length_ms=max_audio_length_ms_per_utterance,
            temperature=0.9,
            topk=50,
        )

        print("Starting buffered generation stream...")
        # Measure time potentially spent setting up the generator before the first yield
        start_time_yield_wait = time.perf_counter()

        # Iterate through the generator yielding decoded audio chunks
        for chunk_idx, audio_chunk in enumerate(stream_generator):
            # audio_chunk has shape (1, num_samples_in_chunk)
            current_chunk_time = time.perf_counter()

            if not first_chunk_received:
                time_to_first_chunk = current_chunk_time - start_time_yield_wait
                first_chunk_received = True
                print(f"Time to first chunk (TTFC): {time_to_first_chunk:.4f} seconds")
                last_chunk_time = current_chunk_time
            else:
                interval = current_chunk_time - last_chunk_time
                chunk_intervals.append(interval)
                last_chunk_time = current_chunk_time
                # print(f"Received audio chunk {chunk_idx+1}, interval: {interval:.4f}s, shape: {audio_chunk.shape}") # Verbose

            # Collect chunks (move to CPU for concatenation/saving)
            generated_audio_chunks.append(audio_chunk.cpu())

        end_time_utterance = time.perf_counter()
        print("Finished buffered generation stream.")

        if not generated_audio_chunks:
            print("WARNING: No audio chunks were generated for this utterance.")
            continue # Skip metrics calculation and context update

        # --- Calculate Metrics ---
        total_generation_time = end_time_utterance - start_time_utterance
        print(f"\nMetrics for Utterance {i+1}:")
        print(f"  Total wall time: {total_generation_time:.4f} seconds")

        if len(chunk_intervals) > 0:
            avg_chunk_interval = sum(chunk_intervals) / len(chunk_intervals)
            print(f"  Average time between chunks: {avg_chunk_interval:.4f} seconds")
        elif first_chunk_received:
             print("  Only one chunk was received.")

        # Concatenate audio to measure duration and calculate RTF
        # Chunks have shape (1, num_samples), concatenate along dim=1
        full_utterance_audio = torch.cat(generated_audio_chunks, dim=1)
        total_samples = full_utterance_audio.shape[1]
        total_audio_duration_sec = total_samples / model_sample_rate
        print(f"  Generated audio duration: {total_audio_duration_sec:.4f} seconds ({total_samples} samples)")

        if total_audio_duration_sec > 0:
            # Calculate RTF using the time from first chunk received to end
            processing_time = end_time_utterance - start_time_yield_wait
            rtf = processing_time / total_audio_duration_sec
            print(f"  Real-Time Factor (RTF): {rtf:.4f} (Processing Time / Audio Duration)")
        else:
            print("  RTF calculation skipped (zero audio duration)")

        # --- Update Context & Save ---
        segment_audio = full_utterance_audio.squeeze(0) # Shape (num_samples,) for Segment and saving
        generated_segments_for_context.append(
            Segment(text=utterance['text'], speaker=utterance['speaker_id'], audio=segment_audio)
        )
        total_conversation_audio_tokens.extend(generated_audio_chunks) # Keep tokens if needed

        output_filename = f"utterance_{i+1}_buffered_stream.wav"
        torchaudio.save(output_filename, full_utterance_audio, model_sample_rate)
        print(f"  Saved utterance to {output_filename}")

    print("\n--- Finished Conversation ---")

    # Optional: Save the full conversation concatenated from segments
    if generated_segments_for_context:
        print("\nSaving full conversation...")
        all_audio_segments = torch.cat([seg.audio for seg in generated_segments_for_context], dim=0)
        torchaudio.save(
            "full_conversation_buffered_stream.wav",
            all_audio_segments.unsqueeze(0), # Add channel dim
            model_sample_rate
        )
        print("Successfully generated full_conversation_buffered_stream.wav")
    else:
        print("No utterances were successfully generated, skipping final save.")


if __name__ == "__main__":
    main() 