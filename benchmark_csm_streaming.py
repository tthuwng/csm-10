from typing import Dict, List, Tuple, Optional
import time
import torch
import torchaudio
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import os
from generator import load_csm_1b, Segment, Generator
from huggingface_hub import hf_hub_download

# Disable torch compile if needed (match the other script for consistency if it solves issues)
# os.environ["NO_TORCH_COMPILE"] = "1"

@dataclass
class StreamingBenchmarkResult:
    utterance_index: int
    word_count: int
    approx_context_words: int
    time_to_first_chunk_ms: float
    avg_chunk_time_ms: float # Average time between subsequent chunks
    total_stream_time_s: float
    total_audio_duration_s: float
    simulated_rtf: float # Total stream time / total audio duration
    num_chunks: int


def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def run_streaming_benchmark(
    output_dir: str = "benchmark_results_streaming",
    device: str = "cuda",
    num_runs: int = 3, # Fewer runs might be ok for streaming focus
    temperature: float = 0.9,
    topk: int = 50,
    max_audio_length_ms: float = 15000,
    stream_chunk_frames: int = 6,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Using device: {device}")
    print(f"Streaming chunk size: {stream_chunk_frames} frames (~{stream_chunk_frames * 80} ms)")
    generator = load_csm_1b(device)

    # --- Prepare Prompts ---
    prompt_filepath_a = hf_hub_download(
        repo_id="sesame/csm-1b",
        filename="prompts/conversational_a.wav"
    )
    prompt_a_text = (
        "like revising for an exam I'd have to try and like keep up the momentum because I'd "
        "start really early I'd be like okay I'm gonna start revising now and then like "
        "you're revising for ages and then I just like start losing steam I didn't do that "
        "for the exam we had recently to be fair that was a more of a last minute scenario "
        "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
        "sort of start the day with this not like a panic but like a"
    )
    prompt_a = prepare_prompt(prompt_a_text, 0, prompt_filepath_a, generator.sample_rate)

    prompt_filepath_b = hf_hub_download(
        repo_id="sesame/csm-1b",
        filename="prompts/conversational_b.wav"
    )
    prompt_b_text = (
        "like a super Mario level. Like it's very like high detail. And like, once you get "
        "into the park, it just like, everything looks like a computer game and they have all "
        "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
        "will have like a question block. And if you like, you know, punch it, a coin will "
        "come out. So like everyone, when they come into the park, they get like this little "
        "bracelet and then you can go punching question blocks around."
    )
    prompt_b = prepare_prompt(prompt_b_text, 1, prompt_filepath_b, generator.sample_rate)
    prompt_segments = [prompt_a, prompt_b]

    # --- Define Conversation ---
    conversation = [
        {"text": "Hey how are you doing?", "speaker_id": 0},
        {"text": "Pretty good, pretty good. How about you?", "speaker_id": 1},
        {"text": "I'm great! So happy to be speaking with you today.", "speaker_id": 0},
        {"text": "Me too! This is some cool stuff, isn't it? Let's talk a bit longer to see how context affects performance.", "speaker_id": 1},
        {"text": "Absolutely. The ability to maintain context in a conversation like this is crucial for natural interaction.", "speaker_id": 0}
    ]
    
    all_results = [] # Store results for each utterance's benchmark
    generated_segments = [] # Store generated segments for context accumulation

    print("Performing warmup run (streaming)...")
    try:
        # Consume the generator fully for warmup
        _ = list(generator.generate(
            text="This is a short sentence for warmup.",
            speaker=0,
            context=[prompt_segments[0]], # Use only one prompt for warmup
            max_audio_length_ms=2000,
            temperature=temperature,
            topk=topk,
            stream=True,
            stream_chunk_frames=stream_chunk_frames
        ))
        print("Warmup complete.")
    except Exception as e:
        print(f"Warmup failed: {e}. Proceeding with benchmark runs, but results might be less stable.")

    # --- Benchmark Streaming Conversation ---
    for utterance_idx, utterance_info in enumerate(conversation):
        utterance_text = utterance_info["text"]
        speaker_id = utterance_info["speaker_id"]
        print(f"\nBenchmarking Streaming Utterance {utterance_idx+1}/{len(conversation)} (Speaker {speaker_id}): \"{utterance_text[:60]}...\"")

        current_context = prompt_segments + generated_segments
        approx_context_words = sum(len(seg.text.split()) for seg in current_context)

        utterance_run_results: List[Optional[StreamingBenchmarkResult]] = []
        first_run_full_audio = None
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}")
            run_chunk_times = []
            run_audio_chunks = []
            
            stream_start_time = time.time()
            first_chunk_time = -1.0
            last_chunk_yield_time = stream_start_time
            
            try:
                stream_generator = generator.generate(
                    text=utterance_text,
                    speaker=speaker_id,
                    context=current_context,
                    max_audio_length_ms=max_audio_length_ms,
                    temperature=temperature,
                    topk=topk,
                    stream=True,
                    stream_chunk_frames=stream_chunk_frames
                )
                
                chunk_index = 0
                for chunk_audio in stream_generator:
                    current_time = time.time()
                    if chunk_index == 0:
                        first_chunk_time = current_time - stream_start_time
                    else:
                        run_chunk_times.append(current_time - last_chunk_yield_time)
                    
                    run_audio_chunks.append(chunk_audio)
                    last_chunk_yield_time = current_time
                    chunk_index += 1
                
                stream_end_time = time.time() # Time when generator finishes

                if not run_audio_chunks:
                    print("  Run failed: No audio chunks generated.")
                    utterance_run_results.append(None)
                    continue

                # --- Calculate metrics for this run ---
                total_stream_time_s = stream_end_time - stream_start_time
                num_chunks = len(run_audio_chunks)
                full_audio = torch.cat(run_audio_chunks)
                total_audio_duration_s = len(full_audio) / generator.sample_rate
                
                time_to_first_chunk_ms = first_chunk_time * 1000 if first_chunk_time >= 0 else -1.0
                avg_chunk_time_ms = (np.mean(run_chunk_times) * 1000) if run_chunk_times else 0.0
                simulated_rtf = total_stream_time_s / total_audio_duration_s if total_audio_duration_s > 0 else float('inf')
                
                result = StreamingBenchmarkResult(
                    utterance_index=utterance_idx,
                    word_count=len(utterance_text.split()),
                    approx_context_words=approx_context_words,
                    time_to_first_chunk_ms=time_to_first_chunk_ms,
                    avg_chunk_time_ms=avg_chunk_time_ms,
                    total_stream_time_s=total_stream_time_s,
                    total_audio_duration_s=total_audio_duration_s,
                    simulated_rtf=simulated_rtf,
                    num_chunks=num_chunks,
                )
                utterance_run_results.append(result)
                
                # Save full audio from first run for context
                if run == 0:
                    first_run_full_audio = full_audio.detach().clone()
                    # Optionally save the full reconstructed audio
                    filename_prefix = "".join(c if c.isalnum() else "_" for c in utterance_text[:20])
                    save_path = output_path / f"sample_utt{utterance_idx+1}_{filename_prefix}_{len(utterance_text.split())}_words_reconstructed.wav"
                    print(f"    Saving reconstructed sample to {save_path}")
                    torchaudio.save(
                        save_path,
                        full_audio.unsqueeze(0).cpu(),
                        generator.sample_rate
                    )

            except Exception as e:
                print(f"  Run {run+1} failed unexpectedly: {e}")
                import traceback
                traceback.print_exc()
                utterance_run_results.append(None)

        # --- Averaging for the current utterance ---
        valid_results = [r for r in utterance_run_results if r is not None and r.total_audio_duration_s > 0]
        num_valid_runs = len(valid_results)

        if num_valid_runs == 0:
            print(f"  No valid runs completed for this utterance. Skipping average calculation and context update.")
            all_results.append({
                "utterance_index": utterance_idx,
                "utterance": utterance_text,
                "speaker_id": speaker_id,
                "word_count": len(utterance_text.split()),
                "approx_context_words": approx_context_words,
                "error": "All runs failed or produced no audio.",
                "num_valid_runs": 0,
            })
            continue

        # --- Calculate Average Results ---
        avg_result_dict = {
            "utterance_index": utterance_idx,
            "utterance": utterance_text,
            "speaker_id": speaker_id,
            "word_count": len(utterance_text.split()),
            "approx_context_words": approx_context_words,
            "avg_ttfc_ms": np.mean([r.time_to_first_chunk_ms for r in valid_results if r.time_to_first_chunk_ms >= 0]),
            "avg_chunk_time_ms": np.mean([r.avg_chunk_time_ms for r in valid_results]),
            "avg_total_stream_time_s": np.mean([r.total_stream_time_s for r in valid_results]),
            "avg_total_audio_duration_s": np.mean([r.total_audio_duration_s for r in valid_results]),
            "avg_simulated_rtf": np.mean([r.simulated_rtf for r in valid_results]),
            "avg_num_chunks": np.mean([r.num_chunks for r in valid_results]),
            "num_valid_runs": num_valid_runs,
        }
        all_results.append(avg_result_dict)

        print(f"  Avg TTFC: {avg_result_dict['avg_ttfc_ms']:.2f} ms")
        print(f"  Avg Sim RTF: {avg_result_dict['avg_simulated_rtf']:.3f}")
        print(f"  (Based on {num_valid_runs}/{num_runs} valid runs)")

        # --- Accumulate context for the *next* utterance (if first run succeeded) ---
        if first_run_full_audio is not None:
            generated_segments.append(Segment(text=utterance_text, speaker=speaker_id, audio=first_run_full_audio))
        else:
            print(f"  Warning: First run failed or produced no audio for utterance {utterance_idx+1}, context will not include this segment.")

    print("\n--- Streaming Benchmark Summary ---")
    for result in all_results:
        print(f"\nUtterance {result['utterance_index']+1}/{len(conversation)} (Speaker {result['speaker_id']}, {result['word_count']} words, Context: ~{result['approx_context_words']} words):")
        print(f"  Text: \"{result['utterance'][:80]}...\"")
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Avg Time to First Chunk (TTFC): {result['avg_ttfc_ms']:.2f} ms")
            print(f"  Avg Time Between Chunks:      {result['avg_chunk_time_ms']:.2f} ms")
            print(f"  Avg Total Stream Time:          {result['avg_total_stream_time_s']:.3f} s")
            print(f"  Avg Simulated RTF:            {result['avg_simulated_rtf']:.3f}")
            print(f"  Avg Audio Duration:             {result['avg_total_audio_duration_s']:.3f} s")
            print(f"  Avg Number of Chunks:         {result['avg_num_chunks']:.1f}")
            print(f"  Valid Runs: {result['num_valid_runs']}/{num_runs}")

    print("--- End Streaming Benchmark Summary ---")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark CSM model streaming performance")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run benchmark on (cuda, cpu)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results_streaming",
                        help="Directory to save benchmark results and samples")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of runs for each test case (after warmup)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Temperature for generation")
    parser.add_argument("--topk", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--max-length", type=float, default=15000,
                        help="Maximum audio length in milliseconds for generation")
    parser.add_argument("--chunk-frames", type=int, default=6,
                        help="Number of frames per streaming chunk (e.g., 6 frames = 0.48s)")
    
    args = parser.parse_args()
    
    if "cuda" in args.device and not torch.cuda.is_available():
        print(f"Warning: Requested device '{args.device}' but CUDA is not available. Falling back to CPU.")
        args.device = "cpu"

    run_streaming_benchmark(
        output_dir=args.output_dir,
        device=args.device,
        num_runs=args.num_runs,
        temperature=args.temperature,
        topk=args.topk,
        max_audio_length_ms=args.max_length,
        stream_chunk_frames=args.chunk_frames,
    ) 