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

@dataclass
class BenchmarkResult:
    total_time: float
    text_tokenization_time: float
    generation_time: float
    audio_decoding_time: float
    audio_duration: float
    rtf: float
    frames_per_second: float
    watermarking_time: Optional[float] = None

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

def run_benchmark(
    output_dir: str = "benchmark_results",
    device: str = "cuda",
    num_runs: int = 5,
    temperature: float = 0.9,
    topk: int = 50,
    max_audio_length_ms: float = 15000,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Using device: {device}")
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

    print("Performing warmup run...")
    try:
        _ = generator.generate(
            text="This is a short sentence for warmup.",
            speaker=0,
            context=[prompt_segments[0]], # Use only one prompt for warmup
            max_audio_length_ms=2000,
            temperature=temperature,
            topk=topk,
            return_timers=False,
            disable_watermark=True,
        )
        print("Warmup complete.")
    except Exception as e:
        print(f"Warmup failed: {e}. Proceeding with benchmark runs, but results might be less stable.")

    # --- Benchmark Conversation ---
    for utterance_idx, utterance_info in enumerate(conversation):
        utterance_text = utterance_info["text"]
        speaker_id = utterance_info["speaker_id"]
        print(f"\nBenchmarking Utterance {utterance_idx+1}/{len(conversation)} (Speaker {speaker_id}): \"{utterance_text[:60]}...\"")

        # Determine context *before* the runs for this utterance
        current_context = prompt_segments + generated_segments
        # Calculate approximate context length for reporting later
        approx_context_words = sum(len(seg.text.split()) for seg in current_context)

        utterance_run_results = []
        first_run_audio = None # Store audio from the first successful run
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}")
            try:
                audio, timers = generator.generate(
                    text=utterance_text,
                    speaker=speaker_id,
                    context=current_context, # Use the same context for all runs
                    max_audio_length_ms=max_audio_length_ms,
                    temperature=temperature,
                    topk=topk,
                    return_timers=True,
                    disable_watermark=True, # Keep watermark disabled for benchmark consistency
                )

                audio_duration = len(audio) / generator.sample_rate if generator.sample_rate > 0 else 0
                rtf = timers["total"] / audio_duration if audio_duration > 0 else float('inf')
                num_frames = int(timers.get("num_frames", 0))
                fps = num_frames / timers["generation"] if timers["generation"] > 0 else 0
                
                result = BenchmarkResult(
                    total_time=timers["total"],
                    text_tokenization_time=timers["tokenization"],
                    generation_time=timers["generation"],
                    audio_decoding_time=timers["audio_decoding"],
                    audio_duration=audio_duration,
                    rtf=rtf,
                    frames_per_second=fps,
                    watermarking_time=timers.get("watermarking")
                )
                utterance_run_results.append(result)

                # Save the audio from the first successful run for context building
                if run == 0 and audio_duration > 0:
                    first_run_audio = audio.detach().clone()

                # Save sample WAV file for the first run
                if run == 0 and audio_duration > 0:
                    filename_prefix = "".join(c if c.isalnum() else "_" for c in utterance_text[:20])
                    save_path = output_path / f"sample_utt{utterance_idx+1}_{filename_prefix}_{len(utterance_text.split())}_words.wav"
                    print(f"    Saving sample to {save_path}")
                    torchaudio.save(
                        save_path,
                        audio.unsqueeze(0).cpu(),
                        generator.sample_rate
                    )
            except ValueError as e:
                print(f"  Run {run+1} failed: {e}")
                utterance_run_results.append(None)
            except Exception as e:
                print(f"  Run {run+1} failed unexpectedly: {e}")
                import traceback
                traceback.print_exc()
                utterance_run_results.append(None)

        # --- Averaging for the current utterance ---
        valid_results = [r for r in utterance_run_results if r is not None and r.audio_duration > 0]
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
            continue # Skip context update if all runs failed

        # --- Calculate Average Results --- 
        avg_result = {
            "utterance_index": utterance_idx,
            "utterance": utterance_text,
            "speaker_id": speaker_id,
            "word_count": len(utterance_text.split()),
            "approx_context_words": approx_context_words, # Store context length used
            "avg_total_time": np.mean([r.total_time for r in valid_results]),
            "avg_tokenization_time": np.mean([r.text_tokenization_time for r in valid_results]),
            "avg_generation_time": np.mean([r.generation_time for r in valid_results]),
            "avg_audio_decoding_time": np.mean([r.audio_decoding_time for r in valid_results]),
            "avg_audio_duration": np.mean([r.audio_duration for r in valid_results]),
            "avg_rtf": np.mean([r.rtf for r in valid_results]),
            "avg_fps": np.mean([r.frames_per_second for r in valid_results]),
            "num_valid_runs": num_valid_runs,
        }
        all_results.append(avg_result)

        print(f"  Average RTF: {avg_result['avg_rtf']:.3f}")
        print(f"  Average FPS: {avg_result['avg_fps']:.2f}")
        print(f"  (Based on {num_valid_runs}/{num_runs} valid runs)")

        # --- Accumulate context for the *next* utterance (if first run succeeded) ---
        if first_run_audio is not None:
            generated_segments.append(Segment(text=utterance_text, speaker=speaker_id, audio=first_run_audio))
        else:
            # This case should ideally not happen if num_valid_runs > 0, but added for safety
            print(f"  Warning: First run failed or produced no audio for utterance {utterance_idx+1}, context will not include this segment.")

    print("\n--- Benchmark Summary ---")
    for result in all_results:
        print(f"\nUtterance {result['utterance_index']+1}/{len(conversation)} (Speaker {result['speaker_id']}, {result['word_count']} words, Context: ~{result['approx_context_words']} words):")
        print(f"  Text: \"{result['utterance'][:80]}...\"")
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Avg RTF (Total Time / Audio Duration): {result['avg_rtf']:.3f}")
            print(f"  Avg FPS (Frames / Generation Time):  {result['avg_fps']:.2f}")
            print(f"  Avg Total Time: {result['avg_total_time']:.3f} s")
            print(f"    Avg Tokenization:  {result['avg_tokenization_time']*1000:.2f} ms")
            print(f"    Avg Generation:    {result['avg_generation_time']:.3f} s")
            print(f"    Avg Audio Decoding:{result['avg_audio_decoding_time']*1000:.2f} ms")
            print(f"  Avg Audio Duration: {result['avg_audio_duration']:.3f} s")
            print(f"  Valid Runs: {result['num_valid_runs']}/{num_runs}")

    print("--- End Benchmark Summary ---")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark CSM model RTF performance using refactored Generator")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run benchmark on (cuda, cpu)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results_refactored",
                        help="Directory to save benchmark results and samples")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="Number of runs for each test case (after warmup)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Temperature for generation")
    parser.add_argument("--topk", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--max-length", type=float, default=15000,
                        help="Maximum audio length in milliseconds for generation")
    
    args = parser.parse_args()
    
    if "cuda" in args.device and not torch.cuda.is_available():
        print(f"Warning: Requested device '{args.device}' but CUDA is not available. Falling back to CPU.")
        args.device = "cpu"

    run_benchmark(
        output_dir=args.output_dir,
        device=args.device,
        num_runs=args.num_runs,
        temperature=args.temperature,
        topk=args.topk,
        max_audio_length_ms=args.max_length,
    ) 