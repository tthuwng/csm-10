from dataclasses import dataclass
from typing import List, Tuple, Dict, Union, Generator as PyGenerator
import time

import torch
from huggingface_hub import hf_hub_download
from models import Model
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        self.sample_rate = mimi.sample_rate
        self.device = device
        
        # cache for tokenized prompts
        self._prompt_cache = {}
        
        # context tokens cache
        self.ctx_tokens = []
        self.ctx_tokens_mask = []

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (text, speaker)
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33, device=self.device).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33, device=self.device).bool()
        text_frame[:, -1] = torch.tensor(text_tokens, device=self.device)
        text_frame_mask[:, -1] = True

        result = (text_frame, text_frame_mask)
        self._prompt_cache[cache_key] = result
        return result

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"

        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        eos_frame = torch.zeros(audio_tokens.size(0), 1, device=self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33, device=self.device).long()
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33, device=self.device).bool()
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        return audio_frame, audio_frame_mask

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def update_ctx_tokens(self, context: List[Segment]) -> None:
        """Update the cached context tokens."""
        start_time = time.time()
        self.ctx_tokens, self.ctx_tokens_mask = zip(*[self._tokenize_segment(seg) for seg in context]) if context else ([], [])
        duration = (time.time() - start_time)
        print(f"update_ctx_tokens: {duration*1000:.02f} ms")

    @torch.inference_mode()
    def _prepare_prompt_tokens(self, text: str, speaker: int, context: List[Segment]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare tokens for generation, using cached context if available."""
        tokens, tokens_mask = (self.ctx_tokens, self.ctx_tokens_mask)
        start_time = time.time()
        gen_tokens, gen_masks = self._tokenize_text_segment(text, speaker)
        duration = (time.time() - start_time)
        print(f"_prepare_prompt_tokens: text: {duration*1000:.02f} ms")
        return (
            torch.cat([*tokens, gen_tokens], dim=0).long().to(self.device),
            torch.cat([*tokens_mask, gen_masks], dim=0).bool().to(self.device),
        )

    @torch.inference_mode()
    def generate_stream(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> PyGenerator[torch.Tensor, None, None]:
        """Generate audio stream from text."""
        self._model.reset_caches()
        max_generation_len = int(max_audio_length_ms / 80)
        prompt_tokens, prompt_tokens_mask = self._prepare_prompt_tokens(text, speaker, context)

        curr_tokens, curr_tokens_mask = prompt_tokens.unsqueeze(0), prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0), device=self.device).unsqueeze(0).long()

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        with self._audio_tokenizer.streaming(1):
            zeros_1_1 = torch.zeros(1, 1, device=self.device).long()
            zeros_1_1_bool = torch.zeros(1, 1, device=self.device).bool()

            for i in range(max_generation_len):
                sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                if torch.all(sample == 0):
                    break  # EOS

                yield self._audio_tokenizer.decode(sample.unsqueeze(-1)).squeeze().unsqueeze(1)

                curr_tokens = torch.cat([sample, zeros_1_1], dim=1).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [torch.ones_like(sample, dtype=torch.bool), zeros_1_1_bool], dim=1
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        """Generate audio from text."""
        samples = self.generate_stream(
            text,
            speaker,
            context,
            max_audio_length_ms,
            temperature,
            topk
        )
        
        return torch.cat(list(samples))


def load_csm_1b(device: str = "cuda") -> Generator:
    print("Loading CSM-1B model...")
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    compile_start = time.time()
    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
        print(f"Model compiled successfully in {time.time() - compile_start:.2f} seconds.")
    except Exception as e:
        print(f"Warning: Model compilation failed: {e}. Continuing without compilation.")

    generator = Generator(model)
    return generator