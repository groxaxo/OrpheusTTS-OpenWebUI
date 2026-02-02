"""
Orpheus TTS Inference Module for LMStudio GGUF Model Integration.

This module provides text-to-speech generation using OpenAI-compatible API endpoints
from LMStudio or other compatible inference servers. Based on patterns from
Lex-au/Orpheus-FastAPI.

Features:
- OpenAI-compatible API integration
- Support for any Orpheus GGUF model loaded in LMStudio
- Text sanitization integration
- Performance monitoring
- Batched generation for long text
- Crossfade audio stitching
"""

import os
import sys
import requests
import json
import time
import wave
import numpy as np
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Generator, Union, Tuple
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Helper to detect if running in Uvicorn's reloader subprocess
# This uses an environment variable approach which is more reliable than
# checking sys.argv patterns that may vary between uvicorn versions
def is_reloader_process():
    """
    Check if the current process is a uvicorn reloader subprocess.
    
    Uses an environment variable flag that gets set on first run.
    If UVICORN_STARTED is already set when this process starts,
    then this is a reload/subprocess, not the original process.
    """
    return os.environ.get('UVICORN_STARTED') == 'true'

IS_RELOADER = is_reloader_process()
if not IS_RELOADER:
    os.environ['UVICORN_STARTED'] = 'true'

# Try to detect hardware capabilities
import torch

HIGH_END_GPU = False
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    gpu_mem_gb = props.total_memory / (1024**3)
    HIGH_END_GPU = (gpu_mem_gb >= 16.0 or 
                    props.major >= 8 or 
                    (gpu_mem_gb >= 12.0 and props.major >= 7))
    if not IS_RELOADER:
        logger.info(f"Hardware: CUDA GPU detected - {props.name} ({gpu_mem_gb:.2f} GB)")
else:
    if not IS_RELOADER:
        logger.info("Hardware: CPU only (No CUDA GPU detected)")

# Configuration from environment variables
API_URL = os.environ.get("ORPHEUS_API_URL", "http://127.0.0.1:1234/v1/completions")
REQUEST_TIMEOUT = int(os.environ.get("ORPHEUS_API_TIMEOUT", "120"))
MAX_TOKENS = int(os.environ.get("ORPHEUS_MAX_TOKENS", "8192"))
TEMPERATURE = float(os.environ.get("ORPHEUS_TEMPERATURE", "0.6"))
TOP_P = float(os.environ.get("ORPHEUS_TOP_P", "0.9"))
REPETITION_PENALTY = 1.1  # Hardcoded for stability
SAMPLE_RATE = int(os.environ.get("ORPHEUS_SAMPLE_RATE", "24000"))
MODEL_NAME = os.environ.get("ORPHEUS_MODEL_NAME", "")  # Empty means use whatever is loaded

HEADERS = {"Content-Type": "application/json"}

# Voice definitions by language
ENGLISH_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
FRENCH_VOICES = ["pierre", "amelie", "marie"]
GERMAN_VOICES = ["jana", "thomas", "max"]
KOREAN_VOICES = ["유나", "준서"]
HINDI_VOICES = ["ऋतिका"]
MANDARIN_VOICES = ["长乐", "白芷"]
SPANISH_VOICES = ["javi", "sergio", "maria"]
ITALIAN_VOICES = ["pietro", "giulia", "carlo"]

AVAILABLE_VOICES = (
    ENGLISH_VOICES + 
    FRENCH_VOICES + 
    GERMAN_VOICES + 
    KOREAN_VOICES + 
    HINDI_VOICES + 
    MANDARIN_VOICES + 
    SPANISH_VOICES + 
    ITALIAN_VOICES
)
DEFAULT_VOICE = "tara"

# Map voices to languages
VOICE_TO_LANGUAGE = {}
VOICE_TO_LANGUAGE.update({voice: "english" for voice in ENGLISH_VOICES})
VOICE_TO_LANGUAGE.update({voice: "french" for voice in FRENCH_VOICES})
VOICE_TO_LANGUAGE.update({voice: "german" for voice in GERMAN_VOICES})
VOICE_TO_LANGUAGE.update({voice: "korean" for voice in KOREAN_VOICES})
VOICE_TO_LANGUAGE.update({voice: "hindi" for voice in HINDI_VOICES})
VOICE_TO_LANGUAGE.update({voice: "mandarin" for voice in MANDARIN_VOICES})
VOICE_TO_LANGUAGE.update({voice: "spanish" for voice in SPANISH_VOICES})
VOICE_TO_LANGUAGE.update({voice: "italian" for voice in ITALIAN_VOICES})

AVAILABLE_LANGUAGES = ["english", "french", "german", "korean", "hindi", "mandarin", "spanish", "italian"]

# Token processing
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# Cache for token IDs
token_id_cache = {}
MAX_CACHE_SIZE = 10000


class PerformanceMonitor:
    """Track and report performance metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.token_count = 0
        self.audio_chunks = 0
        self.last_report_time = time.time()
        self.report_interval = 2.0
        
    def add_tokens(self, count: int = 1) -> None:
        self.token_count += count
        self._check_report()
        
    def add_audio_chunk(self) -> None:
        self.audio_chunks += 1
        self._check_report()
        
    def _check_report(self) -> None:
        current_time = time.time()
        if current_time - self.last_report_time >= self.report_interval:
            self.report()
            self.last_report_time = current_time
            
    def report(self) -> None:
        elapsed = time.time() - self.start_time
        if elapsed < 0.001:
            return
        tokens_per_sec = self.token_count / elapsed
        est_duration = self.audio_chunks * 0.085
        logger.info(f"Progress: {tokens_per_sec:.1f} tokens/sec, est. {est_duration:.1f}s audio")


# Global performance monitor
perf_monitor = PerformanceMonitor()


def format_prompt(prompt: str, voice: str = DEFAULT_VOICE) -> str:
    """Format prompt for Orpheus model with voice prefix and special tokens."""
    if voice not in AVAILABLE_VOICES:
        logger.warning(f"Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE
    
    formatted_prompt = f"{voice}: {prompt}"
    special_start = "<|audio|>"
    special_end = "<|eot_id|>"
    
    return f"{special_start}{formatted_prompt}{special_end}"


def turn_token_into_id(token_string: str, index: int) -> Optional[int]:
    """
    Convert token string to ID with caching.
    
    Args:
        token_string: The token string to convert
        index: Position index used for token offset calculation
        
    Returns:
        Token ID if valid, None otherwise
    """
    cache_key = (token_string, index % 7)
    if cache_key in token_id_cache:
        return token_id_cache[cache_key]
    
    if CUSTOM_TOKEN_PREFIX not in token_string:
        return None
    
    token_string = token_string.strip()
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    
    if last_token_start == -1:
        return None
    
    last_token = token_string[last_token_start:]
    
    if not (last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">")):
        return None
    
    try:
        number_str = last_token[14:-1]
        token_id = int(number_str) - 10 - ((index % 7) * 4096)
        
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = token_id
        
        return token_id
    except (ValueError, IndexError):
        return None


def generate_tokens_from_api(
    prompt: str, 
    voice: str = DEFAULT_VOICE,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY
) -> Generator[str, None, None]:
    """Generate tokens from text using OpenAI-compatible API with streaming."""
    start_time = time.time()
    formatted_prompt = format_prompt(prompt, voice)
    logger.info(f"Generating speech for: {prompt[:50]}...")
    
    payload = {
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
        "stream": True
    }
    
    if MODEL_NAME:
        payload["model"] = MODEL_NAME
    
    session = requests.Session()
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            response = session.post(
                API_URL,
                headers=HEADERS,
                json=payload,
                stream=True,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}")
                if response.status_code >= 500:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return
            
            token_counter = 0
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        
                        if data_str.strip() == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                token_chunk = data['choices'][0].get('text', '')
                                for token_text in token_chunk.split('>'):
                                    token_text = f'{token_text}>'
                                    token_counter += 1
                                    perf_monitor.add_tokens()
                                    if token_text:
                                        yield token_text
                        except json.JSONDecodeError as e:
                            logger.debug(f"Error decoding JSON: {e}")
                            continue
            
            generation_time = time.time() - start_time
            tokens_per_second = token_counter / generation_time if generation_time > 0 else 0
            logger.info(f"Token generation complete: {token_counter} tokens in {generation_time:.2f}s")
            return
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {REQUEST_TIMEOUT} seconds")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached. Token generation failed.")
                return
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error to API at {API_URL}")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached. Token generation failed.")
                return


# SNAC model singleton for efficient audio decoding
_snac_model = None
_snac_device = None


def _get_snac_model():
    """Get or initialize the SNAC model (singleton pattern)."""
    global _snac_model, _snac_device
    
    if _snac_model is None:
        try:
            from snac import SNAC
            _snac_device = "cuda" if torch.cuda.is_available() else "cpu"
            _snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(_snac_device)
            if not IS_RELOADER:
                logger.info(f"SNAC model loaded on {_snac_device}")
        except ImportError:
            logger.error("SNAC library not installed. Run: pip install snac")
            return None, None
        except Exception as e:
            logger.error(f"Error loading SNAC model: {e}")
            return None, None
    
    return _snac_model, _snac_device


def convert_to_audio(multiframe: List[int], count: int) -> Optional[bytes]:
    """
    Convert token frames to audio.
    
    This requires the SNAC model for audio decoding.
    Uses a singleton pattern for efficient model reuse.
    """
    model, snac_device = _get_snac_model()
    
    if model is None:
        return None
    
    try:
        if len(multiframe) < 7:
            return None
        
        num_frames = len(multiframe) // 7
        frame = multiframe[:num_frames * 7]
        
        # Pre-allocate tensors
        codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=snac_device)
        codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=snac_device)
        codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=snac_device)
        
        frame_tensor = torch.tensor(frame, dtype=torch.int32, device=snac_device)
        
        for j in range(num_frames):
            idx = j * 7
            codes_0[j] = frame_tensor[idx]
            codes_1[j * 2] = frame_tensor[idx + 1]
            codes_1[j * 2 + 1] = frame_tensor[idx + 4]
            codes_2[j * 4] = frame_tensor[idx + 2]
            codes_2[j * 4 + 1] = frame_tensor[idx + 3]
            codes_2[j * 4 + 2] = frame_tensor[idx + 5]
            codes_2[j * 4 + 3] = frame_tensor[idx + 6]
        
        codes = [
            codes_0.unsqueeze(0),
            codes_1.unsqueeze(0),
            codes_2.unsqueeze(0)
        ]
        
        # Validate token ranges
        if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or
            torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or
            torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
            return None
        
        with torch.inference_mode():
            audio_hat = model.decode(codes)
            audio_slice = audio_hat[:, :, 2048:4096]
            
            if snac_device == "cuda":
                audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
                audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
            else:
                detached_audio = audio_slice.detach().cpu()
                audio_np = detached_audio.numpy()
                audio_int16 = (audio_np * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
        
        perf_monitor.add_audio_chunk()
        return audio_bytes
        
    except Exception as e:
        logger.error(f"Error converting to audio: {e}")
        return None


async def tokens_decoder(token_gen) -> Generator[bytes, None, None]:
    """Async token decoder with early first-chunk processing."""
    buffer = []
    count = 0
    first_chunk_processed = False
    min_frames_first = 7
    min_frames_subsequent = 28
    process_every = 7
    
    async for token_text in token_gen:
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            
            if not first_chunk_processed:
                if count >= min_frames_first:
                    buffer_to_proc = buffer[-min_frames_first:]
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        first_chunk_processed = True
                        yield audio_samples
            else:
                if count % process_every == 0 and count >= min_frames_subsequent:
                    buffer_to_proc = buffer[-min_frames_subsequent:]
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples


def tokens_decoder_sync(syn_token_gen, output_file: Optional[str] = None) -> List[bytes]:
    """Synchronous wrapper for token decoder."""
    queue_size = 100 if HIGH_END_GPU else 50
    audio_queue = queue.Queue(maxsize=queue_size)
    audio_segments = []
    
    wav_file = None
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        wav_file = wave.open(output_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
    
    producer_done_event = threading.Event()
    producer_started_event = threading.Event()
    batch_size = 32 if HIGH_END_GPU else 16
    
    async def async_token_gen():
        batch = []
        for token in syn_token_gen:
            batch.append(token)
            if len(batch) >= batch_size:
                for t in batch:
                    yield t
                batch = []
        for t in batch:
            yield t
    
    async def async_producer():
        try:
            producer_started_event.set()
            async for audio_chunk in tokens_decoder(async_token_gen()):
                if audio_chunk:
                    audio_queue.put(audio_chunk)
        except Exception as e:
            logger.error(f"Error in token processing: {e}")
        finally:
            producer_done_event.set()
            audio_queue.put(None)
    
    def run_async():
        asyncio.run(async_producer())
    
    thread = threading.Thread(target=run_async, name="TokenProcessor")
    thread.daemon = True
    thread.start()
    
    producer_started_event.wait(timeout=5.0)
    
    write_buffer = bytearray()
    buffer_max_size = 1024 * 1024
    
    while True:
        try:
            audio = audio_queue.get(timeout=0.1)
            if audio is None:
                break
            
            audio_segments.append(audio)
            
            if wav_file:
                write_buffer.extend(audio)
                if len(write_buffer) >= buffer_max_size:
                    wav_file.writeframes(write_buffer)
                    write_buffer = bytearray()
        except queue.Empty:
            if producer_done_event.is_set() and audio_queue.empty():
                break
    
    if thread.is_alive():
        thread.join(timeout=10.0)
    
    if wav_file and len(write_buffer) > 0:
        wav_file.writeframes(write_buffer)
    
    if wav_file:
        wav_file.close()
    
    return audio_segments


def split_text_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    parts = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        
        if char in (' ', '\n', '\t') and len(current_sentence) > 1:
            prev_char = current_sentence[-2]
            if prev_char in ('.', '!', '?'):
                if len(current_sentence) > 3 and current_sentence[-3] not in ('.', ' '):
                    parts.append(current_sentence.strip())
                    current_sentence = ""
    
    if current_sentence.strip():
        parts.append(current_sentence.strip())
    
    # Combine short segments
    min_chars = 20
    combined_sentences = []
    i = 0
    
    while i < len(parts):
        current = parts[i]
        while i < len(parts) - 1 and len(current) < min_chars:
            i += 1
            current += " " + parts[i]
        combined_sentences.append(current)
        i += 1
    
    return combined_sentences


def stitch_wav_files(input_files: List[str], output_file: str, crossfade_ms: int = 50):
    """Stitch multiple WAV files together with crossfading."""
    if not input_files:
        return
    
    if len(input_files) == 1:
        import shutil
        shutil.copy(input_files[0], output_file)
        return
    
    crossfade_samples = int(SAMPLE_RATE * crossfade_ms / 1000)
    
    final_audio = np.array([], dtype=np.int16)
    first_params = None
    
    for i, input_file in enumerate(input_files):
        try:
            with wave.open(input_file, 'rb') as wav:
                if first_params is None:
                    first_params = wav.getparams()
                
                frames = wav.readframes(wav.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)
                
                if i == 0:
                    final_audio = audio
                else:
                    if len(final_audio) >= crossfade_samples and len(audio) >= crossfade_samples:
                        fade_out = np.linspace(1.0, 0.0, crossfade_samples)
                        fade_in = np.linspace(0.0, 1.0, crossfade_samples)
                        
                        crossfade_region = (final_audio[-crossfade_samples:] * fade_out + 
                                           audio[:crossfade_samples] * fade_in).astype(np.int16)
                        
                        final_audio = np.concatenate([final_audio[:-crossfade_samples], 
                                                    crossfade_region, 
                                                    audio[crossfade_samples:]])
                    else:
                        final_audio = np.concatenate([final_audio, audio])
        except Exception as e:
            logger.error(f"Error processing file {input_file}: {e}")
            if i == 0:
                raise
    
    with wave.open(output_file, 'wb') as output_wav:
        output_wav.setparams(first_params)
        output_wav.writeframes(final_audio.tobytes())


def generate_speech_from_api(
    prompt: str,
    voice: str = DEFAULT_VOICE,
    output_file: Optional[str] = None,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    use_batching: bool = True,
    max_batch_chars: int = 1000
) -> List[bytes]:
    """
    Generate speech from text using Orpheus model via LMStudio API.
    
    Args:
        prompt: Text to convert to speech
        voice: Voice to use (default: tara)
        output_file: Optional output WAV file path
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        use_batching: Whether to use batched generation for long text
        max_batch_chars: Maximum characters per batch
        
    Returns:
        List of audio byte segments
    """
    from .sanitizer import TextSanitizer
    
    # Sanitize input text
    sanitizer = TextSanitizer()
    prompt = sanitizer.sanitize(prompt)
    
    logger.info(f"Starting speech generation for '{prompt[:50]}...'")
    
    global perf_monitor
    perf_monitor = PerformanceMonitor()
    
    start_time = time.time()
    
    # Short text - direct generation
    if not use_batching or len(prompt) < max_batch_chars:
        result = tokens_decoder_sync(
            generate_tokens_from_api(
                prompt=prompt,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=REPETITION_PENALTY
            ),
            output_file=output_file
        )
        
        end_time = time.time()
        logger.info(f"Speech generation completed in {end_time - start_time:.2f} seconds")
        return result
    
    # Long text - batched generation
    logger.info(f"Using sentence-based batching for {len(prompt)} characters")
    
    sentences = split_text_into_sentences(prompt)
    logger.info(f"Split text into {len(sentences)} segments")
    
    batches = []
    current_batch = ""
    
    for sentence in sentences:
        if len(current_batch) + len(sentence) > max_batch_chars and current_batch:
            batches.append(current_batch)
            current_batch = sentence
        else:
            if current_batch:
                current_batch += " "
            current_batch += sentence
    
    if current_batch:
        batches.append(current_batch)
    
    logger.info(f"Created {len(batches)} batches for processing")
    
    all_audio_segments = []
    batch_temp_files = []
    
    for i, batch in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} characters)")
        
        temp_output_file = None
        if output_file:
            temp_output_file = f"outputs/temp_batch_{i}_{int(time.time())}.wav"
            batch_temp_files.append(temp_output_file)
        
        batch_segments = tokens_decoder_sync(
            generate_tokens_from_api(
                prompt=batch,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=REPETITION_PENALTY
            ),
            output_file=temp_output_file
        )
        
        all_audio_segments.extend(batch_segments)
    
    if output_file and batch_temp_files:
        stitch_wav_files(batch_temp_files, output_file)
        
        for temp_file in batch_temp_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_file}: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if all_audio_segments:
        total_bytes = sum(len(segment) for segment in all_audio_segments)
        duration = total_bytes / (2 * SAMPLE_RATE)
        logger.info(f"Generated {duration:.2f} seconds of audio in {total_time:.2f} seconds")
        logger.info(f"Realtime factor: {duration/total_time:.2f}x")
    
    return all_audio_segments


def list_available_voices():
    """List all available voices."""
    print("Available voices (in order of conversational realism):")
    for i, voice in enumerate(AVAILABLE_VOICES):
        marker = "★" if voice == DEFAULT_VOICE else " "
        language = VOICE_TO_LANGUAGE.get(voice, "unknown")
        print(f"{marker} {voice} ({language})")
    
    print("\nAvailable emotion tags:")
    print("<laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>")
