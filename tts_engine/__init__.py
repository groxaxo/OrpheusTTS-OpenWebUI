"""
Orpheus TTS Engine module for LMStudio GGUF model integration.
Provides text-to-speech generation using OpenAI-compatible API endpoints.
"""

from .inference import (
    generate_speech_from_api,
    generate_tokens_from_api,
    AVAILABLE_VOICES,
    DEFAULT_VOICE,
    VOICE_TO_LANGUAGE,
    AVAILABLE_LANGUAGES,
    SAMPLE_RATE,
)
from .sanitizer import TextSanitizer, NormalizationOptions

__all__ = [
    "generate_speech_from_api",
    "generate_tokens_from_api",
    "AVAILABLE_VOICES",
    "DEFAULT_VOICE",
    "VOICE_TO_LANGUAGE",
    "AVAILABLE_LANGUAGES",
    "SAMPLE_RATE",
    "TextSanitizer",
    "NormalizationOptions",
]
