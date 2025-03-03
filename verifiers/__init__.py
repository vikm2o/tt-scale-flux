from .gemini_verifier import GeminiVerifier
from .weighted_verifier import WeightedGeminiVerifier


from .openai_verifier import OpenAIVerifier
from .weighted_verifier import WeightedOpenAIVerifier

from .qwen_verifier import QwenVerifier
from .weighted_verifier import WeightedQwenVerifier

from .claude_verifier import ClaudeVerifier
from .weighted_verifier import WeightedClaudeVerifier


from .weighted_verifier import create_weighted_verifier


# Build dictionary of available verifiers
SUPPORTED_VERIFIERS = {
    "qwen": QwenVerifier,
    "gemini": GeminiVerifier,
    "openai": OpenAIVerifier,
    "claude": ClaudeVerifier,
    "weighted_qwen": WeightedQwenVerifier,
    "weighted_gemini": WeightedGeminiVerifier,
    "weighted_openai": WeightedOpenAIVerifier,
    "weighted_claude": WeightedClaudeVerifier,
}

# Filter out unavailable verifiers
SUPPORTED_VERIFIERS = {k: v for k, v in SUPPORTED_VERIFIERS.items() if v is not None}

# Get supported metrics only for available verifiers
SUPPORTED_METRICS = {k: getattr(v, "SUPPORTED_METRIC_CHOICES", None) for k, v in SUPPORTED_VERIFIERS.items()}