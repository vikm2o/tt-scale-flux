try:
    from .gemini_verifier import GeminiVerifier
    from .weighted_verifier import WeightedGeminiVerifier
except ImportError as e:
    GeminiVerifier = None
    WeightedGeminiVerifier = None

try:
    from .openai_verifier import OpenAIVerifier
    from .weighted_verifier import WeightedOpenAIVerifier
except ImportError as e:
    OpenAIVerifier = None
    WeightedOpenAIVerifier = None

try:
    from .qwen_verifier import QwenVerifier
    from .weighted_verifier import WeightedQwenVerifier
except ImportError as e:
    QwenVerifier = None
    WeightedQwenVerifier = None

try:
    from .claude_verifier import ClaudeVerifier
    from .weighted_verifier import WeightedClaudeVerifier
except ImportError as e:
    ClaudeVerifier = None
    WeightedClaudeVerifier = None

# Import the factory function
try:
    from .weighted_verifier import create_weighted_verifier
except ImportError as e:
    create_weighted_verifier = None

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