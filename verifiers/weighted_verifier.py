import os
import sys
from typing import Dict, Optional, Any, List, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.insert(0, current_dir)
sys.path.insert(0, root_dir)

from verifiers.gemini_verifier import GeminiVerifier
from verifiers.claude_verifier import ClaudeVerifier
from verifiers.openai_verifier import OpenAIVerifier
from verifiers.qwen_verifier import QwenVerifier


class WeightedVerifierMixin:
    """Mixin class that adds weighting functionality to any verifier."""
    
    def _apply_weights(self, weights=None):
        """Apply weights to the system prompt."""
        # Only proceed if weights are provided
        if not weights:
            return
            
        # Get original prompt
        original_prompt = getattr(self, "system_instruction", None) or getattr(self, "system_prompt", None) or getattr(self, "verifier_prompt", "")
        
        # Format weights for inclusion in the prompt
        weight_instructions = "\n\nWhen calculating the overall_score, use these specific weights:\n"
        
        # Normalize weights if needed
        total_weight = sum(weights.values())
        if total_weight == 0:
            return original_prompt
            
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Add each metric weight to the instructions
        metric_choices = getattr(self, "SUPPORTED_METRIC_CHOICES", [])
        for metric in metric_choices:
            if metric != "overall_score":  # Skip overall_score itself
                weight = normalized_weights.get(metric, 0.0)
                weight_percent = round(weight * 100)
                weight_instructions += f"- {metric}: {weight_percent}%\n"
        
        # Create modified prompt with weights
        modified_prompt = original_prompt + weight_instructions
        return modified_prompt


class WeightedGeminiVerifier(WeightedVerifierMixin, GeminiVerifier):
    """A weighted version of the Gemini verifier."""
    
    def __init__(self, seed=1994, model_name="gemini-2.0-flash", **kwargs):
        # Initialize the base verifier
        super().__init__(seed=seed, model_name=model_name, **kwargs)
        
        # Get weights from kwargs
        weights = kwargs.pop("weights", None)
        print(f"Using weights: {weights}")
        # Apply weights to system instruction
        if weights:
            modified_prompt = self._apply_weights(weights)
            self.system_instruction = modified_prompt
            # Update the generation config with modified system instruction
            self.generation_config.system_instruction = self.system_instruction


class WeightedClaudeVerifier(WeightedVerifierMixin, ClaudeVerifier):
    """A weighted version of the Claude verifier."""
    
    def __init__(self, seed=1994, model_name="claude-3-7-sonnet-20240229", **kwargs):
        # Initialize the base verifier
        super().__init__(seed=seed, model_name=model_name, **kwargs)
        # Get weights from kwargs
        weights = kwargs.pop("weights", None)
        print(f"Using weights: {weights}")
        
        # Apply weights to system prompt
        if weights:
            modified_prompt = self._apply_weights(weights)
            self.system_prompt = modified_prompt


class WeightedOpenAIVerifier(WeightedVerifierMixin, OpenAIVerifier):
    """A weighted version of the OpenAI verifier."""
    
    def __init__(self, seed=1994, model_name="gpt-4-vision-preview", **kwargs):
        # Initialize the base verifier
        super().__init__(seed=seed, model_name=model_name, **kwargs)

        # Get weights from kwargs
        weights = kwargs.pop("weights", None)
        print(f"Using weights: {weights}")
        
        # Apply weights to system prompt
        if weights:
            modified_prompt = self._apply_weights(weights)
            self.system_message = modified_prompt


class WeightedQwenVerifier(WeightedVerifierMixin, QwenVerifier):
    """A weighted version of the Qwen verifier."""
    
    def __init__(self, seed=1994, model_name="qwen-vl-max", **kwargs):
        # Initialize the base verifier
        super().__init__(seed=seed, model_name=model_name, **kwargs)
        # Get weights from kwargs
        weights = kwargs.pop("weights", None)
        print(f"Using weights: {weights}")
        # Apply weights to prompt
        if weights:
            modified_prompt = self._apply_weights(weights)
            self.verifier_prompt = modified_prompt


# Factory function to create weighted verifiers
def create_weighted_verifier(verifier_type="gemini", weights=None, **kwargs):
    """
    Create a weighted verifier of the specified type.
    
    Args:
        verifier_type: The type of verifier (gemini, claude, openai, qwen)
        weights: Dictionary mapping metric names to weight values (0.0-1.0)
        **kwargs: Additional arguments for the verifier
    
    Returns:
        A weighted verifier instance
    """
    verifier_map = {
        "gemini": WeightedGeminiVerifier,
        "claude": WeightedClaudeVerifier,
        "openai": WeightedOpenAIVerifier,
        "qwen": WeightedQwenVerifier
    }
    
    verifier_class = verifier_map.get(verifier_type.lower())
    if not verifier_class:
        raise ValueError(f"Unsupported verifier type: {verifier_type}")
    
    return verifier_class(weights=weights, **kwargs)


# Example usage
if __name__ == "__main__":
    # Example with custom weights
    weights = {
        "accuracy_to_prompt": 0.4,
        "creativity_and_originality": 0.3,
        "visual_quality_and_realism": 0.2,
        "consistency_and_cohesion": 0.1,
    }
    
    # Create a weighted verifier using the factory function
    verifier = create_weighted_verifier(
        verifier_type="gemini",
        weights=weights,
        model_name="gemini-2.0-flash"
    )
    
    # Use it like any other verifier
    # ...