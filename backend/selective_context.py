"""Context-gated prediction choices using training-only transition indexes."""


def has_second_order(index, history):
    return len(history) >= 2 and bool(index.second_order.get(f"{history[-2]}:{history[-1]}"))


def context_action(index, history, family, config):
    """Keep exact Markov-2; require development evidence for neural fallback."""
    if has_second_order(index, history):
        return "markov2", "exact_second_order"
    validated = family in config.get("neural_fallback_families", [])
    if family == "recursive_traversal" and not config.get("allow_recursive_neural", False):
        validated = False
    if validated:
        return "neural", "validated_neural_fallback"
    if history and index.successors.get(history[-1]):
        return "markov1", "unvalidated_family_markov_fallback"
    return "skip", "no_supported_fallback"
