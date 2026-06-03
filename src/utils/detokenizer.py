from spacy.lang.en import English
from spacy.tokens import Doc


# Initialize the English language model
_nlp = None


def _get_nlp():
    """Lazy initialization of spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = English()
    return _nlp


def detokenize(text: str) -> str:
    """
    Detokenize text using spaCy's detokenizer.

    This function takes tokenized text (with spaces around punctuation)
    and reconstructs it with proper spacing using spaCy's linguistic rules.

    Args:
        text: Tokenized text with spaces around punctuation

    Returns:
        Detokenized text with proper punctuation spacing
    """
    nlp = _get_nlp()

    # Split the text into tokens
    tokens = text.split()

    # Create a Doc object from the token strings
    doc = Doc(nlp.vocab, words=tokens)

    # Use spaCy's detokenizer to reconstruct the text
    detokenized = doc.text

    return detokenized
