from transformers import pipeline
import os
import logging

# Full environment configuration to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
logging.getLogger("transformers").setLevel(logging.ERROR)

def load_corrector():
    """Load optimized text correction model"""
    return pipeline(
        "text2text-generation",
        model="pszemraj/flan-t5-large-grammar-synthesis",
        device=-1,
        truncation=True,
        max_length=512,
        use_fast=True
    )

def correct_ocr_text(text: str) -> str:
    """Enhanced OCR correction with literary text specialization"""
    Print("NLP running ......")
    corrector = load_corrector()
    
    # Specialized prompt for literary OCR errors
    prompt = (
        "Correct OCR errors in this literary text excerpt while preserving:\n"
        "1. Dialogue punctuation and formatting\n"
        "2. Narrative structure\n"
        "3. Original paragraph breaks\n\n"
        "Common errors to fix:\n"
        "- Character names and proper nouns\n"
        "- Missing/damaged characters (e.g., 'bok' → 'back')\n"
        "- Letter transpositions (e.g., 'wodiing' → 'woodsing')\n"
        "- Split/merged words\n"
        "- Missing punctuation\n"
        "Return only the corrected text without any additional commentary, headers, or explanations.\n\n"
        f"INPUT TEXT:\n{text}\n\n"
        "CORRECTED TEXT:"
    )
    
    result = corrector(
        prompt,
        num_beams=7,
        early_stopping=True,
        repetition_penalty=2.5,
        length_penalty=1.5,
        max_length=512,
        no_repeat_ngram_size=3
    )
    
    output = result[0]["generated_text"]
    
    # Extract text after the marker
    _, _, after = output.partition("INITIAL TEXT: ")
    
    return format_output(after)

def format_output(text: str) -> str:
    # Continue with further cleaning if necessary
    replacements = {
        '“': '"', '”': '"', "‘": "'", "’": "'",
        "\n ": "\n", "  ": " ", " :": ":",
        " ,": ",", " .": ".", " ;": ";"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)

# if __name__ == "__main__":
#     input_text = """
#         Prologue
#         “He should start bok
#         
#         Dared urged s the wods bgun to grw dar around the.
#         
#         "The wildiing re deaad
#         
#         “To the dead frighten you?"
#     """
#     
#     output = correct_ocr_text(input_text)
#     
#     print("Original:")
#     print(input_text)
#     print("\nCorrected:")
#     print(output)