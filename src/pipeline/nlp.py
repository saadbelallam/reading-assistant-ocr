################
#    Step 6    #
################
import os
import warnings
import transformers
import torch
from collections import defaultdict

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
warnings.filterwarnings("ignore")  # Suppress Python warnings
transformers.logging.set_verbosity_error()  # Suppress Hugging Face Transformers logs
torch.set_warn_always(False)  # Suppress PyTorch warnings

from transformers import CamembertTokenizer, CamembertForMaskedLM

# Lazy-loaded globals
_tokenizer = None
_model = None

def download_models():
    """Download and save the CamemBERT tokenizer and model to a local directory."""
    

    # Specify the local directory
    model_dir = "models/camembert-base"
    os.makedirs(model_dir, exist_ok=True)

    # Download and save the tokenizer and model
    print("└──Downloading CamemBERT tokenizer and model...")
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    tokenizer.save_pretrained(model_dir)

    model = CamembertForMaskedLM.from_pretrained("camembert-base")
    model.save_pretrained(model_dir)

    print(f"└──CamemBERT model and tokenizer saved to {model_dir}")

def _load_model(model_dir: str):
    """Lazy-load the tokenizer and model from local directory."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        local_model_dir = model_dir
        if not os.path.exists(local_model_dir):
            download_models()
        _tokenizer = CamembertTokenizer.from_pretrained(local_model_dir)
        _model = CamembertForMaskedLM.from_pretrained(local_model_dir)
    
    return _tokenizer, _model

def calculate_similarity(ocr_word, candidate_word):
    """Calculate the number of matched letters between ocr_word and candidate_word without repetition."""
    ocr = ocr_word.lower()
    candidate = candidate_word.lower()
    
    ocr_counts = defaultdict(int)
    for c in ocr:
        ocr_counts[c] += 1
    
    candidate_counts = defaultdict(int)
    for c in candidate:
        candidate_counts[c] += 1
    
    score = 0
    for c in candidate_counts:
        if c in ocr_counts:
            score += min(candidate_counts[c], ocr_counts[c])
    return score

def correct_text(text: str, masks: list, output_dir: str, nlp_config: dict) -> str:
    """
    Remplace les <mask> dans le texte par des mots prédits par le modèle.
    """
    print("Running NLP ...")
    model_dir = nlp_config["model_path"]
    similarity_threshold = nlp_config["similarity_threshold"]
    top_k = nlp_config["top_k"]

    tokenizer, model = _load_model(model_dir)

    # Tokenizer le texte
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    
    # Prédire les mots masqués
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Récupérer les prédictions pour les tokens masqués
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_tokens = []

    for i, mask_index in enumerate(mask_token_index):
        ocr_prediction = masks[i]

        # Get top predictions
        top_predictions = logits[0, mask_index].topk(top_k).indices
        candidates = [tokenizer.decode([idx]).strip() for idx in top_predictions]

        # Find best match with OCR prediction
        best_match = None
        highest_score = -1
        for token in candidates:
            similarity = calculate_similarity(ocr_prediction, token)
            
            if similarity > highest_score:
                highest_score = similarity
                best_match = token

        ocr_len = len(ocr_prediction)
        similarity_ratio = highest_score / ocr_len if ocr_len > 0 else 0.0

        if similarity_ratio < similarity_threshold or not best_match:
            best_match = ocr_prediction
        predicted_tokens.append(best_match)

    # Remplacer les <mask> par les prédictions
    corrected_text = text
    for token in predicted_tokens:
        corrected_text = corrected_text.replace("<mask>", token, 1)
    
    # Save results
    with open(f"{output_dir}/6_nlp_output.txt", "w") as f:
        f.write(corrected_text)
    return corrected_text