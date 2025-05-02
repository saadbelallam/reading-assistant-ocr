################
#    Step 6    #
################
import os
import warnings
import torch
from collections import defaultdict
from transformers import CamembertTokenizer, CamembertForMaskedLM
from fuzzywuzzy import fuzz
from Levenshtein import distance as levenshtein_distance

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
torch.set_warn_always(False)

# Globals for lazy loading
_tokenizer = None
_model = None

def download_models():
    """Télécharge le modèle CamemBERT localement."""
    model_dir = "models/camembert-base"
    os.makedirs(model_dir, exist_ok=True)

    print("└──Téléchargement du modèle CamemBERT...")
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    tokenizer.save_pretrained(model_dir)

    model = CamembertForMaskedLM.from_pretrained("camembert-base")
    model.save_pretrained(model_dir)
    print(f"└──Modèle et tokenizer enregistrés dans {model_dir}")

def _load_model(model_dir: str):
    """Charge le tokenizer et le modèle."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        if not os.path.exists(model_dir):
            download_models()
        _tokenizer = CamembertTokenizer.from_pretrained(model_dir)
        _model = CamembertForMaskedLM.from_pretrained(model_dir)
    return _tokenizer, _model

def calculate_similarity(ocr_word, candidate_word, method="fuzz"):
    ocr = ocr_word.lower()
    candidate = candidate_word.lower()

    if method == "levenshtein":
        max_len = max(len(ocr), len(candidate))
        return 1.0 - levenshtein_distance(ocr, candidate) / max_len if max_len > 0 else 0.0

    elif method == "fuzz":
        return fuzz.ratio(ocr, candidate) / 100.0

    else:  # rudimentary
        ocr_counts = defaultdict(int)
        for c in ocr:
            ocr_counts[c] += 1
        candidate_counts = defaultdict(int)
        for c in candidate:
            candidate_counts[c] += 1
        score = sum(min(candidate_counts[c], ocr_counts[c]) for c in candidate_counts)
        return score / len(ocr) if len(ocr) > 0 else 0.0

def correct_text(text: str, masks: list, output_dir: str, nlp_config: dict) -> str:
    """
    Remplace les <mask> dans le texte par les meilleures prédictions du modèle.
    """
    print("⏳ Traitement NLP ...")
    model_dir = nlp_config["model_path"]
    similarity_threshold = nlp_config["similarity_threshold"]
    top_k = nlp_config["top_k"]
    similarity_method = nlp_config.get("similarity_method", "rudimentary")

    tokenizer, model = _load_model(model_dir)
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_tokens = []

    for i, mask_index in enumerate(mask_token_index):
        ocr_prediction = masks[i]
        top_predictions = logits[0, mask_index].topk(top_k).indices
        candidates = [tokenizer.decode([idx]).strip() for idx in top_predictions]

        best_match = None
        highest_score = -1.0
        for token in candidates:
            score = calculate_similarity(ocr_prediction, token, method=similarity_method)
            if score > highest_score:
                highest_score = score
                best_match = token

        if highest_score < similarity_threshold or not best_match:
            best_match = ocr_prediction
        predicted_tokens.append(best_match)

    # Remplacer les masques
    corrected_text = text
    for token in predicted_tokens:
        corrected_text = corrected_text.replace("<mask>", token, 1)

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/6_nlp_output.txt", "w") as f:
        f.write(corrected_text)

    return corrected_text

