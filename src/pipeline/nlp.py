################
#    Step 6    #
################

from transformers import CamembertTokenizer, CamembertForMaskedLM
import torch

# Charger le modèle et le tokenizer
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")

def correct_text(text: str, output_dir: str) -> str:
    """
    Remplace les <mask> dans le texte par des mots prédits par le modèle.
    """
    print("Running NLP ...")
    # Tokenizer le texte
    inputs = tokenizer(text, return_tensors="pt")
    
    # Prédire les mots masqués
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Récupérer les prédictions pour les tokens masqués
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_tokens = []
    for mask_index in mask_token_index:
        predicted_token_id = logits[0, mask_index].argmax(dim=-1)
        predicted_token = tokenizer.decode(predicted_token_id)
        predicted_tokens.append(predicted_token)
    
    # Remplacer les <mask> par les prédictions
    corrected_text = text
    for token in predicted_tokens:
        corrected_text = corrected_text.replace("<mask>", token, 1)
    
    # Save results
    with open(f"{output_dir}/6_nlp_output.txt", "w") as f:
        f.write(corrected_text)
    return corrected_text