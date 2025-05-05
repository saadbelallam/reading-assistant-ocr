from transformers import CamembertTokenizer, CamembertForMaskedLM

model_name = "camembert-base"
target_dir = "models/camembert-base"

print(f"Téléchargement du tokenizer vers {target_dir} ...")
tokenizer = CamembertTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(target_dir)

print(f"Téléchargement du modèle vers {target_dir} ...")
model = CamembertForMaskedLM.from_pretrained(model_name)
model.save_pretrained(target_dir)

print("Téléchargement terminé !")
