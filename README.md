# Guide :
Se placer dans src puis `python3 image_vision.py` 


# 26/01/2025

Prochain RDV le 6/02 à 16h.  
[GitHub Repository](https://github.com/xiaoyu258/DocProj) for image correction

Consignes du projet :  
[Consignes sur GitLab](https://gitlab.ensimag.fr/projets-de-sp-cialit/infos)  

## Pour la fois prochaine :
- tests : identifier les forces et faiblesses  
- intégration de l'outil de correction de déformation  
- début de revue de littérature sur les techniques OCR et corrections par modèles de langage  
- établissement de métriques : à regarder precision, recall, F1 score, ROC curve  


# 06/02/2025

Prochain RDV le 19/02 à 14h.  


Quelques références :

NLP pour correction d'OCR:
https://medium.com/doma/using-nlp-bert-to-improve-ocr-accuracy-385c98ae174c
https://ieeexplore.ieee.org/document/10463478
https://blog.ml6.eu/ocr-correction-with-byt5-5994d1217c07


Faire votre propre jeu de référence : imprimer un (ou idéalement plusieurs) texte dont vous avez le contenu sous forme texte, le prendre en photo sous plusieurs angles (stratifié par difficulté) et construire des métriques permettant de mesurer la qualité du modèle.

      
Bien garder en tête qu'on peut s'appuyer sur plusieurs prises de vues du même texte, afin de trouver un texte consensus.
