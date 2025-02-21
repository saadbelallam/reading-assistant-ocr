# detects the language before jumping

################
#    Step 6    #
################

import language_tool_python

def correct_text(texte: str) -> str:
    """
    Corrige le texte français en utilisant LanguageTool.
    
    Cette fonction corrige les erreurs telles que :
      - Mots coupés ou fusionnés
      - Caractères manquants ou endommagés
      - Transpositions de lettres
      - Ponctuation manquante
    
    Args:
        texte (str): Texte OCR brut.
        
    Returns:
        str: Texte corrigé.
    """
    print("Running NLP ...")
    tool = language_tool_python.LanguageTool('fr')
    # Check for issues
    matches = tool.check(texte)
    # Use the tool's own correct method
    texte_corrige = tool.correct(texte)
    return texte_corrige
