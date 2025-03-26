import Levenshtein
from collections import Counter
import string

class Measures:
    def __init__(self):
        # les mesures ne sont pas encore calculés...
        self.levenshetein_distance = None
        self.character_accuracy = None
        self.substitutions = None
        self.deletions = None
        self.insertions = None
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TP_words = {}
        self.FN_words = {}
        self.FP_words = {}
        self.precision = 0
        self.recall = 0
        self.F1_score = 0

    
    def __str__(self):

        return (
            "-------------- Test Results -------------- \n"
            f"Levenshtein Distance : {self.levenshetein_distance}\n"
            f"Character Accuracy : {self.character_accuracy} %\n"
            f"Substitutions : {self.substitutions}\n"
            f"Deletions : {self.deletions}\n"
            f"Insertions : {self.insertions}\n"
            f"TP (true positives) : {self.TP}\n"
            f"FN (false negatives) : {self.FN}\n"
            f"FP (false positives) : {self.FP}\n"
            f"precision : {self.precision}\n"
            f"recall : {self.recall}\n"
            f"F1 score : {self.F1_score}"
        )
    



class MetricsComputation:
    def __init__(self):
        self.measures = Measures() # contient toutes les données des mesures effectuées
        

    def compute_levenshtein_distance(self, ocr_output, original_input):
        distance = Levenshtein.distance(ocr_output, original_input)
        self.measures.levenshetein_distance = distance

    def compute_char_accuracy(self, ocr_output, original_input):
        if self.measures.levenshetein_distance is None: # pas encore calculé ? 
            self.compute_levenshtein_distance(ocr_output, original_input) # on la calcule

        # On applique la formule pour trouver le char_accuracy
        distance = self.measures.levenshetein_distance 
        N = len(original_input)
        self.measures.character_accuracy = (1 - distance / N) * 100

    def compute_char_SDI(self, ocr_output, original_input):
        substitutions, deletions, insertions = 0, 0, 0

        # Comptage des erreurs
        for op, _, _ in Levenshtein.editops(ocr_output, original_input):
            if op == 'replace':
                substitutions += 1  
            elif op == 'insert':
                insertions += 1  
            elif op == 'delete':
                deletions += 1  

        # mettre à jour les données dans measures 
        self.measures.substitutions = substitutions
        self.measures.deletions = deletions
        self.measures.insertions = insertions


    def compute_all(self, ocr_output, original_input):
        self.compute_char_accuracy(ocr_output, original_input)
        self.compute_levenshtein_distance(ocr_output, original_input)
        self.compute_char_SDI(ocr_output, original_input)
        self.compute_TP_FN_FP(ocr_output, original_input)

    def print_results(self):
        print(self.measures)

    def compute_TP_FN_FP(self, ocr_output, original_input):
        output_split = ocr_output.split(" ")
        input_split = original_input.split(" ")

        output_counter = Counter(output_split) # par exemple {hello : 1, world : 1}
        input_counter = Counter(input_split) # meme chose mais avec l'input

        TP = 0 # Un mot bien reconnu et rédigé correctement 
        FN = 0 # Un mot mal reconnu ou oublié	
        FP = 0 # Un mot ajouté (ou modifié) qui n'était pas dans l'original_input

        for word in input_counter:
            true_positives = min(input_counter[word], output_counter[word])
            if true_positives > 0:

                TP += true_positives
                self.measures.TP_words[word] = true_positives

            missing_words = input_counter[word] - output_counter[word]            
            if missing_words > 0:
                FN += missing_words
                self.measures.FN_words[word] =  missing_words



        
        for word in output_counter:
            added_words = output_counter[word] - input_counter[word]

            if added_words > 0:
                FP += added_words
                self.measures.FP_words[word] = added_words

        self.measures.TP = TP
        self.measures.FN = FN
        self.measures.FP = FP

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        self.measures.precision = precision
        self.measures.recall = recall
        if precision + recall == 0:
            self.measures.F1_score = 0
        else:
            self.measures.F1_score = 2 * (precision * recall) /(precision + recall)