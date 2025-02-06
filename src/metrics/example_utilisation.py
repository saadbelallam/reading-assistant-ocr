import metrics 

original_input = "Voici un exemple d'utilisation des metriques. Ceci est le texte original sans erreurs."
ocr_output = "Voici un exempl d'utilisotion de metriques. Ceci est le texte texte original sans sans err"

metrics_computation = metrics.MetricsComputation()
metrics_computation.compute_all(ocr_output, original_input)
metrics_computation.print_results()