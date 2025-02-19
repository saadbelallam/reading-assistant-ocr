import metrics
import sys 


def utilisation():
    print("Utilisation results.py <output.txt> <ground_truth.txt>")

def main():
    if len(sys.argv) != 3:
        utilisation()
        sys.exit(1)

    output_file = sys.argv[1]
    ground_truth_file = sys.argv[2]

    with open(output_file, 'r', encoding='utf-8') as f:
        ocr_output = f.read()

    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        original_input = f.read()

    metrics_computation = metrics.MetricsComputation()
    metrics_computation.compute_all(ocr_output, original_input)
    metrics_computation.print_results()


main()