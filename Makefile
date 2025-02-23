# -------------------------------
# Pipeline Execution Targets
# -------------------------------
# Base directories
RESULT_DIR = tests/results
IMAGE_DIR = tests/images

# Process a single image (pass 'image=<path>' and optional 'time=true')
process-image:
	@echo "\n🚀 Processing image: $(image)"
	@if [ "$(time)" = "true" ]; then TIMING_FLAG=-t; else TIMING_FLAG=; fi; \
	python src/main.py --input $(image) --output-dir $(RESULT_DIR) $$TIMING_FLAG

# Process all images in tests/images (organized by category)
process-all:
	@echo "\n🔍 Processing all images in $(IMAGE_DIR):"
	@find $(IMAGE_DIR) -type f \( -iname '*.jpg' -o -iname '*.png' \) | while read file; do \
	    echo "\n📂 Processing $$file"; \
	    if [ "$(time)" = "true" ]; then TIMING_FLAG=-t; else TIMING_FLAG=; fi; \
	    python src/main.py --input $$file --output-dir $(RESULT_DIR) $$TIMING_FLAG; \
	done

# Generic evaluation target.
evaluate:
	@if [ -z "$(image)" ]; then \
	    echo "Usage: make evaluate image=<path_to_image>"; \
	    exit 1; \
	fi; \
	python3 src/utils/metrics_evaluation.py $(image)

# Evaluate all images in tests/images
evaluate_all:
	@echo "\n🔍 Evaluating all images in $(IMAGE_DIR):"
	@find $(IMAGE_DIR) -type f \( -iname '*.jpg' -o -iname '*.png' \) | while read file; do \
	    echo "\n📂 Evaluating $$file"; \
	    python3 src/utils/metrics_evaluation.py "$$file"; \
	done

# Clean generated results
clean:
	@echo "\n🧹 Cleaning up generated results in $(RESULT_DIR)/**"
	@rm -rf $(RESULT_DIR)

# Help target
help:
	@echo "\nUsage:"
	@echo "  make process-image image=<path> [time=true]  Process a single image"
	@echo "  make process-all [time=true]                 Process all images in $(IMAGE_DIR)"
	@echo "  make evaluate image=<path>                   Evaluate OCR output of a single image"
	@echo "  make evaluate_all                            Evaluate OCR output for all images in $(IMAGE_DIR)"
	@echo "  make clean                                   Remove all generated results in $(RESULT_DIR)"
	@echo "  make help                                    Show this help message"

.PHONY: process-image process-all evaluate clean help