# -------------------------------
# Pipeline Execution Targets
# -------------------------------
# Base directories
RESULT_DIR = tests/results
IMAGE_DIR = tests/images

export PYTHONPATH := $(PYTHONPATH):$(CURDIR)
# Install all dependencies
install:
	@echo "Exécution du script d'installation (setup.sh)..."
	@chmod +x setup.sh && ./setup.sh

# Supprimer tout ce qui a été généré
delete:
	@echo "Suppression de l'environnement et des modèles"
	rm -rf .venv/
	rm -rf models/camembert-base/
	@echo "Tout a été nettoyé."

# Process a single image (pass 'image=<path>' and optional 'time=true')
process-image:
	@echo "\n Processing image: $(image)"
	@if [ "$(time)" = "true" ]; then TIMING_FLAG=-t; else TIMING_FLAG=; fi; \
	python3 src/main.py --input $(image) --output-dir $(RESULT_DIR) $$TIMING_FLAG 

# Process all images in tests/images (organized by category)
process-all:
	@echo "\n🔍 Processing all images in $(IMAGE_DIR):"
	@find $(IMAGE_DIR) -type f \( -iname '*.jpg' -o -iname '*.png' \) | while read file; do \
	    echo "\n Processing $$file"; \
	    if [ "$(time)" = "true" ]; then TIMING_FLAG=-t; else TIMING_FLAG=; fi; \
	    python3 src/main.py --input $$file --output-dir $(RESULT_DIR) $$TIMING_FLAG; \
	done

# Generic evaluation target.
evaluate:
	@if [ -z "$(image)" ]; then \
	    echo "Usage: make evaluate image=<path_to_image>"; \
	    exit 1; \
	fi; \
	python3 src/utils/metrics_evaluation.py $(image)

# Evaluate all images in tests/images
evaluate-all:
	@echo "\n Evaluating all results in $(RESULT_DIR):"
	@python3 src/utils/metrics_evaluation_folder.py $(RESULT_DIR)

MAKEFLAGS += --no-print-directory

# Capture image from webcam and process it
webcam:
	@echo "\n Starting webcam capture..."
	@python3 src/utils/webcam_capture.py $(IMAGE_DIR)
	@if [ -n "$$(ls -t $(IMAGE_DIR)/webcam/*.jpg 2>/dev/null | head -1)" ]; then \
	    latest_image=$$(ls -t $(IMAGE_DIR)/webcam/*.jpg | head -1); \
	    make process-image image=$$latest_image; \
	else \
	    @echo "\n No webcam images found to process."; \
	fi

# Clean generated results
clean:
	@echo "\n Cleaning up generated results in $(RESULT_DIR)/**"
	@rm -rf $(RESULT_DIR)

# Help target
help:
	@echo "\nUsage:"
	@echo "  make install                                 Install all dependencies from requirements.txt"
	@echo "  make process-image image=<path> [time=true]  Process a single image"
	@echo "  make process-all [time=true]                 Process all images in $(IMAGE_DIR)"
	@echo "  make evaluate image=<path>                   Evaluate OCR output of a single image"
	@echo "  make evaluate-all                            Evaluate OCR output for all images in $(IMAGE_DIR)"
	@echo "  make clean                                   Remove all generated results in $(RESULT_DIR)"
	@echo "  make help                                    Show this help message"

.PHONY: install process-image process-all evaluate clean help