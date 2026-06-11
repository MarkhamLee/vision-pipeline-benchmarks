# Markham Lee (C) 2023 - 2026
# https://github.com/MarkhamLee/vision-pipeline-benchmarks
# Hardware-specific YOLO model loaders.
# Add a new class per hardware target (e.g., RKNNYoloLoader for NPU).

import sys
import torch
from ultralytics import YOLO
from utils.logging_utils import console_logging

logger = console_logging('model_loader')


class CudaYoloLoader:

    def __init__(self) -> None:
        # Note: logger name uses instance logger, not module-level logger
        self.device = self._set_gpu_accel_parameters()

    def _set_gpu_accel_parameters(self) -> str:
        if torch.cuda.is_available():
            device = "cuda:0"
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info("CUDA available — running on device: %s", device)
            return device

        logger.warning("CUDA not available, shutting down")
        sys.exit(1)

    def load_yolo_model(self, model_path: str) -> YOLO:
        """Load a YOLO model and move it to the configured CUDA device."""
        model = YOLO(model_path)
        model.to(self.device)
        logger.info("YOLO model loaded: %s on %s", model_path, self.device)
        return model
