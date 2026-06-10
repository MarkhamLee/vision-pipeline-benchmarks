import os
import sys
import torch
from ultralytics import YOLO


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.logging_utils import console_logging  # noqa: E402

logger = console_logging('model_loader')

# separate class for each type of model loader
# e.g., separate class for an RKNN loader


class CudaYoloLoader:

    def _init__(
      self
    ):
        self.cuda_logger = console_logging('cuda-logger')
        self._set_gpu_accel_parameters()

    def _set_gpu_accel_parameters(self) -> str:
        if torch.cuda.is_available():
            device = "cuda:0"
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            self.logger.info("Running on device: %s", device)
            return device
        self.logger.warning("CUDA not available, shutting down test")
        sys.exit(1)

    def _load_yolo_model(self, model_path: str) -> YOLO:
        model = YOLO(model_path)
        model.to(self.device)
        self.logger.info("YOLO model loaded from: %s", model_path)
        return model
