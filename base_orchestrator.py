# Markham Lee (C) 2023 - 2026
# https://github.com/MarkhamLee/vision-pipeline-benchmarks
# Abstract base class defining the shared interface for all pipeline
# orchestrators. Both SequentialOrchestrator and AsyncOrchestrator
# must implement run() and get_metrics(), allowing the harness runner
# to treat them interchangeably.

from abc import ABC, abstractmethod


class BaseOrchestrator(ABC):

    @abstractmethod
    def run(self, source) -> None:
        """Execute the pipeline against the provided video source."""
        ...

    @abstractmethod
    def get_metrics(self) -> dict:
        """Return a dict describing the pipeline configuration and
        last-known runtime metrics. Used by the harness reporter."""
        ...
