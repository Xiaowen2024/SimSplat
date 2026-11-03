from __future__ import annotations

from abc import ABC, abstractmethod

from sim_splat.types import DataparserOutputs


class DataParser(ABC):
    """Converts raw scene files into framework-neutral scene metadata."""

    @abstractmethod
    def get_dataparser_outputs(self) -> DataparserOutputs:
        """Parse one scene once; never load full RGB tensors here."""
        raise NotImplementedError
