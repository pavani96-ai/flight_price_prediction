from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    training_file_name: Path
    testing_file_name: Path