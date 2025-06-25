from configargparse import ArgParser
from pathlib import Path

class Config:
    def __init__(self):
        parser = ArgParser(
            default_config_files=["configs/default.yaml"]
        )
        parser.add("--config", is_config_file=True, help="Path to config file")

        # Arguments disponibles dans le .yaml ou la ligne de commande
        parser.add("--root_dir", type=Path, help="Project root directory", default=Path("."))
        parser.add("--debug_mode", type=bool, help="Enable debug logs", default=False)

        # Parse all args
        self.args = parser.parse_args() # Convertit en objet

    def __getattr__(self, name):
        return getattr(self.args, name)

# Singleton accessible dans tout le projet
config = Config()
