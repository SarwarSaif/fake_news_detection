# data_utils.py

import csv
import json
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jsonlines
import numpy as np
import torch
import yaml

# --- Constants for Datasets and Modes ---
# Prevents typos and makes the code easier to maintain.
DATASET_LIAR = "LIAR-PLUS"
DATASET_CONSTRAINT = "Constraint"
DATASET_GOSSIPCOP = "GOSSIPCOP"
DATASET_POLITIFACT = "POLITIFACT"

MODE_BINARY = "binary"
MODE_MULTICLASS = "multiclass"


class TELLERDataLoader:
    """
    A comprehensive data loader for the TELLER project.
    Handles loading, processing, and structuring data from various fake news datasets.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the data loader with a configuration dictionary.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters like
                                    dataset_name, data_path, mode, gq_file, sq_file, etc.
        """
        self.config = config
        self.dataset_name = config["dataset_name"]
        self.data_path = Path(config["data_path"])
        
        # Define label mapping rules based on dataset
        self._label_rules = self._get_label_rules()

    def _get_label_rules(self) -> Dict[str, Dict[str, str]]:
        """Defines the label re-mapping rules for supported datasets."""
        rules = {
            DATASET_LIAR: {
                MODE_MULTICLASS: {
                    'half-true': 'half true', 'false': 'false', 'mostly-true': 'mostly true',
                    'barely-true': 'barely true', 'true': 'true', 'pants-fire': 'pants fire'
                },
                MODE_BINARY: {
                    'false': 'false', 'mostly-true': 'true', 'barely-true': 'false',
                    'true': 'true', 'pants-fire': 'false'
                }
            },
            DATASET_CONSTRAINT: {MODE_BINARY: {'fake': 'false', 'real': 'true'}},
            DATASET_GOSSIPCOP: {MODE_BINARY: {'fake': 'false', 'real': 'true'}},
            DATASET_POLITIFACT: {MODE_BINARY: {'fake': 'false', 'real': 'true'}}
        }
        return rules

    def _load_and_combine_splits(self) -> Dict[str, List[Dict]]:
        """Loads train, validation, and test splits from jsonl files."""
        return {
            "train": read_jsonl_file(self.data_path / "train.jsonl"),
            "val": read_jsonl_file(self.data_path / "val.jsonl"),
            "test": read_jsonl_file(self.data_path / "test.jsonl"),
        }

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Processes a single raw data sample into a standardized format."""
        mode = self.config.get("mode", MODE_BINARY)
        label_map = self._label_rules.get(self.dataset_name, {}).get(mode, {})
        
        if self.dataset_name == DATASET_LIAR:
            raw_label = sample.get("label")
            if raw_label not in label_map:
                return None
            return {
                "id": sample.get("statement id"),
                "message": sample.get("statement", "").strip(),
                "evidence": sample.get("evidence", "").strip(),
                "label": label_map.get(raw_label),
            }
        elif self.dataset_name == DATASET_CONSTRAINT:
            return {
                "id": str(sample.get("id")),
                "message": sample.get("tweet", "").strip(),
                "evidence": None, # Evidence handled separately if needed
                "label": label_map.get(sample.get("label")),
            }
        elif self.dataset_name in [DATASET_GOSSIPCOP, DATASET_POLITIFACT]:
            return {
                "id": str(sample.get("id")),
                "message": sample.get("message", "").strip(),
                "evidence": None, # Evidence handled separately if needed
                "label": label_map.get(sample.get("label")),
            }
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")

    def load_processed_data(self) -> Dict[str, Union[List[Dict], Dict]]:
        """
        Loads and processes data for training/evaluating the full TELLER framework.
        This corresponds to the logic of the original `load_data_for_expert`.

        Returns:
            Dict: A dictionary containing 'train', 'val', 'test' splits,
                  and optionally 'gq' (Generated Questions truth values) and
                  'sq' (Sub-Questions).
        """
        all_splits = self._load_and_combine_splits()
        processed_dataset = {}

        for split_name, split_data in all_splits.items():
            processed_split = []
            for sample in split_data:
                processed_sample = self._process_sample(sample)
                if processed_sample:
                    processed_split.append(processed_sample)
            processed_dataset[split_name] = processed_split
        
        # Load supplementary GQ/SQ files
        if self.config.get("gq_file"):
            gq_path = self.data_path / self.config["gq_file"]
            gq_data = read_json_file(gq_path)
            
            if self.config.get("evo_flag") and self.config.get("evo_file"):
                evo_path = self.data_path / self.config["evo_file"]
                evo_data = read_json_file(evo_path)
                gq_data = self._concatenate_evo_gq(gq_data, evo_data)
                
            processed_dataset["gq"] = gq_data
        
        if self.config.get("sq_file"):
            sq_path = self.data_path / self.config["sq_file"]
            processed_dataset["sq"] = read_json_file(sq_path)

        return processed_dataset

    @staticmethod
    def _concatenate_evo_gq(dict1: Dict, dict2: Dict) -> Dict:
        """Merges two dictionaries of generated questions for intervention experiments."""
        result_dict = deepcopy(dict1)
        for key, value in dict2.items():
            if key in result_dict:
                result_dict[key].update(value)
            else:
                result_dict[key] = value
        return result_dict

# --- Generic Helper Functions ---

def read_jsonl_file(file_path: Union[str, Path]) -> List[Dict]:
    """Reads a JSON Lines file and returns a list of dictionaries."""
    with jsonlines.open(file_path, mode='r') as reader:
        return [item for item in reader]

def read_json_file(file_path: Union[str, Path]) -> Any:
    """Reads a standard JSON file."""
    with open(file_path, mode='r', encoding='utf-8') as f:
        return json.load(f)

def write_json_file(data: Any, path: Union[str, Path]):
    """Writes data to a JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4) # indent for readability

def read_yaml_file(file_path: Union[str, Path]) -> Dict:
    """Reads a YAML file safely."""
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            raise

def seed_everything(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Example Usage ---
if __name__ == "__main__":
    # This section demonstrates how to use the new DataLoader class.
    # It mimics loading a configuration from a YAML file.
    
    print("--- DEMONSTRATING THE NEW DATALOADER ---")
    
    # 1. Define a mock configuration (in a real scenario, this would be loaded from a YAML file)
    mock_config = {
        "dataset_name": DATASET_LIAR,
        "data_path": "./path/to/your/liar/data", # IMPORTANT: Change this to your actual data path
        "mode": MODE_BINARY,
        "gq_file": "flan-t5-xl_logics.json",
        "sq_file": "sq.json",
        "evo_flag": False,
        "evo_file": None,
    }

    print(f"Using configuration:\n{json.dumps(mock_config, indent=2)}\n")

    # 2. Set a seed for reproducibility
    seed_everything(42)

    # 3. Create mock data files for the demonstration to run
    # (In a real project, these files would already exist)
    print("Creating mock data files for demonstration...")
    mock_data_path = Path(mock_config["data_path"])
    mock_data_path.mkdir(parents=True, exist_ok=True)
    
    mock_train_data = [
        {'statement id': '1.json', 'statement': 'This is a true statement.', 'label': 'true', 'evidence': 'Source A says so.'},
        {'statement id': '2.json', 'statement': 'This is a false statement.', 'label': 'pants-fire', 'evidence': 'Source B refutes it.'}
    ]
    with jsonlines.open(mock_data_path / "train.jsonl", "w") as writer:
        writer.write_all(mock_train_data)
    with jsonlines.open(mock_data_path / "val.jsonl", "w") as writer:
        writer.write_all(mock_train_data)
    with jsonlines.open(mock_data_path / "test.jsonl", "w") as writer:
        writer.write_all(mock_train_data)
        
    mock_gq_data = {"1.json": {"Q1": 0.9, "Q2": -0.8}, "2.json": {"Q1": -0.7, "Q2": 0.6}}
    write_json_file(mock_gq_data, mock_data_path / mock_config["gq_file"])
    
    mock_sq_data = {"1.json": {"STATEMENTS": ["S1", "S2"]}, "2.json": {"STATEMENTS": ["S3"]}}
    write_json_file(mock_sq_data, mock_data_path / mock_config["sq_file"])
    
    print("Mock files created.\n")

    # 4. Instantiate the DataLoader and load the data
    try:
        data_loader = TELLERDataLoader(config=mock_config)
        full_dataset = data_loader.load_processed_data()

        # 5. Inspect the loaded data
        print("--- INSPECTING LOADED DATA ---")
        print(f"Train set has {len(full_dataset['train'])} samples.")
        print("First train sample:", full_dataset['train'][0])
        print("\nGenerated Questions (GQ) data for first sample:", full_dataset.get('gq', {}).get('1.json'))
        print("Sub-Questions (SQ) data for first sample:", full_dataset.get('sq', {}).get('1.json'))

    except FileNotFoundError:
        print("\nERROR: Mock data path not found.")
        print(f"Please create the directory: {mock_data_path} or update the path in `mock_config`.")
    except Exception as e:
        print(f"An error occurred: {e}")

        