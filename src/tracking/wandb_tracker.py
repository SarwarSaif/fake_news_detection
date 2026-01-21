# src/tracking/wandb_tracker.py
import wandb
from .base_tracker import BaseTracker

# src/tracking/wandb_tracker.py

class WandbTracker(BaseTracker):
    def __init__(self, project, run_name=None, config=None, group=None):
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self.group = group
        self.run = None
        # Initialize a persistent table
        self.prediction_table = wandb.Table(
            columns=[
                "sample_id", "text", "subjectivity", 
                "reasoning", "similarity", "final_verdict", "ground_truth"
            ]
        )

    def init(self):
        self.run = wandb.init(
            project=self.project,
            name=self.run_name,
            config=self.config,
            group=self.group
        )

    def log_llm_interaction(self, sample_id, text, features, verdict, label):
        """Adds a row to the persistent table."""
        self.prediction_table.add_data(
            sample_id,
            text[:500],        # Text
            features[0],       # Subjectivity
            features[1],       # Reasoning
            features[2],       # CLIP Similarity
            verdict,           # Prediction
            label              # Actual
        )

    def finish(self):
        # Log the completed table at the very end
        if self.run:
            wandb.log({"evaluation_results": self.prediction_table})
            wandb.finish()
