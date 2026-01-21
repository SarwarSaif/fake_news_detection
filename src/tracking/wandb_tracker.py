import wandb
from .base_tracker import BaseTracker

class WandbTracker(BaseTracker):
    def __init__(self, project, run_name=None, config=None, group=None):
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self.group = group
        self.run = None
        # Persistent table to be logged at the end
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

    # THIS WAS THE MISSING PIECE
    def log(self, metrics: dict, step: int = None):
        """Standard logging for metrics like loss and accuracy."""
        if self.run:
            wandb.log(metrics, step=step)

    def log_llm_interaction(self, sample_id, text, features, verdict, label):
        """Adds a row to the persistent table for qualitative analysis."""
        self.prediction_table.add_data(
            sample_id,
            text[:500] if text else "",
            features[0],       # Subjectivity
            features[1],       # Reasoning
            features[2],       # CLIP Similarity
            verdict,           # Prediction
            label              # Actual
        )

    def finish(self):
        """Uploads the table and closes the run."""
        if self.run:
            # Only log the table if it has data
            if len(self.prediction_table.data) > 0:
                wandb.log({"evaluation_results": self.prediction_table})
            wandb.finish()