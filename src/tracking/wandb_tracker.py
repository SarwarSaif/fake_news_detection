# src/tracking/wandb_tracker.py
import wandb
from .base_tracker import BaseTracker

class WandbTracker(BaseTracker):
    def __init__(self, project, run_name=None, config=None, group=None):
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self.group = group
        self.run = None

    def init(self):
        self.run = wandb.init(
            project=self.project,
            name=self.run_name,
            config=self.config,
            group=self.group  # pass group if set
        )

    def log(self, metrics: dict, step: int = None):
        wandb.log(metrics, step=step)

    def log_llm_interaction(
        self, sample_id, dataset, image_path, text, raw_output, parsed, verdict, confidence, step=None
    ):
        """Convenience wrapper for logging fact-check samples with scalars + table-friendly strings."""

        # Scalars / numeric metrics
        metrics = {
            "confidence_mean": confidence,
        }

        # Try to log image (if available)
        if image_path and isinstance(image_path, str) and image_path.strip():
            try:
                metrics["sample_image"] = wandb.Image(
                    image_path,
                    caption=f"Verdict: {verdict} | Text: {text[:200]}..."
                )
            except Exception as e:
                print(f"⚠️ Could not log image to wandb: {e}")

        # Create a table for richer logs (strings, lists, JSON)
        table = wandb.Table(
            columns=[
                "sample_id",
                "dataset",
                "text",
                "model_raw_output",
                "parsed_json",
                "final_verdict",
            ]
        )
        table.add_data(
            sample_id,
            dataset,
            text[:1000],             # truncate long text
            raw_output[:1000] if isinstance(raw_output, str) else str(raw_output),
            str(parsed),             # force JSON/dict to string
            verdict,
        )

        # Log both metrics + table
        wandb.log({"metrics": metrics, "predictions": table}, step=step)


    def finish(self):
        wandb.finish()
