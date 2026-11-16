"""Dedicated worker for persistent challenger jobs. Run separately from API."""
from __future__ import annotations

import argparse
import logging
import time
import uuid

from src.modelops.execution_runner import run_isolated
from src.modelops.training_jobs import TrainingJobStore

logger = logging.getLogger(__name__)


def process_one(jobs: TrainingJobStore, worker_id: str) -> bool:
    job = jobs.claim(worker_id)
    if not job:
        return False
    try:
        outcome = run_isolated(
            "train",
            owner_id=job["owner_id"],
            workspace_id=job["workspace_id"],
            min_f1_gain=job["min_f1_gain"],
        )
        jobs.complete(job["id"], worker_id, outcome)
        logger.info("Training job %s completed", job["id"])
    except Exception:
        logger.error("Training job %s failed in execution worker", job["id"])
        jobs.fail(job["id"], worker_id, "Training failed. Check dataset and model compatibility.")
    return True


def main():
    parser = argparse.ArgumentParser(description="ModelOps challenger training worker")
    parser.add_argument("--once", action="store_true", help="Process one job, then exit")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be between 1 and 60")
    logging.basicConfig(level=logging.INFO)
    jobs = TrainingJobStore()
    jobs.initialize()
    worker_id = uuid.uuid4().hex
    logger.info("ModelOps worker started")
    while True:
        processed = process_one(jobs, worker_id)
        if args.once:
            break
        if not processed:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
