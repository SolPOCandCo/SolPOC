from multiprocessing import freeze_support


def main() -> int:
    """Lance SolPOC Optimizer."""

    freeze_support()

    from solpoc_optimizer.experiences_scheduler.scheduler import (
        main as run_scheduler,
    )

    return run_scheduler()
