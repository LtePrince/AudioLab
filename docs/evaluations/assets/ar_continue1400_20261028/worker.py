"""Continue E14/E18 unchanged; reuse the established fixed-15 evaluator."""
from pathlib import Path
import importlib.util

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/anchored_ar.py').is_file())
SOURCE = ROOT/'docs/evaluations/assets/ar_continue1200_20261027/worker.py'
spec = importlib.util.spec_from_file_location('continued1200_evaluator', SOURCE)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def evaluate_with_fixed15(args, budget, device):
    previous.original_evaluate(args, budget, device)
    # The four e1200 baselines already exist and are frozen in launch_plan.json.
    # This hook executes within the original trainer's RNG-preserving fork.
    previous.evaluate15(args.checkpoint, args.mode, args.variant,
                        args.output/'fixed15', budget, device)


if __name__ == '__main__':
    previous.training.evaluate = evaluate_with_fixed15
    previous.training.main()
