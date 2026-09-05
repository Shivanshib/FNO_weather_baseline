"""
Trains and evaluates (coarse inference target) the SAME model config at 
several different seeds (default 3: 42, 43, 44), then aggregates the mean 
and standard deviation of the lat-weighted RMSE. Both the training curve 
and the evaluation RMSE-vs-lead-time/ACC-vs-lead-time curves across these seeds

Usage:
    python scripts/run_seed_ensemble.py --config configs/baseline_fno.yaml
    python scripts/run_seed_ensemble.py --config configs/baseline_fno.yaml --experiment configs/experiments/target_mode_direct_80_50.yaml
    python scripts/run_seed_ensemble.py --config configs/baseline_fno.yaml --seeds 0 1 2 --targets coarse
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from weather_fno.config import derive_run_paths, load_config, resolve_device, save_config_snapshot, set_seed
from weather_fno.inference.evaluate import run_full_evaluation
from weather_fno.training.run import run_training
from weather_fno.utils.plotting import plot_acc_vs_lead_time, plot_history_mean_std, plot_rmse_vs_lead_time


HEADLINE_CHANNELS = ["t2m", "z500", "mslp", "u10"]

DEFAULT_SEEDS = [42, 43, 44]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_fno.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    # coarse targets
    parser.add_argument("--targets", type=str, nargs="+", default=["coarse"])
    args = parser.parse_args()

    if len(set(args.seeds)) < 2:
        raise ValueError(f"--seeds needs at least 2 DISTINCT values to compute a standard "
                          f"deviation, got {args.seeds}")

    base_cfg = load_config(args.config, override_path=args.experiment)
    base_run_name = base_cfg.run_name

    available = {t.name for t in base_cfg.inference.targets}
    unknown = set(args.targets) - available
    if unknown:
        raise ValueError(f"--targets {sorted(unknown)} not found. Available: {sorted(available)}")
    base_cfg.inference.targets = [t for t in base_cfg.inference.targets if t.name in args.targets]

    histories = []
    eval_results_by_target = {t.name: [] for t in base_cfg.inference.targets}

    for seed in args.seeds:
        # A per-seed Config, not a per-seed experiment YAML -- only
        # run_name/training.seed differ from base_cfg, so mutating an
        # already-resolved Config in memory is simpler than trying to
        # layer a second --experiment override file on top of the user's
        # own (load_config only supports one).
        cfg = copy.deepcopy(base_cfg)
        reuse_base_run = seed == base_cfg.training.seed
        if not reuse_base_run:
            cfg.run_name = f"{base_run_name}_seed{seed}"
            cfg.training.seed = seed
            derive_run_paths(cfg)  # re-derive every output path for the new run_name

        best_ckpt_path = Path(cfg.training.checkpoint_dir) / "best.pt"
        already_trained = reuse_base_run and best_ckpt_path.exists()
        already_evaluated = already_trained and all(
            (Path(cfg.inference.output_dir) / f"{t.name}_eval_metrics.npz").exists()
            for t in cfg.inference.targets
        )

        tag = "  (reusing existing run -- matches the base config's own seed)" if already_trained else ""
        print(f"\n{'=' * 70}\nSeed {seed}  ({cfg.run_name}){tag}\n{'=' * 70}")

        if already_trained:
            history = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)["history"]
        else:
            device = resolve_device(cfg.training.device)
            set_seed(cfg.training.seed)  
            save_config_snapshot(cfg)
            history = run_training(cfg, device)
        histories.append(history)

        if already_evaluated:
            results = {
                t.name: dict(np.load(Path(cfg.inference.output_dir) / f"{t.name}_eval_metrics.npz"))
                for t in cfg.inference.targets
            }
        else:
            results = run_full_evaluation(cfg, headline_channels=HEADLINE_CHANNELS)
        for name, result in results.items():
            eval_results_by_target[name].append(result)

    ensemble_dir = Path(base_cfg.training.output_dir) / f"{base_run_name}_seed_ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 70}\nAggregating {len(args.seeds)} seeds -> {ensemble_dir}\n{'=' * 70}")

    # Training curve: mean/std of lat-weighted RMSE (sqrt of the
    # logged MSE) per epoch, across seeds.
    train_rmse_per_seed = [np.sqrt(np.asarray(h["train_loss"])) for h in histories]
    val_rmse_per_seed = [np.sqrt(np.asarray(h["val_loss"])) for h in histories]
    lengths = {len(a) for a in train_rmse_per_seed}
    if len(lengths) > 1:
        print(f"[run_seed_ensemble] seeds trained for different epoch counts "
              f"{sorted(len(a) for a in train_rmse_per_seed)} (early stopping?) -- truncating the "
              f"training-curve aggregate to the shortest, {min(lengths)} epochs, for a fair per-epoch mean/std.")
    min_len = min(lengths)
    train_rmse_stack = np.stack([a[:min_len] for a in train_rmse_per_seed])
    val_rmse_stack = np.stack([a[:min_len] for a in val_rmse_per_seed])
    train_rmse_mean, train_rmse_std = train_rmse_stack.mean(axis=0), train_rmse_stack.std(axis=0)
    val_rmse_mean, val_rmse_std = val_rmse_stack.mean(axis=0), val_rmse_stack.std(axis=0)

    history_path = ensemble_dir / "history_ensemble_metrics.npz"
    np.savez(history_path, seeds=np.array(args.seeds), pretrain_epochs=base_cfg.training.epochs,
             train_rmse_mean=train_rmse_mean, train_rmse_std=train_rmse_std,
             val_rmse_mean=val_rmse_mean, val_rmse_std=val_rmse_std)
    print(f"saved training-curve ensemble metrics to {history_path}")

    plot_history_mean_std(
        train_rmse_mean, train_rmse_std, val_rmse_mean, val_rmse_std,
        str(ensemble_dir / f"{base_run_name}_ensemble_loss.png"),
        run_name=base_run_name, pretrain_epochs=base_cfg.training.epochs,
    )
    plot_history_mean_std(
        train_rmse_mean, train_rmse_std, val_rmse_mean, val_rmse_std,
        str(ensemble_dir / f"{base_run_name}_ensemble_loss_log.png"),
        run_name=base_run_name, log_scale=True, pretrain_epochs=base_cfg.training.epochs,
    )
    print(f"saved training-curve ensemble plots to {ensemble_dir}")

    # Evaluation: mean/std of model_rmse/model_acc per target, across
    # seeds a single reference value from the first seed, not an aggregate. 
    for target in base_cfg.inference.targets:
        results = eval_results_by_target[target.name]

        lead_hours = results[0]["lead_hours"]
        model_rmse_stack = np.stack([r["model_rmse"] for r in results])
        model_rmse_mean, model_rmse_std = model_rmse_stack.mean(axis=0), model_rmse_stack.std(axis=0)
        persistence_rmse = results[0]["persistence_rmse"]

        metrics_to_save = {
            "seeds": np.array(args.seeds), "lead_hours": lead_hours,
            "model_rmse_mean": model_rmse_mean, "model_rmse_std": model_rmse_std,
            "persistence_rmse": persistence_rmse,
        }

        has_climatology = all("model_acc" in r for r in results)
        if has_climatology:
            model_acc_stack = np.stack([r["model_acc"] for r in results])
            metrics_to_save["model_acc_mean"] = model_acc_stack.mean(axis=0)
            metrics_to_save["model_acc_std"] = model_acc_stack.std(axis=0)
            metrics_to_save["climatology_rmse"] = results[0]["climatology_rmse"]

        metrics_path = ensemble_dir / f"{target.name}_ensemble_eval_metrics.npz"
        np.savez(metrics_path, **metrics_to_save)
        print(f"[{target.name}] saved eval ensemble metrics to {metrics_path}")

        rmse_plot_result = {"lead_hours": lead_hours, "model_rmse": model_rmse_mean,
                             "model_rmse_std": model_rmse_std, "persistence_rmse": persistence_rmse}
        if has_climatology:
            rmse_plot_result["climatology_rmse"] = metrics_to_save["climatology_rmse"]
        scorecard_path = ensemble_dir / f"{target.name}_ensemble_rmse_vs_lead_time.png"
        plot_rmse_vs_lead_time(rmse_plot_result, base_cfg.data.channels, HEADLINE_CHANNELS, str(scorecard_path),
                                title=f"{target.name} -- RMSE vs lead time (mean ± std, {len(args.seeds)} seeds)")
        print(f"[{target.name}] saved RMSE ensemble scorecard to {scorecard_path}")

        if has_climatology:
            acc_plot_result = {"lead_hours": lead_hours, "model_acc": metrics_to_save["model_acc_mean"],
                                "model_acc_std": metrics_to_save["model_acc_std"]}
            acc_path = ensemble_dir / f"{target.name}_ensemble_acc_vs_lead_time.png"
            plot_acc_vs_lead_time(acc_plot_result, base_cfg.data.channels, HEADLINE_CHANNELS, str(acc_path),
                                   title=f"{target.name} -- ACC vs lead time (mean ± std, {len(args.seeds)} seeds)")
            print(f"[{target.name}] saved ACC ensemble scorecard to {acc_path}")

    print(f"\nDone -- {len(args.seeds)} seeds ({args.seeds}), ensemble outputs in {ensemble_dir}")


if __name__ == "__main__":
    main()
