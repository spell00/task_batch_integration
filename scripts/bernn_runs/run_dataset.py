#!/usr/bin/env python3
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
import wandb

from bernn import TrainAEClassifierHoldout
from bernn.config.training_config import TrainingConfig
import bernn.dl.train.train_ae_classifier_holdout as _bernn_holdout


# BERNN 1.0.6 compatibility: TrainAEClassifierHoldout calls TensorboardLoggingAE
# without the newer required `zinb` argument. Keep the patch local to this runner.
_OriginalTensorboardLoggingAE = _bernn_holdout.TensorboardLoggingAE


class _TensorboardLoggingAECompat(_OriginalTensorboardLoggingAE):
    def __init__(self, hparams_filepath, params, variational, tw, tl, dloss,
                 pseudo, train_after_warmup, berm, args, zinb=False):
        super().__init__(
            hparams_filepath,
            params,
            variational=variational,
            zinb=zinb,
            tw=tw,
            tl=tl,
            dloss=dloss,
            pseudo=pseudo,
            train_after_warmup=train_after_warmup,
            berm=berm,
            args=args,
        )


_bernn_holdout.TensorboardLoggingAE = _TensorboardLoggingAECompat


REPO_ROOT = Path(__file__).resolve().parents[2]
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "adlab")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "BERNN_OpenProblems_batch_integration")


def git_value(*args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def log_training_code(run):
    artifact = wandb.Artifact(
        name=f"training-code-{run.id}",
        type="code",
        metadata={
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
            "bernn_version": importlib.metadata.version("bernn"),
            "torch_version": torch.__version__,
        },
    )

    adapter_dir = REPO_ROOT / "src" / "methods" / "bernn"
    runner_dir = REPO_ROOT / "scripts" / "bernn_runs"
    if adapter_dir.exists():
        artifact.add_dir(str(adapter_dir), name="openproblems_adapter")
    if runner_dir.exists():
        artifact.add_dir(str(runner_dir), name="training_runner")

    bernn_dir = Path(importlib.util.find_spec("bernn").origin).parent
    artifact.add_dir(str(bernn_dir), name="bernn_pypi_1.0.6")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "git.txt").write_text(
            "\n".join(
                [
                    f"commit={git_value('rev-parse', 'HEAD')}",
                    f"branch={git_value('rev-parse', '--abbrev-ref', 'HEAD')}",
                    f"remote={git_value('remote', 'get-url', 'origin')}",
                ]
            )
            + "\n"
        )
        versions = {}
        for package in [
            "bernn",
            "torch",
            "torchvision",
            "tensorflow",
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "anndata",
            "wandb",
        ]:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        (td / "versions.json").write_text(json.dumps(versions, indent=2, sort_keys=True))
        diff = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "diff", "HEAD"],
            text=True,
            capture_output=True,
        ).stdout
        (td / "git_diff.patch").write_text(diff)
        artifact.add_file(str(td / "git.txt"), name="provenance/git.txt")
        artifact.add_file(str(td / "versions.json"), name="provenance/versions.json")
        artifact.add_file(str(td / "git_diff.patch"), name="provenance/git_diff.patch")

    run.log_artifact(artifact)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--layer1", type=int, default=512)
    p.add_argument("--dloss", default="inverseTriplet")
    args = p.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    dataset_slug = input_path.parents[1].name

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"openproblems-{dataset_slug}-bernn-1.0.6",
        group="openproblems-batch-integration-bernn-1.0.6",
        job_type="train_integrate",
        tags=["openproblems", "batch_integration", "bernn", "bernn-1.0.6", dataset_slug],
        sync_tensorboard=True,
        config={
            "dataset_slug": dataset_slug,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "n_hvg": args.n_hvg,
            "n_epochs": args.epochs,
            "warmup": args.warmup,
            "batch_size": args.batch_size,
            "n_layers": args.n_layers,
            "layer1": args.layer1,
            "dloss": args.dloss,
            "scaler": "standard",
            "bernn_version": importlib.metadata.version("bernn"),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        },
    )
    log_training_code(run)

    started = time.time()
    try:
        print(f"[bernn-openproblems] W&B {run.url}", flush=True)
        print(f"[bernn-openproblems] reading {input_path}", flush=True)
        adata = ad.read_h5ad(input_path)
        if "normalized" in adata.layers:
            mat = adata.layers["normalized"]
        else:
            mat = adata.X

        if args.n_hvg:
            if "hvg_score" not in adata.var:
                raise KeyError("Expected adata.var['hvg_score']")
            idx = adata.var["hvg_score"].to_numpy().argsort()[::-1][: args.n_hvg]
            adata = adata[:, idx].copy()
            mat = adata.layers["normalized"] if "normalized" in adata.layers else adata.X

        matrix = mat.toarray() if sparse.issparse(mat) else np.asarray(mat)
        matrix = np.asarray(matrix, dtype=np.float32)
        X = pd.DataFrame(matrix, index=adata.obs_names, columns=adata.var_names)
        y = adata.obs["cell_type"].astype(str).to_numpy()
        batches = adata.obs["batch"].astype(str).to_numpy()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        gpu_memory_gb = (
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if torch.cuda.is_available()
            else None
        )

        run.config.update(
            {
                "dataset_id": str(adata.uns["dataset_id"]),
                "normalization_id": str(adata.uns["normalization_id"]),
                "n_cells": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "n_batches": int(np.unique(batches).size),
                "n_classes": int(np.unique(y).size),
                "input_file_bytes": int(input_path.stat().st_size),
                "device": device,
                "gpu_name": gpu_name,
                "gpu_memory_gb": gpu_memory_gb,
            },
            allow_val_change=True,
        )
        wandb.log(
            {
                "data/n_cells": X.shape[0],
                "data/n_features": X.shape[1],
                "data/n_batches": np.unique(batches).size,
                "data/n_classes": np.unique(y).size,
            }
        )

        print(
            f"[bernn-openproblems] cells={len(X)} features={X.shape[1]} "
            f"batches={len(np.unique(batches))} classes={len(np.unique(y))} "
            f"device={device}",
            flush=True,
        )

        cfg = TrainingConfig(
            n_epochs=args.epochs,
            warmup=args.warmup,
            bs=args.batch_size,
            n_layers=args.n_layers,
            layer1=args.layer1,
            dloss=args.dloss,
            device=device,
            n_repeats=1,
            kan=False,
            log1p=False,
            scaler="standard",
            num_workers=0,
        )
        # Legacy TensorBoard logger fields expected by BERNN 1.0.6.
        # These are logging metadata only; they do not alter optimization.
        _tb_defaults = {
            "berm": "none",
            "rec_loss": "mse",
            "dataset": dataset_slug,
            "strategy": "openproblems_full_data",
            "bad_batches": "",
            "remove_zeros": False,
            "use_mapping": getattr(cfg, "use_mapping", False),
            "csv_file": str(input_path),
            "tied_weights": getattr(cfg, "tied_weights", False),
        }
        for _name, _value in _tb_defaults.items():
            if not hasattr(cfg, _name):
                setattr(cfg, _name, _value)

        trainer = TrainAEClassifierHoldout(
            config=cfg,
            pools=False,
            keep_models=True,
            log_inputs=False,
            log_plots=False,
            log_tb=True,
            log_mlflow=False,
            log_dvclive=False,
        )

        # Mirror BERNN's actual _train(None) defaults and make_params() rules so
        # provenance contains the effective hyperparameters rather than raw
        # TrainingConfig fields that may be absent/None.
        effective_hparams = {
            "lr": getattr(trainer.args, "lr", 1e-3),
            "dropout": getattr(trainer.args, "dropout", 0.1),
            "wd": getattr(trainer.args, "wd", 1e-5),
            "margin": getattr(trainer.args, "margin", 1.0),
            "smoothing": getattr(trainer.args, "smoothing", 0.1),
            "scaler": getattr(trainer.args, "scaler", "standard"),
            "gamma": getattr(trainer.args, "gamma", 0.1),
            "beta": getattr(trainer.args, "beta", 0.1),
            "zeta": getattr(trainer.args, "zeta", 0.0),
            "nu": getattr(trainer.args, "nu", 1.0),
            "thres": getattr(trainer.args, "thres", 0.0),
            "prune_threshold": getattr(trainer.args, "prune_threshold", 0.0),
            "warmup": getattr(trainer.args, "warmup", 100),
            "l1": getattr(trainer.args, "l1", 0.0),
            "reg_entropy": getattr(trainer.args, "reg_entropy", 0.0),
            "n_epochs": getattr(trainer.args, "n_epochs", args.epochs),
            "bs": getattr(trainer.args, "bs", args.batch_size),
            "n_layers": max(1, int(getattr(trainer.args, "n_layers", 1))),
            "dloss": getattr(trainer.args, "dloss", args.dloss),
            "kan": bool(getattr(trainer.args, "kan", False)),
            "log1p": bool(getattr(trainer.args, "log1p", False)),
            "num_workers": int(getattr(trainer.args, "num_workers", 0)),
        }
        effective_hparams.update(trainer.get_default_layer_params())
        if effective_hparams["dloss"] not in [
            "revTriplet", "revDANN", "DANN", "inverseTriplet", "normae"
        ]:
            effective_hparams["gamma"] = 0
        if not bool(getattr(trainer.args, "variational", False)):
            effective_hparams["beta"] = 0
        if not bool(getattr(trainer.args, "use_l1", False)):
            effective_hparams["l1"] = 0
            effective_hparams["reg_entropy"] = 0
        if not bool(getattr(trainer.args, "prune_network", False)):
            effective_hparams["prune_threshold"] = 0
        if float(effective_hparams["prune_threshold"]) > 0:
            effective_hparams["dropout"] = 0
        effective_hparams["thres"] = 0

        run.config.update(
            {f"bernn/{key}": value for key, value in effective_hparams.items()},
            allow_val_change=True,
        )
        # Train on every OpenProblems cell, but avoid BERNN's no-holdout mode
        # duplicating the entire training matrix into full-size valid/test monitors.
        # A fixed representative monitor subset is sufficient for live diagnostics
        # and does not remove any rows from training.
        rng = np.random.default_rng(42)
        monitor_n = min(4096, len(X))
        monitor_idx = np.sort(rng.choice(len(X), size=monitor_n, replace=False))
        X_monitor = X.iloc[monitor_idx].copy()
        y_monitor = y[monitor_idx]
        batches_monitor = batches[monitor_idx]
        run.config.update({"monitor_n": int(monitor_n)}, allow_val_change=True)

        trainer.fit(
            X,
            y,
            groups_train=batches,
            X_valid=X_monitor,
            y_valid=y_monitor,
            groups_valid=batches_monitor,
            X_test=X_monitor,
            y_test=y_monitor,
            groups_test=batches_monitor,
            internal_validation=False,
        )

        # Persist the exact restored best model and all inference-critical state.
        model_dir = output_path.parent / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_dir / "best_model_state.pt"
        config_path = model_dir / "training_config.json"

        scaler_state = {}
        for attr in ["mean_", "scale_", "var_", "n_features_in_"]:
            value = getattr(trainer.scaler, attr, None)
            if value is not None:
                if isinstance(value, np.ndarray):
                    scaler_state[attr] = value.tolist()
                elif isinstance(value, np.generic):
                    scaler_state[attr] = value.item()
                else:
                    scaler_state[attr] = value

        if getattr(trainer, "_label_encoder", None) is not None:
            class_names = trainer._label_encoder.classes_.tolist()
        else:
            class_names = [str(x) for x in np.unique(y)]

        batch_map = {
            str(key): int(value)
            for key, value in getattr(trainer, "batch_map_", {}).items()
        }
        checkpoint = {
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in trainer.ae.state_dict().items()
            },
            "effective_hparams": effective_hparams,
            "scaler_state": scaler_state,
            "feature_names": [str(x) for x in X.columns],
            "class_names": class_names,
            "batch_map": batch_map,
            "dataset_slug": dataset_slug,
            "dataset_id": str(adata.uns["dataset_id"]),
            "normalization_id": str(adata.uns["normalization_id"]),
            "git_commit": git_value("rev-parse", "HEAD"),
            "bernn_version": importlib.metadata.version("bernn"),
            "wandb_run_id": run.id,
        }
        torch.save(checkpoint, checkpoint_path)

        config_payload = {
            key: value for key, value in checkpoint.items()
            if key != "model_state_dict"
        }
        config_path.write_text(
            json.dumps(config_payload, indent=2, sort_keys=True)
        )

        model_artifact = wandb.Artifact(
            name=f"bernn-openproblems-{dataset_slug}-model-{run.id}",
            type="model",
            metadata={
                "dataset_slug": dataset_slug,
                "bernn_version": importlib.metadata.version("bernn"),
                "git_commit": git_value("rev-parse", "HEAD"),
                "best_valid_mcc": float(getattr(trainer, "best_mcc", np.nan)),
            },
        )
        model_artifact.add_file(str(checkpoint_path), name="best_model_state.pt")
        model_artifact.add_file(str(config_path), name="training_config.json")
        if Path(trainer.complete_log_path).exists():
            model_artifact.add_dir(
                trainer.complete_log_path,
                name="tensorboard_and_bernn_checkpoints",
            )
        run.log_artifact(model_artifact)
        run.summary["best_valid_mcc"] = float(
            getattr(trainer, "best_mcc", np.nan)
        )
        run.summary["model_checkpoint"] = str(checkpoint_path)

        X_scaled = trainer._prepare_prediction_matrix(
            X, groups_test=batches, return_batch_ids=False
        )
        embedding = np.asarray(trainer.transform(X_scaled), dtype=np.float32)
        if embedding.shape[0] != adata.n_obs or not np.isfinite(embedding).all():
            raise RuntimeError(f"Invalid embedding shape/values: {embedding.shape}")

        out = ad.AnnData(
            obs=adata.obs[[]],
            var=pd.DataFrame(index=[]),
            obsm={"X_emb": embedding},
            uns={
                "dataset_id": adata.uns["dataset_id"],
                "normalization_id": adata.uns["normalization_id"],
                "method_id": "bernn",
            },
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.write_h5ad(output_path, compression="gzip")

        elapsed = time.time() - started
        wandb.log(
            {
                "result/embedding_dim": int(embedding.shape[1]),
                "result/output_file_bytes": int(output_path.stat().st_size),
                "runtime/seconds": elapsed,
            }
        )
        run.summary["status"] = "completed"
        run.summary["embedding_shape"] = list(embedding.shape)

        output_artifact = wandb.Artifact(
            name=f"bernn-openproblems-{dataset_slug}-{run.id}",
            type="integrated_h5ad",
            metadata={
                "dataset_id": str(adata.uns["dataset_id"]),
                "normalization_id": str(adata.uns["normalization_id"]),
                "embedding_shape": list(embedding.shape),
            },
        )
        output_artifact.add_file(str(output_path), name="integrated.h5ad")
        run.log_artifact(output_artifact)

        print(
            f"[bernn-openproblems] wrote {output_path} embedding={embedding.shape}",
            flush=True,
        )
    except Exception:
        run.summary["status"] = "failed"
        raise
    finally:
        run.finish()


if __name__ == "__main__":
    main()
