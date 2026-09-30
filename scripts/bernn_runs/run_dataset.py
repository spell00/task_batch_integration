#!/usr/bin/env python3
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from bernn import TrainAEClassifierHoldout
from bernn.config.training_config import TrainingConfig

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

    print(f"[bernn-openproblems] reading {args.input}", flush=True)
    adata = ad.read_h5ad(args.input)
    if "normalized" in adata.layers:
        mat = adata.layers["normalized"]
    else:
        mat = adata.X

    if args.n_hvg:
        if "hvg_score" not in adata.var:
            raise KeyError("Expected adata.var['hvg_score']")
        idx = adata.var["hvg_score"].to_numpy().argsort()[::-1][:args.n_hvg]
        adata = adata[:, idx].copy()
        mat = adata.layers["normalized"] if "normalized" in adata.layers else adata.X

    matrix = mat.toarray() if sparse.issparse(mat) else np.asarray(mat)
    matrix = np.asarray(matrix, dtype=np.float32)
    X = pd.DataFrame(matrix, index=adata.obs_names, columns=adata.var_names)
    y = adata.obs["cell_type"].astype(str).to_numpy()
    batches = adata.obs["batch"].astype(str).to_numpy()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[bernn-openproblems] cells={len(X)} features={X.shape[1]} batches={len(np.unique(batches))} classes={len(np.unique(y))} device={device}", flush=True)

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
    trainer = TrainAEClassifierHoldout(
        config=cfg,
        pools=False,
        keep_models=False,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_mlflow=False,
        log_dvclive=False,
    )
    trainer.fit(X, y, groups_train=batches, internal_validation=False)

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
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output, compression="gzip")
    print(f"[bernn-openproblems] wrote {args.output} embedding={embedding.shape}", flush=True)

if __name__ == "__main__":
    main()
