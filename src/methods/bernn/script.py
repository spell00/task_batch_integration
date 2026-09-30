import sys
import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from bernn import TrainAEClassifierHoldout
from bernn.config.training_config import TrainingConfig

## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
    "n_hvg": 2000,
    "n_epochs": 200,
    "warmup": 50,
    "batch_size": 128,
    "n_layers": 2,
    "layer1": 512,
    "dloss": "inverseTriplet",
}
meta = {"name": "bernn"}
## VIASH END

sys.path.append(meta["resources_dir"])
from read_anndata_partial import read_anndata

print("Read input", flush=True)
adata = read_anndata(
    par["input"],
    X="layers/normalized",
    obs="obs",
    var="var",
    uns="uns",
)

if par["n_hvg"]:
    print(f"Select top {par['n_hvg']} highly variable genes", flush=True)
    idx = adata.var["hvg_score"].to_numpy().argsort()[::-1][: par["n_hvg"]]
    adata = adata[:, idx].copy()

print("Prepare BERNN matrix", flush=True)
matrix = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
matrix = np.asarray(matrix, dtype=np.float32)
X = pd.DataFrame(matrix, index=adata.obs_names, columns=adata.var_names)
y = adata.obs["cell_type"].astype(str).to_numpy()
batches = adata.obs["batch"].astype(str).to_numpy()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(
    f"Run BERNN on {X.shape[0]} cells x {X.shape[1]} genes "
    f"using {device}",
    flush=True,
)

args = TrainingConfig(
    n_epochs=int(par["n_epochs"]),
    warmup=int(par["warmup"]),
    bs=int(par["batch_size"]),
    n_layers=int(par["n_layers"]),
    layer1=int(par["layer1"]),
    dloss=str(par["dloss"]),
    device=device,
    n_repeats=1,
    kan=False,
    log1p=False,
    scaler="standard",
)

trainer = TrainAEClassifierHoldout(
    config=args,
    pools=False,
    keep_models=False,
    log_inputs=False,
    log_plots=False,
    log_tb=False,
    log_mlflow=False,
    log_dvclive=False,
)

# Train on every OpenProblems cell. BERNN's legacy no-holdout path copies the
# full training matrix into validation/test monitors, so use a deterministic
# 4,096-cell monitor subset instead. This changes only monitoring/checkpoint
# diagnostics; all rows remain in the training set.
rng = np.random.default_rng(42)
monitor_n = min(4096, len(X))
monitor_idx = np.sort(rng.choice(len(X), size=monitor_n, replace=False))
X_monitor = X.iloc[monitor_idx].copy()
y_monitor = y[monitor_idx]
batches_monitor = batches[monitor_idx]

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

print("Extract BERNN embedding", flush=True)
X_scaled = trainer._prepare_prediction_matrix(
    X,
    groups_test=batches,
    return_batch_ids=False,
)
embedding = np.asarray(trainer.transform(X_scaled), dtype=np.float32)

print("Store output", flush=True)
output = ad.AnnData(
    obs=adata.obs[[]],
    var=pd.DataFrame(index=[]),
    obsm={"X_emb": embedding},
    uns={
        "dataset_id": adata.uns["dataset_id"],
        "normalization_id": adata.uns["normalization_id"],
        "method_id": meta["name"],
    },
)
output.write_h5ad(par["output"], compression="gzip")