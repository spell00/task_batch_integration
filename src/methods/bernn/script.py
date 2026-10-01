import sys
import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from bernn import TrainAEClassifierHoldout
from bernn.config.training_config import TrainingConfig
from bernn.dl.models.pytorch.utils.utils import LogConfusionMatrix as _BERNNLogConfusionMatrix

# BERNN 1.0.6 bugfix: LogConfusionMatrix.add computes int(1000 / batch_size).
# With our reference batch_size=1024 that becomes 0 and crashes after training.
# Confusion-matrix accumulation is not used by OpenProblems here, but preserve
# normal behavior for batch sizes <=1000.
_bernn_cm_add = _BERNNLogConfusionMatrix.add
def _safe_bernn_cm_add(self, lists):
    train_preds = lists.get("train", {}).get("preds", [])
    if train_preds and int(1000 / max(1, train_preds[0].shape[0])) < 1:
        return
    return _bernn_cm_add(self, lists)
_BERNNLogConfusionMatrix.add = _safe_bernn_cm_add

## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
    "n_hvg": 2000,
    "n_epochs": 200,
    "warmup": 50,
    "batch_size": 1024,
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
X = pd.DataFrame(matrix, index=adata.obs_names, columns=adata.var_names)
y = adata.obs["cell_type"].astype(str).to_numpy()
batches = adata.obs["batch"].astype(str).to_numpy()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(
    f"Run BERNN 1.0.6 on {X.shape[0]} cells x {X.shape[1]} genes "
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
    num_workers=0,
    optimize_hyperparams=False,
    groupkfold=False,
)

trainer = TrainAEClassifierHoldout(
    config=args,
    groupkfold=False,
    pools=False,
    keep_models=False,
    log_inputs=False,
    log_plots=False,
    log_tb=False,
    log_mlflow=False,
    log_dvclive=False,
)

# OpenProblems supplies one dataset to the method. BERNN therefore trains on
# every cell; its validation/test loaders are train-resubstitution monitors
# when internal_validation=False and do not remove rows from model fitting.
best_params = {
    "lr": 1e-3,
    "dropout": 0.0,
    "wd": 1e-5,
    "margin": 1.0,
    "smoothing": 0.1,
    "scaler": "standard",
    "gamma": 0.1,
    "beta": 0.0,
    "zeta": 0.0,
    "nu": 1.0,
    "thres": 0.0,
    "prune_threshold": 0.0,
    "warmup": int(par["warmup"]),
    "l1": 0.0,
    "reg_entropy": 0.0,
    "layer1": int(par["layer1"]),
    "layer2": max(16, int(par["layer1"]) // 2),
    "n_layers": int(par["n_layers"]),
}

trainer.fit(
    X,
    y,
    groups_train=batches,
    params=best_params,
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
