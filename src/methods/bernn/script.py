import sys
import anndata as ad
import numpy as np
import pandas as pd
import torch

from bernn import fit_openproblems_once

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

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(
    f"Run BERNN OpenProblems full-dataset fit on {adata.shape[0]} cells x {adata.shape[1]} genes "
    f"using {device}; no train/valid/test holdout and final epoch output",
    flush=True,
)

embedding, trainer = fit_openproblems_once(
    adata,
    layer="X",
    batch_key="batch",
    label_key="cell_type",
    n_hvg=int(par["n_hvg"]),
    dloss=str(par["dloss"]),
    n_epochs=int(par["n_epochs"]),
    warmup=int(par["warmup"]),
    batch_size=int(par["batch_size"]),
    n_layers=int(par["n_layers"]),
    layer1=int(par["layer1"]),
    scaler="standard",
    learning_rate=1e-3,
    weight_decay=1e-5,
    dropout=0.0,
    margin=1.0,
    smoothing=0.1,
    nu=1.0,
    num_workers=0,
    device=device,
    return_trainer=True,
)

print(
    f"Extracted final-epoch BERNN embedding after epoch {trainer._openproblems_last_epoch}",
    flush=True,
)
print("Store output", flush=True)
output = ad.AnnData(
    obs=adata.obs[[]],
    var=pd.DataFrame(index=[]),
    obsm={"X_emb": np.asarray(embedding, dtype=np.float32)},
    uns={
        "dataset_id": adata.uns["dataset_id"],
        "normalization_id": adata.uns["normalization_id"],
        "method_id": meta["name"],
    },
)
output.write_h5ad(par["output"], compression="gzip")
