import sys

import anndata as ad
import torch

from bernn.single_cell import fit_transform_anndata


## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
    "n_hvg": 2000,
    "n_epochs": 3,
    "warmup": 1,
    "batch_size": 64,
    "n_layers": 2,
    "layer1": 128,
    "dloss": "inverseTriplet",
}
meta = {
    "name": "bernn",
    "resources_dir": "src/utils",
}
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(
    f"Run BERNN on {adata.n_obs} cells x {adata.n_vars} genes "
    f"using device={device}",
    flush=True,
)
embedding = fit_transform_anndata(
    adata,
    layer="X",
    batch_key="batch",
    label_key="cell_type",
    n_hvg=par["n_hvg"],
    dloss=par["dloss"],
    n_epochs=par["n_epochs"],
    warmup=par["warmup"],
    batch_size=par["batch_size"],
    n_layers=par["n_layers"],
    layer1=par["layer1"],
    scaler="standard",
    device=device,
)

print("Store outputs", flush=True)
output = ad.AnnData(
    obs=adata.obs[[]],
    obsm={"X_emb": embedding},
    uns={
        "dataset_id": adata.uns["dataset_id"],
        "normalization_id": adata.uns["normalization_id"],
        "method_id": meta["name"],
    },
)

print("Write output to file", flush=True)
output.write_h5ad(par["output"], compression="gzip")
