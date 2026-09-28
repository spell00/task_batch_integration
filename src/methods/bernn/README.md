# BERNN OpenProblems adapter

This component exposes BERNN as an embedding method for the OpenProblems
batch-integration task.

Input:
- AnnData with log-CP10K expression
- obs["batch"] for domain labels
- obs["cell_type"] for supervised biological labels
- var["hvg_score"] for optional HVG selection

Output:
- obsm["X_emb"] containing the BERNN latent representation

The component currently targets the BERNN branch
feature/openproblems-single-cell. Pin this to a commit before submitting the
OpenProblems pull request.
