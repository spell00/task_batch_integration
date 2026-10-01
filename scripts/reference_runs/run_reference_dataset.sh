#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/simonp/task_batch_integration_bernn
DS=${1:?dataset required}
GPU=${2:?gpu required}
PROFILE=reference_default_gkf1_openproblems_exact
STAMP=$(date -u +%Y-%m-%d_%H-%M-%S)
OUT="$ROOT/resources/results/bernn_${PROFILE}_${DS}_${STAMP}"
META="$ROOT/reference_runs/${DS}_${STAMP}.json"
PARAMS="$ROOT/reference_runs/${DS}_${STAMP}.yaml"
mkdir -p "$OUT" "$ROOT/reference_runs" "$ROOT/work/reference_${DS}"
cat > "$PARAMS" <<EOF
input_states: resources_full/task_batch_integration/datasets/cellxgene_census/${DS}/log_cp10k/state.yaml
rename_keys: 'input_dataset:output_dataset;input_solution:output_solution'
output_state: state.yaml
publish_dir: "$OUT"
settings: '{"methods_include": ["bernn"]}'
EOF
python3 - "$DS" "$GPU" "$STAMP" "$OUT" "$META" <<'PY'
import json, subprocess, sys
from pathlib import Path
ds,gpu,stamp,out,meta=sys.argv[1:]
root=Path('/home/simonp/task_batch_integration_bernn')
def sh(*a): return subprocess.check_output(a,text=True,cwd=root).strip()
record={
 'dataset':ds,'profile':'reference_default_gkf1_openproblems_exact','status':'running',
 'started_utc':stamp,'gpu':int(gpu),'bernn_version':'git:2cb042a36315ec589c4ace2ba923068e652eab2a',
 'repo_commit':sh('git','rev-parse','HEAD'),'repo_branch':sh('git','branch','--show-current'),
 'adapter_sha256':sh('sha256sum','src/methods/bernn/script.py').split()[0],
 'config_sha256':sh('sha256sum','src/methods/bernn/config.vsh.yaml').split()[0],
 'publish_dir':out,
 'hparams':{'n_hvg':2000,'n_epochs':200,'warmup':50,'batch_size':1024,
            'n_layers':2,'layer1':512,'layer2':256,'dloss':'inverseTriplet',
            'lr':0.001,'weight_decay':1e-5,'margin':1.0,'smoothing':0.1,
            'gamma':0.1,'scaler':'standard','groupkfold':True,'internal_validation':False,
            'fit_mode':'openproblems_full_dataset','model_selection':'final_epoch','optimization_metric':'openproblems_only'},
 'hparams_source':'reference/default; replace later with dataset-optimized result'
}
Path(meta).write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
PY
export JAVA_HOME=/home/simonp/anaconda3/envs/openproblems-tools
export PATH="$JAVA_HOME/bin:$PATH"
export LANG=C.UTF-8 LC_ALL=C.UTF-8
export NXF_ANSI_LOG=false NXF_SYNTAX_PARSER=v1
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"
set +e
nextflow -log "$OUT/nextflow.log" run . \
  -main-script target/nextflow/workflows/run_benchmark/main.nf \
  -profile docker -entry auto \
  -c common/nextflow_helpers/labels_ci.config \
  -c config/local_gpu${GPU}_gkf1.config \
  -work-dir "$ROOT/work/reference_${DS}" \
  -params-file "$PARAMS" 2>&1 | tee "$OUT/nextflow_console.log"
RC=${PIPESTATUS[0]}
set -e
python3 - "$META" "$RC" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
p=Path(sys.argv[1]); rc=int(sys.argv[2]); x=json.loads(p.read_text())
x['finished_utc']=datetime.now(timezone.utc).isoformat()
x['exit_code']=rc
x['status']='completed' if rc==0 else 'failed'
p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
PY
exit "$RC"
