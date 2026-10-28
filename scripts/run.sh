#!/bin/bash
cd "${WORK_DIR:-/opt/dlami/nvme/work}"
source /opt/pytorch/bin/activate
export HF_HOME=/opt/dlami/nvme/hf PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python "$@"
