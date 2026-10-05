"""Probe a ClearML queue's worker with the training docker image: GPUs, driver, torch, cores, disk, ultralytics.

Why: the agents have no SSH, and Ultralytics will pull its own torch wheel unless the image already
satisfies it. The driver version decides which CUDA wheels can run at all.

Run on the box:  uv run python scripts/probe_agent.py --queue multi-gpu
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess

from clearml import Task

DOCKER_IMAGE = "862264091922.dkr.ecr.eu-central-1.amazonaws.com/model-training:latest"
DOCKER_ARGS = "--gpus all --network host --shm-size=16g -e AWS_PROFILE=clearml-s3 -v /home/ec2-user/.aws:/root/.aws:ro"

ap = argparse.ArgumentParser()
ap.add_argument("--queue", default="multi-gpu")
ap.add_argument(
    "--list-cache", action="store_true", help="also list what fills /clearml_agent_cache"
)
args = ap.parse_args()

task = Task.init(
    project_name="thesis_wp1_benchmark",
    task_name=f"PROBE {args.queue} worker hardware",
    task_type=Task.TaskTypes.custom,
    reuse_last_task_id=False,
    output_uri=False,  # the agent role cannot DeleteObject; Task.init's default probe would kill the task
)
task.set_base_docker(DOCKER_IMAGE, docker_arguments=DOCKER_ARGS)
task.execute_remotely(queue_name=args.queue, exit_process=True)


def sh(cmd: str) -> str:
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=120, check=False
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"failed: {e}"


print("=== nvidia-smi ===")
print(sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"))
print("driver/cuda header:", sh("nvidia-smi | sed -n 3p"))
print("=== cpu / mem ===")
print("nproc:", os.cpu_count(), "| mem GB:", sh("free -g | awk '/Mem/{print $2}'"))
print("=== python / torch in the image ===")
print(
    sh(
        "python -c \"import sys, torch; print(sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())\""
    )
)
print(
    "ultralytics in image:",
    sh('python -c "import ultralytics; print(ultralytics.__version__)" 2>&1 | tail -1'),
)
print(
    "pip torch wheels:",
    sh("pip list 2>/dev/null | grep -i -E '^torch|^ultralytics|^opencv|^numpy'"),
)
print("=== disk ===")
for p in ("/", "/root", "/clearml_agent_cache", "/tmp"):
    try:
        u = shutil.disk_usage(p)
        print(f"  {p:22s} free {u.free / 1e9:7.1f} GB of {u.total / 1e9:7.1f} GB")
    except Exception as e:  # noqa: BLE001
        print(f"  {p:22s} {e}")
if args.list_cache:
    print("=== /clearml_agent_cache: biggest entries ===")
    print(sh("du -sh /clearml_agent_cache/* 2>/dev/null | sort -rh | head -8"))
    print(sh("du -sh /clearml_agent_cache/storage_manager/* 2>/dev/null | sort -rh | head -8"))
    print("--- datasets by size (GB), with last-modified ---")
    print(
        sh(
            'cd /clearml_agent_cache/storage_manager/datasets 2>/dev/null && du -s --block-size=1G * 2>/dev/null | sort -rn | head -40 | while read sz d; do echo "$sz GB  $(stat -c %y "$d" | cut -c1-16)  $d"; done'
        )
    )
    print("--- count ---")
    print(sh("ls /clearml_agent_cache/storage_manager/datasets 2>/dev/null | wc -l"))
    print("--- other big dirs ---")
    print(sh("du -xsh /root/.cache /root/.clearml /opt /tmp 2>/dev/null | sort -rh"))
print("=== clearml.conf of the worker (hosts only) ===")
print(
    sh(
        "grep -h -E 'api_server|web_server|files_server' /root/clearml.conf ~/clearml.conf 2>/dev/null | head -6"
    )
)
