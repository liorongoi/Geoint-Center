# Validate two physical workers

Use a disposable coordinator workspace with synthetic or public test imagery. The Mac can host the coordinator and Worker A. Enroll Worker B on a separate physical computer. Use the same application code and compatible pinned Python packages on both workers. The scheduler checks the code and runtime fingerprint before starting a task. Each host needs at least 12 GiB RAM to enroll and at least a 5 GiB worker budget for synchronized U-Net. GPU leases require a supported device and sufficient VRAM.

## Enroll

1. Have your internal network administrator provide a private HTTPS route to the disposable coordinator. The native web backend listens on loopback, so a second host needs an approved TLS listener or gateway. Restrict the route to the worker enrollment and task endpoints and the approved worker machines. Use a certificate whose subject alternative name matches the worker-facing hostname. Keep the private CA or certificate PEM on the workers.
2. In the coordinator, open **Data & results → Services & workers → Worker pool → Add an approved worker**. Unlock the panel with the coordinator's local services key. Use a distinct name and the worker-facing HTTPS URL. Download a separate one-time invitation for each host and use it within 10 minutes. Keep each invitation inside the approved network.
3. Run the matching offline Python runtime on each machine. In this example, use the local absolute paths to the provisioned runtime, downloaded invitation and CA certificate. The cache must be a dedicated empty directory outside supplier source folders.

   ```sh
   /path/to/python3 /path/to/backend/pool_worker.py --enroll \
     --enrollment-file /private/worker-enroll.json \
     --cache /private/geoint-worker-cache \
     --ca-file /private/coordinator-ca.pem

   /path/to/python3 /path/to/backend/pool_worker.py \
     --config /private/geoint-worker-cache/worker.json
   ```

   In Windows PowerShell, run the corresponding commands with your provisioned `python.exe`, quoted Windows paths, and PowerShell's `&` call operator. Run each command on a single line.
4. In **Worker pool**, enable **Use enrolled workers for supported jobs**, set **U-Net training** to two synchronized workers, and save. Check that both workers report a recent connection, the expected devices and enough available RAM.

## ARCH-02 · Distributed inference

In **Manual inference**, run one supported, qualified classifier on two separate local-image submissions. A qualified U-Net can exercise GPU routing. Random Forest exercises the CPU route. Open both outputs and export a GeoTIFF for each. Verify from the run reports that one completed run used Worker A and the other used Worker B. If both landed on the same host, run another submission while only the intended worker is polling.

For each completed run, save its JSON report from the coordinator's local browser at `/api/inference/runs/RUN_ID`. The report should show `status`, `execution_route`, `distributed.worker_id`, `distributed.device`, `distributed.task_id` and `distributed.attempt`. Record each export filename and SHA-256. Include the actual GPU device in the evidence if CUDA or MPS is part of the test.

## ARCH-04 · Synchronized training and recovery

1. Use **Training & qualification** to create a new U-Net configuration. Use local imagery and separate fitting and validation regions for each class. Run **Check samples**. Keep the job within eight images and a 384 px grid. Use several epochs so there is time to observe a committed round before testing recovery.
2. Confirm both workers contribute to the same training round. In the disposable coordinator, record the run ID, committed round number, checkpoint SHA-256 and the member task IDs. Pause at a checkpoint if needed to make the restart observable. Restart the disposable coordinator with the same workspace and runtime. Resume if it was paused. Verify the same run continues, the checkpoint hash stays the same, and the previously committed round is recorded once.
3. Save the completed training report from `/api/inference/runs/RUN_ID`. Retain `distributed_training.rounds` with each member's worker ID, device, task ID and attempt, each round's checkpoint hash, and the validation manifest and per-class precision, recall and F1. Record the qualification decision. If the candidate meets the configured goals, approve the release, run inference, and record the resulting GeoTIFF filename and SHA-256.

## Evidence to share

- One topology note with coordinator version, two distinct worker IDs, physical host aliases, OS versions, RAM and actual CPU/GPU used. Include a screenshot of both workers connected.
- ARCH-02 evidence for two completed runs on different workers, the two report summaries, and the two GeoTIFF hashes.
- ARCH-04 evidence with restart time, the pre- and post-restart checkpoint hashes for the same committed round, the completed training summary, held-out metrics, and the qualified inference/export summary.
- Record each outcome in **Help → Release validation** and use **Export results** to save `Geo-Intelligence-validation.json`.

Use synthetic or public fixture data and share only the necessary report fields. Keep worker keys, service keys, invitations, raw imagery, private hostnames and full workspace paths local.
