# Remaining requirements

Public-safe summary of the Mac register dated 2026-11-09 and subsequent regression-test merge. Target status has not been independently revalidated.

There are 12 entries without final evidence. Three are active across network safety and physical worker validation. Regression tests do not close these target requirements.

| ID | Status | Requirement and remaining evidence |
| --- | --- | --- |
| POLICY-02 | waiting for target validation | Connected network safety. Capture permitted destinations and forwarding controls on the actual deployment using synthetic data. |
| ARCH-02 | waiting for target validation | Distributed inference. Run inference on two enrolled physical hosts; retain worker/device identity and exported file hashes. CUDA needs compatible hardware. |
| ARCH-04 | waiting for target validation | Distributed training. Run synchronized U-Net across two physical hosts, restart after a committed checkpoint, verify recovery and protected validation splits. |
| DEPLOY-01 | waiting for target validation | Windows current-user installation. Install prerequisites and app under a standard Windows account; verify launch without elevation. |
| DEPLOY-02 | waiting for target validation | Windows offline media. Verify USB paths, receipts/resume and launch without downloads on Windows. |
| DEPLOY-04 | waiting for target validation | Windows update recovery. Verify update and rollback preserve workspace on native Windows. |
| BRAND-01 | waiting for target validation | Native branding. Inspect branding in existing native installer/app on target. |
| MODEL-04 | on hold | Rotated few-shot detector. Keep in Training pending operator review; independent qualification gates remain unmet. |
| MODEL-05 | experimental | DSen2 super-resolution. Keep experimental; performance is scene-dependent and native-resolution validation remains open. |
| AGENT-06 | on hold | Additional commercial engines. Oracle 19c and unreviewed engine contracts remain pending. Existing supported Db2 contract has isolated evidence. |
| DEPLOY-03 | on hold | Mac desktop packaging. Native Mac web remains the current path; desktop/hybrid installer validation is deferred. |
| DEPLOY-05 | on hold | Offline Linux packaging. Deferred by operator instruction. |

Run `bash scripts/check-regressions.sh` after installing `requirements/ci.txt`. GitHub Actions builds the same test container and runs it with networking disabled, read-only source and bounded resources. Tests use synthetic inputs and temporary storage. Physical validation is described in [the worker guide](worker-validation.md).

Private supplier trials and custom integrations requiring supplier access are outside required release scope.

On 23 November 2026, all 60 regressions passed in a clean Python 3.12 Linux ARM64 container on the Mac, using the CI dependency file. GitHub runner execution is reported separately by the PR check.
