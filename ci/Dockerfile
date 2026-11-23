FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
WORKDIR /review
COPY requirements/ci.txt /tmp/ci.txt
RUN apt-get update && apt-get install -y --no-install-recommends openssl libexpat1 libgomp1 && rm -r /var/lib/apt/lists/* && python -m pip install --no-cache-dir -r /tmp/ci.txt
COPY backend /review/backend
COPY scripts/check-regressions.sh /review/scripts/check-regressions.sh
ENTRYPOINT ["/usr/bin/timeout", "120s", "/usr/bin/env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin", "HOME=/tmp", "TMPDIR=/tmp", "/bin/bash", "/review/scripts/check-regressions.sh"]
