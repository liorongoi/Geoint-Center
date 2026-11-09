# Geo Intelligence

From imagery to intelligence.

This is the Mac web application source for build `0.2.25+mac.web.49`. It includes the web workspace, the MC² map integration, and the enrolled worker client. This build validates worker task identifiers and attempt numbers before creating worker task directories.

- [Worker enrollment and validation](docs/worker-validation.md)
- [Application help](backend/static/user-manual.html)
- [MC² license](backend/MC2-LICENSE)

The source snapshot uses the `backend/` layout of the native Mac web release. `backend/serve.py` starts the app and map services on loopback. `requirements/` contains dependency specifications and reference locks. Use the matching provisioned runtime and local model components for inference and training. Runtime components, model weights, workspace data and installer media are distributed separately under their applicable terms.

Catalog and supplier integrations use read-only contracts. Workers use private, enrolled identities and receive only the artifacts needed for their assigned jobs.
