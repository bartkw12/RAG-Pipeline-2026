# UC37.5 RAG Pipeline (Scaffold)

This project is structured as a Retrieval-Augmented Generation pipeline:

- `src/Ingestion/`
- `src/Retrieval/`
- `src/Generation/`
- `src/config/` (environment loading and required variable validation)

## Quick start

1. Clone the repository.
2. Copy `.env.example` to `.env`.
3. Add your own API keys and endpoints in `.env`.
4. Run pipeline modules from `src/` as they are added.

## Security model

- Real secrets stay in `.env` (ignored by git).
- `.env.example` is committed as a template only.
- `config*.json` and common private key/cert files are ignored.

## Environment helpers

Use helpers in `src/config/env_config.py`:

- `load_local_env()` to load local `.env` values.
- `require_env([...])` to fail fast if required variables are missing.
