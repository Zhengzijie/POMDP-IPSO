# Contributing

Contributions that improve reproducibility, documentation, portability, tests, or fair baseline coverage are welcome.

## Before opening a change

1. Search existing Issues to avoid duplicates.
2. For scientific changes, state the hypothesis, benchmark budget, random seeds, and expected output files.
3. Do not add operational coordinates, personal data, credentials, restricted field logs, or unlicensed third-party assets.
4. Preserve the distinction between reconstructed synthetic results and operational evidence.

## Development setup

```bash
python -m venv .venv
python -m pip install -r requirements-dev.txt
python -m pytest
python -m compileall -q scripts
```

## Pull requests

- Keep changes focused and explain the scientific or reproducibility rationale.
- Add or update tests for altered data layouts or generators.
- Report hardware, software versions, seeds, and evaluation budgets for regenerated results.
- Do not overwrite tracked benchmark outputs without explaining every changed file.
- Update `CHANGELOG.md`, the dataset card, and reproduction guide when behavior or data change.

By contributing, you agree that your contribution is licensed under the repository’s MIT License and that you have the right to submit it.
