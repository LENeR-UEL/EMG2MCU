# EMG2MCU

EMG2MCU provides surface-EMG preprocessing, event segmentation, feature
extraction, model selection, and deployable C++/Arduino model generation.

## Run the example

Install the project and its dependencies in the active Python environment:

```powershell
python -m pip install -e .
```

Open [EMG2MCU_example.ipynb](./example/EMG2MCU_example.ipynb) from the
repository root. It generates a deterministic synthetic recording at runtime,
so no real participant data is needed or included in the repository. The
synthetic signals demonstrate the software pipeline only and are not
physiological data. Generated artifacts are written to the ignored
`output_emg2mcu/` directory.

The main entry point is `EMG2MCU.automl`. Configuration dataclasses and
lower-level processing, plotting, feature, and model utilities are also
available from their corresponding package modules.

## Documentation

The full API reference, configuration dataclasses, and pipeline examples
are published on Read the Docs:

- Stable: <https://emg2mcu.readthedocs.io/en/stable/>
- Latest: <https://emg2mcu.readthedocs.io/en/latest/>
- Source for the documentation lives in [./docs](./docs); see
  [./docs/README.md](./docs/README.md) for how to build it locally.

## Development

Tests run on every pull request across Python 3.10, 3.11, and 3.12 via
[.github/workflows/tests.yml](./.github/workflows/tests.yml). Merging to
`main` triggers [the release workflow](./.github/workflows/release.yml),
which auto-bumps the patch version, publishes to PyPI, builds the Sphinx
docs, and opens a GitHub Release whose body is taken from the merged PR's
`## Changelog` section.
