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
