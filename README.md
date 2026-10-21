# EMG2MCU

EMG2MCU provides surface-EMG preprocessing, event segmentation, feature
extraction, model selection, and deployable C++/Arduino model generation.

## Run the example

Install the project and its dependencies in the active Python environment:

```powershell
python -m pip install -e .
```

Open [EMG2MCU_example.ipynb](./example/EMG2MCU_example.ipynb) from the
repository root.
It runs the same example recording and settings as the original notebook, but
writes its artifacts to the ignored `output_emg2mcu/` directory. The original
notebook artifacts in `output/` are left untouched for comparison.

The main entry point is `EMG2MCU.automl`. Configuration dataclasses and
lower-level processing, plotting, feature, and model utilities are also
available from their corresponding package modules.
