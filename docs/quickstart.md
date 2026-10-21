# Quickstart

## Installation

```bash
pip install emg2mcu
```

EMG2MCU requires Python 3.10+.

For local development installs and longer setup notes, see `docs/README.md`
in the repository.

## Minimal example

The simplest way to use EMG2MCU is through the `automl` pipeline:

```python
from EMG2MCU import (
    DataConfig,
    PreprocessingConfig,
    EventsConfig,
    WindowingConfig,
    FeatureConfig,
    ClassificationConfig,
    NormalizeConfig,
    automl,
)

# 1. Configure the data source
data_cfg = DataConfig(
    data_file="recording.csv",
    data_path="data",
    participant="p01",
    label_names=["rest", "fist", "open"],
    channel_names=["CH1"],
)

# 2. Configure the preprocessing (bandpass + notch)
pp_cfg = PreprocessingConfig(
    fs=1000,
    lowcut=30,
    highcut=450,
    notch_freqs=(60,),
    Q=30,
)

# 3. Configure the event segmentation
evt_cfg = EventsConfig()

# 4. Configure the windowing step
win_cfg = WindowingConfig(window_ms=200, step_ms=100)

# 5. Configure the feature extraction
feat_cfg = FeatureConfig(features=["IEMG", "RMS", "MAV", "WL"])

# 6. Configure the classifier benchmark
clf_cfg = ClassificationConfig(models=["random_forest", "lightgbm"])

# 7. Configure the normalization
norm_cfg = NormalizeConfig(method="standard")

# 8. Run the full pipeline
state = automl(
    data_cfg=data_cfg,
    preprocessing_cfg=pp_cfg,
    events_cfg=evt_cfg,
    windowing_cfg=win_cfg,
    feature_cfg=feat_cfg,
    classification_cfg=clf_cfg,
    normalize_cfg=norm_cfg,
)
```

The pipeline state object exposes intermediate results, the benchmarked
models, and the deployable artifacts (C++ source for the selected model and
biquad filter coefficients for the ESP32/Arduino).

## Where to next?

- Browse the {doc}`API reference <api_reference>` for every public symbol.
- Open `example/EMG2MCU_example.ipynb` from the repository root for a
  complete end-to-end demo using a deterministic synthetic recording.
- Use `EMG2MCU.output_context` to redirect generated artifacts away from
  the working directory.
