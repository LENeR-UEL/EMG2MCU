# EMG2MCU

**EMG2MCU** provides surface-EMG preprocessing, event segmentation, feature
extraction, model selection, and deployable C++/Arduino model generation.

The package is designed for the full EMG-to-microcontroller pipeline: clean
raw signals, segment events, extract time-domain and time-series features,
benchmark several classifiers, and emit a deployable C++ implementation of
the selected model (including biquad filter coefficients for the ESP32/Arduino
ADC front-end).

::::{grid} 4
:::{grid-item-card} Getting Started
:link: quickstart
:link-type: doc

Install EMG2MCU and run your first pipeline in minutes.
:::
:::{grid-item-card} Pipeline
:link: quickstart
:link-type: doc

Configuration dataclasses for the end-to-end `automl` pipeline.
:::
:::{grid-item-card} Visualization
:link: api_reference
:link-type: doc

Plotting helpers for raw, filtered, and event-segmented EMG.
:::
:::{grid-item-card} API Reference
:link: api_reference
:link-type: doc

Full API documentation for the EMG2MCU package.
:::
::::

## Quick example

The example notebook generates a deterministic synthetic recording at
runtime, so no real participant data is needed.

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

data_cfg = DataConfig(data_file="recording.csv")
pp_cfg = PreprocessingConfig(fs=1000, lowcut=30, highcut=450)
evt_cfg = EventsConfig()
win_cfg = WindowingConfig(window_ms=200, step_ms=100)
feat_cfg = FeatureConfig(features=["IEMG", "RMS", "MAV", "WL"])
clf_cfg = ClassificationConfig(models=["random_forest", "lightgbm"])
norm_cfg = NormalizeConfig(method="standard")

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

## Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

quickstart
```

```{toctree}
:maxdepth: 1
:caption: API Reference

api_reference
```
