from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gc
import numpy as np

from ._paths import output_context
from .data import import_data, segment_dataset, split_data
from .features import build_featureset, compute_metrics, normalize_data
from .models import model_selection
from .signal import analyze_onsets, debug_esp32_filter, detect_onset, emg_filter, find_events, generate_esp32_code, remove_standby
from .visualization import plot_EMG, plot_envelope, plot_events, plot_fft, plot_mdf, plot_mnf, plot_psd


@dataclass
class DataConfig:
    data_file: str
    data_path: str = ""
    participant: str = ""

    label_names: list | None = None
    channel_names: list | None = None


@dataclass
class PreprocessingConfig:
    fs: int = 1000
    lowcut: float = 30
    highcut: float = 450
    notch_freqs: tuple = (60,)
    Q: float = 30
    signal_padding: float = 0.2
    butter_order: int = 4
    adc_pin: int = 34


@dataclass
class EventsConfig:
    smooth_win_size: int = 500
    onset_sustain_ms: int = 1500
    onset_tol_ratio: float = 0.2
    pre_onset_removal_ratio: float = 0.25
    event_pad_ms: int = 0


@dataclass
class WindowingConfig:
    test_size: float = 0.2
    window_ms: int = 50
    overlap: float = 0.0


@dataclass
class FeatureConfig:
    features: list | None = None
    return_dataframe: bool = True


@dataclass
class NormalizeConfig:
    scale_y: bool = False


@dataclass
class ClassificationConfig:
    algorithms: list = field(default_factory=lambda: [
        "lightgbm",
        "xgboost",
        "catboost",
        "random_forest",
        "extra_trees",
        "logistic_regression",
        "lda",
        "svm",
        "knn",
        "mlp",
    ])
    sort_by: str = "accuracy"


@dataclass
class PipelineState:
    signals: Any = None
    labels: Any = None
    benchmark: dict = field(default_factory=dict)
    deploy: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)


def _load_data(cfg):

    signals, labels = import_data(
        cfg.data_file,
        cfg.data_path
    )

    return {
        "signals": signals,
        "labels": labels,
        "participant": cfg.participant,
        "label_names": cfg.label_names,
        "channel_names": cfg.channel_names,
        "benchmark": {},
        "deploy": {}
    }


def _preprocessing(state, cfg):

    # raw_all, filtered_all, labels_aligned, _, _ = emg_filter(
    #     signals=state["signals"],
    #     label=state["labels"],
    #     fs=cfg.fs,
    #     lowcut=cfg.lowcut,
    #     highcut=cfg.highcut,
    #     mode="benchmark",
    #     folder_name=state["participant"],
    #     notch_freqs=cfg.notch_freqs,
    #     Q=cfg.Q,
    #     padding=cfg.signal_padding,
    #     butter_order=cfg.butter_order
    # )

    raw_all, filtered_deploy, labels_aligned, sos, filter_specs = emg_filter(
    signals=state["signals"],
        label=state["labels"],
        fs=cfg.fs,
        lowcut=cfg.lowcut,
        highcut=cfg.highcut,
        mode="deploy",
        folder_name=state["participant"],
        notch_freqs=cfg.notch_freqs,
        Q=cfg.Q,
        padding=cfg.signal_padding,
        butter_order=cfg.butter_order
    )

    results = debug_esp32_filter(signals=state["signals"], sos_configs=sos, folder_name=state["participant"])
    generate_esp32_code(sos, fs=cfg.fs, adc_pin=cfg.adc_pin, folder_name=state["participant"], filter_specs=filter_specs)

    # # BENCHMARK
    # state["benchmark"]["fs"] = cfg.fs
    # state["benchmark"]["raw"] = raw_all
    # state["benchmark"]["filtered"] = filtered_all
    # state["benchmark"]["labels"] = labels_aligned
    # state["benchmark"]["label_names"] = state["label_names"]
    # state["benchmark"]["channel_names"] = state["channel_names"]

    # Agora o sinal original não é mais necessário
    del state["signals"]
    del sos
    del filter_specs
    del results
    gc.collect()
    # DEPLOY
    state["deploy"]["fs"] = cfg.fs
    state["deploy"]["raw"] = raw_all
    state["deploy"]["filtered"] = filtered_deploy
    state["deploy"]["labels"] = labels_aligned
    state["deploy"]["label_names"] = state["label_names"]
    state["deploy"]["channel_names"] = state["channel_names"]

    return state


def _events(state, cfg):

    # for mode in ["benchmark", "deploy"]:
    for mode in ["deploy"]:

        onsets = detect_onset(
            state[mode]["filtered"],
            fs=state[mode]["fs"],
            folder_name=state["participant"],
            mode=mode,
            window_size=cfg.smooth_win_size,
            sustain_ms=cfg.onset_sustain_ms,
            tolerance_ratio=cfg.onset_tol_ratio
        )

        onset_final = analyze_onsets(onsets)

        raw_cut, filtered_cut, labels_cut = remove_standby(
            state[mode]["filtered"],
            state[mode]["raw"],
            state[mode]["labels"],
            onset_final,
            pre_onset_ratio=cfg.pre_onset_removal_ratio
        )

        events = find_events(
            signal=filtered_cut,
            label=labels_cut,
            class_labels=np.unique(state[mode]["labels"][state[mode]["labels"] >= 0]),
            class_names=state[mode]["label_names"],
            fs=state[mode]["fs"],
            pad_ms=cfg.event_pad_ms)

        state[mode]["raw"] = raw_cut
        state[mode]["filtered"] = filtered_cut
        state[mode]["labels"] = labels_cut
        state[mode]["events"] = events
        state[mode]["smooth_win_size"] = cfg.smooth_win_size

        # Libera variáveis intermediárias
        del onsets
        del onset_final
        del events

        gc.collect()

    return state


def _windowing(state, cfg):

    # for mode in ["benchmark", "deploy"]:
    for mode in ["deploy"]:

        X, y = split_data(
            events_dict=state[mode]["events"],
            mode=mode,
            folder_name=state["participant"],
            test_size=cfg.test_size
        )

        X_segmented, y_segmented, meta = segment_dataset(
            X, y,
            fs=state[mode]["fs"],
            window_ms=cfg.window_ms,
            overlap=cfg.overlap
        )

        # Os X/y antigos não são mais necessários
        del X
        del y
        del meta

        state[mode]["X"] = X_segmented
        state[mode]["y"] = y_segmented

        metrics = compute_metrics(state[mode]["events"], labels=state[mode]["label_names"])

        state[mode]["metrics"] = metrics

        gc.collect()

    return state


def _plots(state, cfg=None):
    """
    Gera todos os plots do pipeline EMG.
    Não modifica o state.
    """
    # modes = ["benchmark", "deploy"]
    modes = ["deploy"]

    # -------------------------
    # 1. EMG (RAW / FILTERED)
    # -------------------------
    for mode in modes:
        # data = state.get(mode, {})
        # labels = data.get("labels")

        # if "raw" in data and labels is not None:
        plot_EMG(
            #data["raw"],
            # labels,
            signals=state[mode]["raw"],
            label_aligned=state[mode]["labels"],
            mode=mode,
            metric="Raw",
            folder_name=state["participant"],
            channel_names=state["channel_names"]
        )

        # if "filtered" in data and labels is not None:
        plot_EMG(
            # data["filtered"],
            # labels,
            signals=state[mode]["filtered"],
            label_aligned=state[mode]["labels"],
            mode=mode,
            metric="Filtered",
            folder_name=state["participant"],
            channel_names=state["channel_names"]
        )

    # -------------------------
    # 2. ENVELOPE
    # -------------------------
    # for mode in modes:
        # data = state.get(mode, {})

        # if "filtered" in data:
        plot_envelope(
            # data["filtered"],
            signals=state[mode]["filtered"],
            mode=mode,
            folder_name=state["participant"],
            channel_names=state["channel_names"],
            window_size=state[mode]["smooth_win_size"]
        )

    # -------------------------
    # 3. FFT / PSD / MDF / MNF
    # -------------------------
    # for mode in modes:
        # data = state.get(mode, {})

        # FFT / PSD → sinal filtrado (não features)
        # if "filtered" in data:
        plot_fft(
            state[mode]["metrics"],
            mode=mode,
            folder_name=state["participant"],
            channel_names=state["channel_names"]
            )
        plot_psd(
            state[mode]["metrics"],
            mode=mode,
            folder_name=state["participant"],
            channel_names=state["channel_names"]
        )

      # MDF / MNF → features
      # if "features" in data:
        plot_mdf(
            state[mode]["metrics"],
            mode=mode,
            labels=state["label_names"],
            folder_name=state["participant"],
            channel_names=state["channel_names"]
        )
        plot_mnf(
            state[mode]["metrics"],
            mode=mode,
            labels=state["label_names"],
            folder_name=state["participant"],
            channel_names=state["channel_names"]
        )

    # -------------------------
    # 4. EVENTS
    # -------------------------
    # for mode in modes:
        # data = state.get(mode, {})
        plot_events(
            signal=state[mode]["filtered"],
            events=state[mode]["events"],
            mode=mode,
            folder_name=state["participant"],
            channel_names=state["channel_names"]
        )


def _feature_engineering(state, cfg):

    # for mode in ["benchmark", "deploy"]:
    for mode in ["deploy"]:

        # X_feat, y = build_featureset(
        #     X=state[mode]["X"],
        #     y=state[mode]["y"],
        #     folder_name=state["participant"],
        #     extraction_method="tsfresh",
        #     mode=mode,
        #     return_dataframe=cfg.return_dataframe
        # )

        # state[mode]["tsfresh_features"] = X_feat

        X_feat, y = build_featureset(
            X=state[mode]["X"],
            y=state[mode]["y"],
            folder_name=state["participant"],
            extraction_method="time_domain",
            mode=mode,
            features=cfg.features,
            return_dataframe=cfg.return_dataframe
        )

        # X segmentado não é mais necessário
        del state[mode]["X"]
        gc.collect()

        state[mode]["td_features"] = X_feat
        state[mode]["y"] = y

    return state


def _normalize(state, cfg):

    # for mode in ["benchmark", "deploy"]:
    for mode in ["deploy"]:

        X_norm = normalize_data(
            X=state[mode]["td_features"],
            y=state[mode]["y"],
            extraction_method="time_domain",
            mode=mode,
            folder_name=state["participant"],
            scale_y=cfg.scale_y
        )

        state[mode]["td_features"] = X_norm

    return state


def _classification(state, cfg):

    results = {}

    # for mode in ["benchmark", "deploy"]:
    for mode in ["deploy"]:

        # results = model_selection(
        #     X=state[mode]["tsfresh_features"],
        #     y=state[mode]["y"],
        #     algorithms=cfg.algorithms,
        #     labels=state[mode]["label_names"],
        #     sort_by=cfg.sort_by,
        #     mode=mode,
        #     folder_name=participant,
        # )

        # state[mode]["tsfresh_results"] = results

        results = model_selection(
            state[mode]["td_features"],
            y=state[mode]["y"],
            algorithms=cfg.algorithms,
            labels=state[mode]["label_names"],
            sort_by=cfg.sort_by,
            mode=mode,
            folder_name=state["participant"],
        )

        state[mode]["td_results"] = results

    return state


def automl(
    data_cfg: DataConfig,
    preprocessing_cfg: PreprocessingConfig | None = None,
    events_cfg: EventsConfig | None = None,
    windowing_cfg: WindowingConfig | None = None,
    feature_cfg: FeatureConfig | None = None,
    normalize_cfg: NormalizeConfig | None = None,
    classify_cfg: ClassificationConfig | None = None,
    run_preprocessing: bool = True,
    run_windowing: bool = True,
    run_events: bool = True,
    run_plots: bool = True,
    run_features: bool = True,
    run_normalization: bool = True,
    run_classification: bool = True,
    output_dir: str | Path | None = None,
) -> dict:
    preprocessing_cfg = preprocessing_cfg or PreprocessingConfig()
    events_cfg = events_cfg or EventsConfig()
    windowing_cfg = windowing_cfg or WindowingConfig()
    feature_cfg = feature_cfg or FeatureConfig()
    normalize_cfg = normalize_cfg or NormalizeConfig()
    classify_cfg = classify_cfg or ClassificationConfig()

    with output_context(output_dir):
        state = _load_data(data_cfg)

        if run_preprocessing:
            state = _preprocessing(state, preprocessing_cfg)
        if run_events:
            state = _events(state, events_cfg)
        if run_windowing:
            state = _windowing(state, windowing_cfg)
        if run_plots:
            _plots(state)
        if run_features:
            state = _feature_engineering(state, feature_cfg)
        if run_normalization:
            state = _normalize(state, normalize_cfg)
        if run_classification:
            state = _classification(state, classify_cfg)

    return state
