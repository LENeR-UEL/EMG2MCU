from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display
from ._paths import output_root
from .signal import envelope_smooth


def visualize_split(X, y):
    """
    Visualiza os conjuntos de treino e teste
    como DataFrame para eventos multicanais.

    Parameters
    ----------
    X : dict
        {"train": X_train, "test": X_test}

    y : dict
        {"train": y_train, "test": y_test}
    """

    rows = []

    for subset in ["train", "test"]:

        X_subset = X[subset]
        y_subset = y[subset]

        for event, label in zip(X_subset, y_subset):

            signals = event["signals"]
            n_channels = signals.shape[0]

            row = {
                "set": subset,
                "event_id": event["event_id"],
                "label": label,
                "event_name": event["event_name"],
                "signals_shape": signals.shape,
                "n_channels": n_channels,
                "n_samples": signals.shape[1],
                "duration_ms": event["event_duration_ms"],
                "start_idx": event["start_idx"],
                "end_idx": event["end_idx"],
            }

            # shape de cada canal
            for ch in range(n_channels):
                row[f"ch{ch+1}_shape"] = signals[ch].shape

            rows.append(row)

    display(pd.DataFrame(rows))

    return pd.DataFrame(rows)


def plot_EMG(signals,
             label_aligned,
             mode="",
             metric="",
             folder_name="SubjectX",
             channel_names=None):

    """
    signals        -> dados já escolhidos (Raw ou Filtered)
    label_aligned  -> labels no tempo
    metric         -> se é raw ou filtered
    mode           -> só usado para organização de pastas
    folder_name    -> pasta base de saída
    """
    # Validação da metrica e modo
    metric = metric.lower()
    mode = mode.lower()

    if metric not in ("raw", "filtered"):
        raise ValueError(
            f"Modo inválido: '{metric}'. O parâmetro 'metric' deve ser 'raw' ou 'filtered'."
        )

    if mode not in ("benchmark", "deploy"):
        raise ValueError(
            f"Modo inválido: '{mode}'. O parâmetro 'mode' deve ser 'benchmark' ou 'deploy'."
        )

    # -------------------------
    # segurança
    # -------------------------
    signals = signals
    label_aligned = np.asarray(label_aligned)

    # -------------------------
    # limites globais
    # -------------------------
    max_global = max(np.max(s) for s in signals)
    min_global = min(np.min(s) for s in signals)

    margem = 0.05 * max(abs(max_global), abs(min_global))
    ylim = (min_global - margem, max_global + margem)

    # -------------------------
    # nomes dos canais
    # -------------------------
    if channel_names is None:
        channel_names = [f"Chn{i+1}" for i in range(len(signals))]

    # -------------------------
    # máscara
    # -------------------------
    mask = label_aligned > 0

    # -------------------------
    # pasta dinâmica baseada no mode
    # -------------------------
    if mode == "benchmark":
      save_path = f"{output_root()}/{folder_name}/[BENCHMARK]Zero-Phase/Plots/{metric.capitalize()}"
    elif mode == "deploy":
      save_path = f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/Plots/{metric.capitalize()}"
    os.makedirs(save_path, exist_ok=True)

    if metric == "raw":
        name_rule = f"{metric}"
    elif metric == "filtered":
        name_rule = f"{metric}_{mode}"

    # -------------------------
    # plot por canal
    # -------------------------
    for i, (sig, titulo) in enumerate(zip(signals, channel_names)):

        sig = np.asarray(sig)

        # segurança de alinhamento
        if len(sig) != len(label_aligned):
            raise ValueError(
                f"Mismatch entre sinal ({len(sig)}) e label ({len(label_aligned)}) no canal {i}"
            )

        plt.figure(figsize=(10, 4))

        plt.plot(sig, label=titulo, linewidth=1)

        # -------------------------
        # label overlay
        # -------------------------
        if (label_aligned != 0).any():
            # label_signal = label_aligned * (max_global - 1)
            baseline = np.median(sig)

            topo = ylim[1] - 0.05 * (ylim[1] - ylim[0])

            max_label = np.max(label_aligned)

            label_signal = np.interp(
                label_aligned,
                [0, max_label],
                [baseline, topo]
            )

            plt.plot(
                label_signal,
                label="Label",
                color="red",
                linewidth=1,
                alpha=0.5
            )

        # -------------------------
        # highlight de ativação
        # -------------------------
        plt.fill_between(
            range(len(label_aligned)),
            ylim[0],
            ylim[1],
            where=mask,
            color="red",
            alpha=0.08
        )

        plt.title(titulo)
        plt.xlabel("Samples")
        plt.ylabel("Amplitude (mV)")
        plt.ylim(ylim)
        plt.legend()
        plt.grid(True)

        plt.savefig(
            f"{save_path}/{name_rule}_{i+1}.svg",
            format="svg",
            bbox_inches="tight"
        )

        # plt.show()
        plt.close()


def plot_events(
    signal,
    events,
    mode="",
    folder_name="SubjectX",
    channel_names=None,
    figsize=(15, 4),
    alpha_signal=0.7
):
    """
    Plota sinais multicanais e destaca eventos.
    """
    mode = mode.lower()

    if mode not in ("benchmark", "deploy"):
        raise ValueError(
            f"Modo inválido: '{mode}'. O parâmetro 'mode' deve ser 'benchmark' ou 'deploy'."
        )

    signal = np.asarray(signal)

    # ==================================================
    # garante shape (n_channels, n_samples)
    # ==================================================
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    n_channels = signal.shape[0]

    # ==================================================
    # nomes dos canais
    # ==================================================
    if channel_names is None:

        channel_names = [
            f"Channel {i+1}"
            for i in range(n_channels)
        ]

    # ==================================================
    # classes
    # ==================================================
    class_names = list(events.keys())

    # ==================================================
    # cores
    # ==================================================
    color_map = {
        0: "gray",
        1: "red",
        2: "blue",
        3: "green",
        4: "orange",
        5: "purple",
    }

    class_color = {}

    if mode == "benchmark":
        save_path = f"{output_root()}/{folder_name}/[BENCHMARK]Zero-Phase/Plots/Events"
    elif mode == "deploy":
        save_path = f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/Plots/Events"

    os.makedirs(save_path, exist_ok=True)

    for idx, cname in enumerate(class_names):

        class_color[cname] = color_map.get(idx, "black")

    # ==================================================
    # plot por canal
    # ==================================================
    for ch in range(n_channels):

        sig = signal[ch]

        plt.figure(figsize=figsize)

        # ==============================================
        # sinal base
        # ==============================================
        plt.plot(
            sig,
            color="lightgray",
            linewidth=1,
            alpha=alpha_signal
        )

        plotted_events = set()

        # ==============================================
        # percorre classes
        # ==============================================
        for idx, class_name in enumerate(class_names):

            class_events = events[class_name]

            color = class_color[class_name]

            # ==========================================
            # percorre eventos
            # ==========================================
            for event in class_events:

                s_event = event["start_event"]
                e_event = event["end_event"]

                s = event["start_idx"]
                e = event["end_idx"]

                event_key = (
                    class_name,
                    s_event,
                    e_event
                )

                # evita duplicação
                if event_key in plotted_events:
                    continue

                plotted_events.add(event_key)

                # ======================================
                # trecho do evento
                # ======================================
                plt.plot(
                    np.arange(s_event, e_event),
                    sig[s_event:e_event],
                    color=color,
                    linewidth=1.5
                )

                # relax sem preenchimento
                if idx != 0:

                    plt.fill_between(
                        np.arange(s_event, e_event),
                        np.min(sig),
                        np.max(sig),
                        color=color,
                        alpha=0.15
                    )

                    # bordas evento
                    plt.axvline(
                        s_event,
                        color=color,
                        linestyle="--",
                        alpha=0.4
                    )

                    plt.axvline(
                        e_event,
                        color=color,
                        linestyle="--",
                        alpha=0.4
                    )

                    # padding
                    plt.axvline(
                        s,
                        color=color,
                        linestyle=":",
                        alpha=0.3
                    )

                    plt.axvline(
                        e,
                        color=color,
                        linestyle=":",
                        alpha=0.3
                    )

        # ==============================================
        # legenda
        # ==============================================
        legend_elements = []

        for cname in class_names:

            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=class_color[cname],
                    lw=3,
                    label=cname
                )
            )

        for cname in class_names[1:]:

            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=class_color[cname],
                    linestyle=':',
                    lw=2,
                    label=f'Padding - {cname}',
                )
            )

        plt.legend(handles=legend_elements)

        plt.title(f"Channel {ch+1} - {channel_names[ch]}")

        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            f"{save_path}/found_events_ch{ch+1}.svg",
            format="svg",
            bbox_inches="tight"
        )


def plot_envelope(signals,
                  mode="",
                  folder_name="SubjectX",
                  channel_names=None,
                  window_size=500):

    """
    signals         -> sinais já processados
    filename        -> nome base dos arquivos
    folder_name     -> pasta base
    channel_names   -> nomes dos canais
    window_size     -> suavização do envelope
    """
    mode = mode.lower()
    if mode not in ("benchmark", "deploy"):
        raise ValueError(
            f"Modo inválido: '{mode}'. O parâmetro 'mode' deve ser 'benchmark' ou 'deploy'."
        )

    # -------------------------
    # segurança
    # -------------------------
    signals = [np.asarray(s) for s in signals]

    # -------------------------
    # normalização
    # -------------------------
    sinais_norm = [
        s / np.max(np.abs(s))
        if np.max(np.abs(s)) != 0 else s
        for s in signals
    ]

    # -------------------------
    # nomes dos canais
    # -------------------------
    if channel_names is None:
        channel_names = [f"Chn{i+1}" for i in range(len(signals))]

    # -------------------------
    # pasta
    # -------------------------
    if mode == "benchmark":
      save_path = f"{output_root()}/{folder_name}/[BENCHMARK]Zero-Phase/Plots/Envelope"
    elif mode == "deploy":
      save_path = f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/Plots/Envelope"
    os.makedirs(save_path, exist_ok=True)

    # -------------------------
    # plot por canal
    # -------------------------
    for i, (sig, titulo) in enumerate(zip(sinais_norm, channel_names)):

        env = envelope_smooth(sig, window_size=window_size, mode='same')

        plt.figure(figsize=(10, 4))

        # sinal EMG
        plt.plot(
            sig,
            label="EMG",
            alpha=0.1,
            linewidth=1
        )

        # envelope
        plt.plot(
            env,
            label="Envelope",
            color="red",
            linewidth=1
        )

        plt.title(titulo)
        plt.xlabel("Samples")
        plt.ylabel("Normalized Amplitude")

        plt.legend()
        plt.grid(True)

        plt.savefig(
            f"{save_path}/envelope_ch{i+1}.svg",
            format="svg",
            bbox_inches="tight"
        )

        # plt.show()
        plt.close()


def plot_fft(metrics, mode="", folder_name="SubjectX", channel_names=None):
    """
    Plota FFT de múltiplos canais.

    Parâmetros
    ----------
    metrics : dict
        Dicionário contendo "fft" com lista de canais:
        metrics["fft"][i] = {"freq": array, "values": array complexo}

    filename : str
        Nome base para salvar os arquivos.

    folder_name : str
        Pasta base do sujeito.

    channel_names : list[str] | None
        Nome dos canais (ex: ["Biceps", "Triceps"]).
        Se None, usa "Canal 1..N".
    """
    mode = mode.lower()

    if mode not in ("benchmark", "deploy"):
        raise ValueError(
            f"Modo inválido: '{mode}'. O parâmetro 'mode' deve ser 'benchmark' ou 'deploy'."
        )


    # =========================================================
    # PATH SETUP
    # =========================================================
    if mode == "benchmark":
      save_path = f"{output_root()}/{folder_name}/[BENCHMARK]Zero-Phase/Plots/FFT"
    elif mode == "deploy":
      save_path = f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/Plots/FFT"
    os.makedirs(save_path, exist_ok=True)

    fft_list = metrics["fft"]

    # fallback de nomes
    if channel_names is None:
        channel_names = [f"Canal {i+1}" for i in range(len(fft_list))]

    # =========================================================
    # PLOTS
    # =========================================================
    for i, ch in enumerate(fft_list):

        freq = ch["freq"]
        values = np.abs(ch["values"])  # magnitude

        plt.figure(figsize=(10, 4))

        plt.plot(freq, values)

        # -------------------------
        # TITLE / LABELS
        # -------------------------
        plt.title(f"FFT - {channel_names[i]}")
        plt.xlabel("Frequência (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        # -------------------------
        # SAVE
        # -------------------------
        plt.savefig(
            f"{save_path}/FFT_ch{i+1}.svg",
            format="svg",
            bbox_inches="tight"
        )

        # plt.show()
        plt.close()


def plot_psd(metrics, mode="", folder_name="SubjectX", channel_names=None):
    """
    Plota Power Spectral Density (PSD) para múltiplos canais.

    Parameters
    ----------
    metrics : dict
        Dicionário contendo a chave "psd", onde cada elemento é:
        {
            "freq": array de frequências,
            "values": array de potência (PSD)
        }

    filename : str
        Sufixo usado no nome dos arquivos salvos.

    folder_name : str
        Pasta base do sujeito/dataset.

    channel_names : list[str] | None
        Nomes dos canais para título dos gráficos.
        Se None, usa "Canal 1", "Canal 2", etc.
    """
    mode = mode.lower()

    if mode not in ("benchmark", "deploy"):
        raise ValueError(
            f"Modo inválido: '{mode}'. O parâmetro 'mode' deve ser 'benchmark' ou 'deploy'."
        )

    # =========================================================
    # PATH SETUP
    # =========================================================
    if mode == "benchmark":
      save_path = f"{output_root()}/{folder_name}/[BENCHMARK]Zero-Phase/Plots/PSD"
    elif mode == "deploy":
      save_path = f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/Plots/PSD"
    os.makedirs(save_path, exist_ok=True)

    psd_list = metrics["psd"]

    # fallback de nomes
    if channel_names is None:
        channel_names = [f"Canal {i+1}" for i in range(len(psd_list))]

    # =========================================================
    # PLOTS
    # =========================================================
    for i, ch in enumerate(psd_list):

        freq = ch["freq"]
        values = ch["values"]

        plt.figure(figsize=(10, 4))

        plt.plot(freq, values)

        # -------------------------
        # TITLE / LABELS
        # -------------------------
        plt.title(f"PSD - {channel_names[i]}")
        plt.xlabel("Frequência (Hz)")
        plt.ylabel("Potência (mV²/Hz)")
        plt.grid(True)

        # -------------------------
        # SAVE
        # -------------------------
        plt.savefig(
            f"{save_path}/PSD_ch{i+1}.svg",
            format="svg",
            bbox_inches="tight"
        )

        # plt.show()
        plt.close()


def plot_mdf(
    metrics,
    labels,
    mode="",
    folder_name="SubjectX",
    channel_names=None
):
    mode = mode.lower()
    if mode not in ("benchmark", "deploy"):
        raise ValueError(
            f"Modo inválido: '{mode}'. O parâmetro 'mode' deve ser 'benchmark' ou 'deploy'."
        )

    if mode == "benchmark":
      save_path = f"{output_root()}/{folder_name}/[BENCHMARK]Zero-Phase/Plots/MDF"
    elif mode == "deploy":
      save_path = f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/Plots/MDF"
    os.makedirs(save_path, exist_ok=True)

    n_channels = len(metrics["mdf"][labels[0]])

    if channel_names is None:
        channel_names = [f"Canal {i+1}" for i in range(n_channels)]

    for ch in range(n_channels):

        plt.figure(figsize=(12, 4))

        for label in labels:
            values = metrics["mdf"][label][ch]

            plt.plot(values, marker=".", markersize=4, label=label)

        plt.title(f"MDF - {channel_names[ch]}")
        plt.xlabel("Evento")
        plt.ylabel("MDF (Hz)")
        plt.grid(True)
        plt.legend()

        plt.savefig(
            f"{save_path}/mdf_ch{ch+1}.svg",
            format="svg",
            bbox_inches="tight"
        )

        # plt.show()
        plt.close()


def plot_mnf(
    metrics,
    labels,
    mode="",
    folder_name="SubjectX",
    channel_names=None
):
    """
    Plota a Frequência Média de Potência (MNF) para múltiplas classes.
    """
    mode = mode.lower()
    if mode not in ("benchmark", "deploy"):
        raise ValueError(
            f"Modo inválido: '{mode}'. O parâmetro 'mode' deve ser 'benchmark' ou 'deploy'."
        )

    if mode == "benchmark":
      save_path = f"{output_root()}/{folder_name}/[BENCHMARK]Zero-Phase/Plots/MNF"
    elif mode == "deploy":
      save_path = f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/Plots/MNF"
    os.makedirs(save_path, exist_ok=True)

    mnf = metrics["mnf"]

    n_channels = len(mnf[labels[0]])

    if channel_names is None:
        channel_names = [f"Canal {i+1}" for i in range(n_channels)]

    for ch in range(n_channels):

        plt.figure(figsize=(12, 4))

        for label in labels:
            plt.plot(
                mnf[label][ch], marker=".", markersize=4, label=label)

        plt.title(f"MNF - {channel_names[ch]}")
        plt.xlabel("Evento")
        plt.ylabel("MNF (Hz)")
        plt.grid(True)
        plt.legend()

        plt.savefig(
            f"{save_path}/mnf_ch{ch+1}.svg",
            format="svg",
            bbox_inches="tight"
        )

        # plt.show()
        plt.close()
