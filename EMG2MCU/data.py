from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ._paths import output_root
from .visualization import visualize_split


def import_data(recording_file, path):
    dataset = f"{path}/{recording_file}"

    # Encontrar início dos dados (EndOfHeader)
    start = 0
    with open(dataset, "r") as f:
        for i, line in enumerate(f):
            if "EndOfHeader" in line:
                start = i + 1
                break

    # Ler dataframe
    df = pd.read_csv(
        dataset,
        sep=r"\s+",
        skiprows=start,
        engine="python",
        header=None
    )

    # Número total de colunas
    num_cols = df.shape[1]

    # Colunas base fixas
    base_cols = ["nSeq", "I1", "I2", "O1", "O2"]

    # Detectar canais A*
    num_A = num_cols - len(base_cols)
    A_cols = [f"A{i+1}" for i in range(num_A)]

    df.columns = base_cols + A_cols

    # Detectar sinais EMG válidos
    signals = [
        df[col].values for col in df.columns
        if col.startswith("A") and (df[col] != 0).any()
    ]

    print(f"{len(signals)} canais EMG detectados")

    # Detectar label
    label = None

    if "O1" in df.columns and (df["O1"] != 0).any():
        label = df["O1"].values
        print("Usando label digital (O1)")
        print("Label analógica vazia/inativa")

    elif "I1" in df.columns and (df["I1"] != 0).any():
        label = df["I1"].values
        print("Usando label analógica (I1)")
        print("Label digital vazia/inativa")

    else:
        print("Nenhuma label válida encontrada")

    return signals, label


def split_data(events_dict, mode="", folder_name="SubjectX", test_size=0.2, random_state=42):
    """
    Divide eventos multicanais em treino e teste.

    Returns
    -------
    X : dict
        {
            "train": np.ndarray(dtype=object),
            "test": np.ndarray(dtype=object)
        }

    y : dict
        {
            "train": np.ndarray,
            "test": np.ndarray
        }
    """
    mode = mode.lower()

    if mode not in ("benchmark", "deploy"):
        raise ValueError(
            f"Modo inválido: '{mode}'. O parâmetro 'mode' deve ser 'benchmark' ou 'deploy'."
        )

    X_data = []
    y_data = []

    for class_events in events_dict.values():
        for event in class_events:
            X_data.append(event)
            y_data.append(event["event_label"])

    X_data = np.array(X_data, dtype=object)
    y_data = np.array(y_data)

    X_train, X_test, y_train, y_test = train_test_split(
        X_data,
        y_data,
        test_size=test_size,
        random_state=random_state,
        stratify=y_data
    )

    X = {
        "train": X_train,
        "test": X_test
    }

    y = {
        "train": y_train,
        "test": y_test
    }

    split_df = visualize_split(X, y)

    #salva o pipeline inteiro
    if mode == "benchmark":
      save_path = f"{output_root()}/{folder_name}/[BENCHMARK]Zero-Phase/Data"
    elif mode == "deploy":
      save_path = f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/Data"
    os.makedirs(save_path, exist_ok=True)

    split_df.to_csv(f"{save_path}/split_details.csv", index=False)

    return X, y


def segment_dataset(X, y, fs=1000, window_ms=50, overlap=0.0):
    """
    Segmenta eventos multicanais em janelas temporais mantendo a
    separação treino/teste previamente definida em split_data.

    Parameters
    ----------
    X : dict
        Dicionário contendo os eventos já separados em treino e teste.

        Estrutura esperada:
        {
            "train": np.ndarray(dtype=object),
            "test": np.ndarray(dtype=object)
        }

        Cada evento deve possuir pelo menos:
        {
            "event_id": int,
            "signals": np.ndarray
        }

        onde signals possui shape:
        (n_channels, n_samples)

    y : dict
        Dicionário contendo os rótulos dos eventos.

        Estrutura esperada:
        {
            "train": np.ndarray,
            "test": np.ndarray
        }

    fs : int, default=1000
        Frequência de amostragem dos sinais em Hz.

    window_ms : float, default=50
        Tamanho da janela em milissegundos.

    overlap : float, default=0.0
        Sobreposição entre janelas.

        Valores típicos:
        - 0.0 = sem overlap
        - 0.5 = 50% de overlap
        - 0.75 = 75% de overlap

    Returns
    -------
    X_out : dict
        Janelas segmentadas.

        Estrutura:
        {
            "train": np.ndarray(dtype=object),
            "test": np.ndarray(dtype=object)
        }

        Cada elemento possui shape:
        (n_channels, window_size_samples)

    y_out : dict
        Labels correspondentes às janelas.

        Estrutura:
        {
            "train": np.ndarray,
            "test": np.ndarray
        }

    metadata : dict
        Metadados de cada janela gerada.

        Estrutura:
        {
            "train": list,
            "test": list
        }

        Cada item contém:
        {
            "event_id": int,
            "label": int,
            "window_id": int,
            "start_idx": int,
            "end_idx": int,
            "window_size_samples": int,
            "window_size_ms": float,
            "overlap": float,
            "n_channels": int
        }

        Descrição dos campos:

        - event_id:
            Identificador do evento original do qual a janela foi extraída.

        - label:
            Classe associada ao evento/janela.

        - window_id:
            Índice único da janela dentro do conjunto
            ("train" ou "test").

        - start_idx:
            Índice inicial da janela no sinal original.

        - end_idx:
            Índice final da janela no sinal original.

        - window_size_samples:
            Tamanho da janela em número de amostras.

        - window_size_ms:
            Tamanho da janela em milissegundos.

        - overlap:
            Taxa de sobreposição utilizada na segmentação.

        - n_channels:
            Número de canais presentes no sinal.

    Notes
    -----
    Esta função NÃO realiza train_test_split.

    A separação treino/teste deve ser realizada previamente por
    uma função como split_data().

    Dessa forma, todas as janelas derivadas de um evento permanecem
    no mesmo conjunto (treino ou teste), evitando data leakage.
    """

    win_size = int((window_ms / 1000) * fs)

    if win_size <= 0:
        raise ValueError("window_ms muito pequeno.")

    step = int(win_size * (1 - overlap))

    if step <= 0:
        raise ValueError("overlap muito alto.")

    X_out = {}
    y_out = {}
    metadata = {}

    for split in ["train", "test"]:

        X_windows = []
        y_windows = []
        meta = []

        for event, label in zip(X[split], y[split]):

            signals = event["signals"]

            n_channels, n_samples = signals.shape

            if n_samples < win_size:
                continue

            for start in range(
                0,
                n_samples - win_size + 1,
                step
            ):
                end = start + win_size

                window = signals[:, start:end]

                X_windows.append(window)
                y_windows.append(label)

                meta.append({
                    "event_id": event["event_id"],
                    "label": int(label),
                    "window_id": len(X_windows) - 1,
                    "start_idx": int(start),
                    "end_idx": int(end),
                    "window_size_samples": int(win_size),
                    "window_size_ms": float(window_ms),
                    "overlap": float(overlap),
                    "n_channels": int(n_channels)
                })

        X_out[split] = np.array(X_windows, dtype=object)
        y_out[split] = np.array(y_windows)
        metadata[split] = meta

    return X_out, y_out, metadata
