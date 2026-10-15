from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, RobustScaler
from tsfresh import select_features, extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
from ._paths import output_root


def _mnf_mdf_from_psd(freqs, psd):
    """
    Calcula MNF e MDF a partir do PSD
    """
    mnf = np.sum(freqs * psd) / np.sum(psd) # Frequência média, para análise geral de espectro

    cumsum = np.cumsum(psd)
    half_power = cumsum[-1] / 2
    mdf = freqs[np.where(cumsum >= half_power)[0][0]] # Frequência mediana, para fadiga muscular

    return mnf, mdf


def _window_psd_metrics(segment, fs=1000):
    """
    Calcula PSD + MNF + MDF para um segmento (janela)
    """
    freqs, psd = welch(segment, fs=fs, nperseg=min(256, len(segment))) # nperseg 256 é padrão do scipy
    mnf, mdf = _mnf_mdf_from_psd(freqs, psd)
    return mnf, mdf


def compute_metrics(events, labels, fs=1000):

    # primeiro evento válido
    first_event = None
    for label in labels:
        if len(events[label]) > 0:
            first_event = events[label][0]
            break

    if first_event is None:
        raise ValueError("Nenhum evento encontrado.")

    n_channels = first_event["signals"].shape[0]

    metrics = {
        "fft": [],
        "psd": [],
        "mnf": {
            label: [[] for _ in range(n_channels)]
            for label in labels
        },
        "mdf": {
            label: [[] for _ in range(n_channels)]
            for label in labels
        }
    }

    # =====================================================
    # FFT / PSD GLOBAL POR CANAL
    # =====================================================
    for ch in range(n_channels):

        all_segments = []

        for label in labels:
            for event in events[label]:
                all_segments.append(event["signals"][ch])

        full_signal = np.concatenate(all_segments)

        # FFT
        fft_freq = np.fft.rfftfreq(
            len(full_signal),
            d=1 / fs
        )

        fft_vals = np.fft.rfft(full_signal)

        metrics["fft"].append({
            "freq": fft_freq,
            "values": fft_vals
        })

        # PSD
        psd_freq, psd_vals = welch(
            full_signal,
            fs=fs
        )

        metrics["psd"].append({
            "freq": psd_freq,
            "values": psd_vals
        })

    # =====================================================
    # MNF / MDF POR EVENTO
    # =====================================================
    for label in labels:

        for event in events[label]:

            signals = event["signals"]

            for ch in range(n_channels):

                segment = signals[ch]

                mnf, mdf = _window_psd_metrics(
                    segment,
                    fs
                )

                metrics["mnf"][label][ch].append(mnf)
                metrics["mdf"][label][ch].append(mdf)

    return metrics


def extract_ts_features(X):
    """
    Extrai atributos temporais utilizando TSFresh para cada janela de sinal,
    preservando a separação entre os conjuntos de treino e teste.

    Cada janela é convertida para o formato longo (long format) exigido pelo
    TSFresh, onde cada canal é tratado como uma série temporal distinta
    associada a uma mesma observação (`id`). Em seguida, são extraídas as
    features utilizando `EfficientFCParameters`.

    Parameters
    ----------
    X : dict[str, array-like]
        Dicionário contendo os conjuntos de dados separados por split.

        Estrutura esperada:

        {
            "train": array-like de shape (n_windows_train,
                                         n_channels,
                                         window_size),
            "test": array-like de shape (n_windows_test,
                                        n_channels,
                                        window_size)
        }

        Cada elemento representa uma janela temporal multicanal, onde:

        - `n_channels` é o número de canais/sensores.
        - `window_size` é o número de amostras da janela.

    Returns
    -------
    X_ts : dict[str, pandas.DataFrame]
        Dicionário contendo as features extraídas para cada split.

        Estrutura:

        {
            "train": pd.DataFrame,
            "test": pd.DataFrame
        }

        Cada DataFrame possui:

        - Uma linha por janela temporal.
        - Uma coluna para cada feature calculada pelo TSFresh.
        - Índice correspondente ao identificador da janela (`w0`, `w1`, ...).

    Notes
    -----
    O processo executado para cada split consiste em:

    1. Converter as janelas para o formato longo contendo as colunas:
       `id`, `channel`, `time` e `value`.
    2. Aplicar `tsfresh.extract_features()` utilizando
       `EfficientFCParameters`.
    3. Imputar valores ausentes por meio de `tsfresh.utilities.dataframe_functions.impute`.
    4. Ordenar o índice das features resultantes.

    Examples
    --------
    >>> X = {
    ...     "train": np.random.randn(100, 3, 256),
    ...     "test": np.random.randn(20, 3, 256)
    ... }
    >>> X_ts = extract_ts_features(X)
    >>> X_ts["train"].shape
    (100, n_features)
    >>> X_ts["test"].shape
    (20, n_features)
    """

    X_ts = {}

    for split in ["train", "test"]:

        rows = []
        X_windows = X[split]

        n_windows = len(X_windows)

        # ----------------------------
        # LONG FORMAT
        # ----------------------------
        for w, window in enumerate(X_windows):

            n_channels = window.shape[0]

            for ch in range(n_channels):

                signal = window[ch]

                for t, value in enumerate(signal):

                    rows.append({
                        "id": f"w{w}",
                        "channel": str(ch),
                        "time": t,
                        "value": value
                    })

        df_long = pd.DataFrame(rows)

        # ----------------------------
        # TSFRESH EXTRACTION
        # ----------------------------
        feats = extract_features(
            df_long,
            column_id="id",
            column_sort="time",
            column_kind="channel",
            column_value="value",
            default_fc_parameters=EfficientFCParameters(),
            disable_progressbar=False
        )

        impute(feats)
        feats = feats.sort_index()

        X_ts[split] = feats

    return X_ts


def select_ts_features(features_train, y_train, mode="", folder_name="SubjectX", return_dataframe=False):
    """
    Realiza a seleção de atributos extraídos pelo TSFresh utilizando
    `tsfresh.feature_selection.select_features`.

    A função garante o alinhamento entre as amostras de entrada e os rótulos,
    executa a seleção automática de features relevantes, gera um relatório
    resumido da seleção realizada e salva esse relatório em disco.

    Parameters
    ----------
    features_train : pandas.DataFrame
        DataFrame contendo as features extraídas pelo TSFresh.
        Cada linha representa uma amostra e cada coluna uma feature.

    y_train : array-like or pandas.Series
        Vetor de rótulos correspondente às amostras de `features_train`.

    folder_name : str, optional, default="SubjectX"
        Nome da pasta onde será salvo o arquivo
        ``tsfresh_summary.txt`` contendo o resumo da seleção.

    return_dataframe : bool, optional, default=False
        Se ``True``, retorna um DataFrame contendo apenas as features
        selecionadas. Caso contrário, retorna apenas os valores em formato
        NumPy array.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        Features selecionadas após o processo de seleção.

        - Se ``return_dataframe=True``: retorna um ``pandas.DataFrame``.
        - Se ``return_dataframe=False``: retorna um ``numpy.ndarray``.

    Raises
    ------
    ValueError
        Caso o número de amostras em `features_train` seja diferente do
        número de rótulos em `y_train`.

    Notes
    -----
    O índice de `features_train` e `y_train` é redefinido internamente para
    garantir alinhamento posicional entre amostras e rótulos.

    Além do retorno, a função:

    - Exibe no console um resumo da seleção realizada.
    - Salva o resumo no diretório de saída configurado para o participante.
    - Lista as features mantidas e descartadas após a seleção.

    Examples
    --------
    >>> X_selected = select_ts_features(X_train, y_train)

    >>> X_selected_df = select_ts_features(
    ...     X_train,
    ...     y_train,
    ...     folder_name="Patient01",
    ...     return_dataframe=True
    ... )
    """
    mode = mode.lower()
    if mode not in ("benchmark", "deploy"):
        raise ValueError(
            f"Modo inválido: '{mode}'. O parâmetro 'mode' deve ser 'benchmark' ou 'deploy'."
        )

    # -------------------------
    # garantir alinhamento
    # -------------------------
    features_train = features_train.reset_index(drop=True) #elimina index "herdado", pq as vezes se eu só fazer y_train.index = features_train.index pode vir na ordem errada, então reseta e garante a posição vdd
    y_train = pd.Series(y_train).reset_index(drop=True)

    if len(features_train) != len(y_train):
        raise ValueError(
            f"Mismatch: {len(features_train)} features vs {len(y_train)} labels"
        )

    # -------------------------
    # seleção
    # -------------------------

    n_before = features_train.shape[1]

    features_selected = select_features(features_train, y_train)

    n_after = features_selected.shape[1]

    features_selected = features_selected.reset_index(drop=True)

    # -------------------------
    # logging
    # -------------------------
    features_total = list(features_train.columns)
    features_selected_columns = list(features_selected.columns)

    features_discarded = list(
        set(features_total) - set(features_selected_columns)
    )

    # -------------------------
    # texto do resumo
    # -------------------------
    summary_text = f"""
TSFRESH FEATURE SELECTION SUMMARY
=================================
Desalinhado (amostras): {"SIM" if features_train.shape[0] != features_selected.shape[0] else "NÃO"}

Features antes da seleção: {n_before}
Features depois da seleção: {n_after}
Features removidas: {n_before - n_after}

---------------------------------
FEATURES TOTAIS
---------------------------------

{features_total}

---------------------------------
FEATURES SELECIONADAS
---------------------------------

{features_selected_columns}

---------------------------------
FEATURES DESCARTADAS
---------------------------------

{features_discarded}
"""
    print(summary_text)

    # -------------------------
    # save
    # -------------------------
    if mode == "benchmark":
      save_path = f"{output_root()}/{folder_name}/[BENCHMARK]Zero-Phase/Data/TSFRESH_features"
    elif mode == "deploy":
      save_path = f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/Data/TSFRESH_features"
    os.makedirs(save_path, exist_ok=True)

    with open(f"{save_path}/tsfresh_summary.txt", "w") as f:
        f.write(summary_text)

    if not return_dataframe:
        return features_selected.values

    return features_selected


def iegm(signal):
    return np.sum(np.abs(signal))


def mav(signal):
    return np.mean(np.abs(signal))


def rms(signal):
    return np.sqrt(np.mean(signal**2))


def ssc(signal, threshold=1e-6):

    count = 0

    for i in range(1, len(signal)-1):

        diff1 = signal[i] - signal[i-1]
        diff2 = signal[i] - signal[i+1]

        if (
            (diff1 * diff2 > 0) and
            (abs(diff1) >= threshold or abs(diff2) >= threshold)
        ):
            count += 1

    return count


def var(signal):
    return np.var(signal)


def wamp(signal, threshold=0.01):
    count = 0
    for i in range(len(signal)-1):
        if np.abs(signal[i] - signal[i+1]) >= threshold:
            count += 1
    return count


def wl(signal):
    return np.sum(np.abs(np.diff(signal)))


def zc(signal, threshold=1e-6):
    count = 0
    for i in range(len(signal)-1):
        if (signal[i] * signal[i+1] < 0) and (np.abs(signal[i] - signal[i+1]) >= threshold):
            count += 1
    return count


def extract_td_features(X, y, threshold_ratio=0.01, features=["IEMG", "MAV", "RMS", "VAR", "WL", "ZC", "SSC", "WAMP"]):
    """
    Extrai features temporais de janelas multicanais mantendo a
    separação treino/teste.

    Parameters
    ----------
    X : dict
        Dicionário contendo janelas segmentadas.

        Estrutura:
        {
            "train": np.ndarray(dtype=object),
            "test": np.ndarray(dtype=object)
        }

        Cada janela possui shape:
            (n_channels, window_size)

    y : dict
        Dicionário contendo os labels das janelas.

        Estrutura:
        {
            "train": np.ndarray,
            "test": np.ndarray
        }

    threshold_ratio : float, default=0.01
        Threshold adaptativo utilizado pelas features:
        ZC, SSC e WAMP.

    features : list
        Lista de features a serem extraídas.

        Opções:
        ["IEMG", "MAV", "RMS", "VAR",
         "WL", "ZC", "SSC", "WAMP"]

    Returns
    -------
    X_features : dict
        Features extraídas.

        Estrutura:
        {
            "train": pd.DataFrame,
            "test": pd.DataFrame
        }

        Cada coluna segue o padrão:

            FEATURE_CHN

        Exemplo:

            MAV_CH1
            RMS_CH1
            WL_CH2

    y_features : dict
        Labels correspondentes às features.

        Estrutura:
        {
            "train": np.ndarray,
            "test": np.ndarray
        }
    """

    feature_rows = []

    for window in X:

        row = {}

        n_channels = window.shape[0]

        for ch in range(n_channels):

            sig = np.asarray(window[ch])

            thr = threshold_ratio * np.max(np.abs(sig))

            feats = {}

            if "IEMG" in features:
                feats["IEMG"] = iegm(sig)

            if "MAV" in features:
                feats["MAV"] = mav(sig)

            if "RMS" in features:
                feats["RMS"] = rms(sig)

            if "VAR" in features:
                feats["VAR"] = var(sig)

            if "WL" in features:
                feats["WL"] = wl(sig)

            if "ZC" in features:
                feats["ZC"] = zc(sig, thr)

            if "SSC" in features:
                feats["SSC"] = ssc(sig, thr)

            if "WAMP" in features:
                feats["WAMP"] = wamp(sig, thr)

            for feat_name, feat_value in feats.items():
                row[f"{feat_name}_CH{ch+1}"] = feat_value

        feature_rows.append(row)

    X_features = pd.DataFrame(feature_rows)
    y_features = np.array(y)

    return X_features, y_features


def build_featureset(
    X,
    y,
    return_dataframe=False,
    extraction_method="time_domain",
    mode="",
    features=["IEMG", "MAV", "RMS", "VAR", "WL", "ZC", "SSC", "WAMP"],
    folder_name="SubjectX"
):
    """
    Constrói o dataset final de features mantendo separação treino/teste.

    Parameters
    ----------
    X : dict
        Janelas segmentadas.

        Estrutura:
        {
            "train": np.ndarray(dtype=object),
            "test": np.ndarray(dtype=object)
        }

    y : dict
        Labels das janelas.

        Estrutura:
        {
            "train": np.ndarray,
            "test": np.ndarray
        }

    return_dataframe : bool
        Se True, retorna DataFrames.
        Se False, retorna numpy arrays.

    extraction_method : str
        Tipo de features:
        - "time_domain"
        - "tsfresh"

    features : list
        Features do domínio do tempo.

    Returns
    -------
    X_features : dict
        {
            "train": DataFrame ou ndarray,
            "test": DataFrame ou ndarray
        }

    y_features : dict
        {
            "train": np.ndarray,
            "test": np.ndarray
        }
    """
    mode = mode.lower()
    extraction_method = extraction_method.lower()

    if mode not in ("benchmark", "deploy"):
        raise ValueError(
            f"Modo inválido: '{mode}'. O parâmetro 'mode' deve ser 'benchmark' ou 'deploy'."
        )

    if extraction_method == "tsfresh":
        extraction_method_path = "TSFRESH"
    elif extraction_method == "time_domain":
        extraction_method_path = "TD"

    X_features = {}
    y_features = {}

    if extraction_method == "tsfresh":
        X_ts = extract_ts_features(X)

    for split in ["train", "test"]:

        # =========================
        # TIME DOMAIN (NÃO ALTERADO)
        # =========================
        if extraction_method == "time_domain":

            feats_df, labels = extract_td_features(
                X=X[split],
                y=y[split],
                features=features
            )

        # =========================
        # TSFRESH (CORRIGIDO)
        # =========================
        elif extraction_method == "tsfresh":
            # feats_df = extract_ts_features(X)[split]
            # feats_df = pd.DataFrame(feats_df)
            # feats_df = pd.DataFrame(X_ts)
            feats_df = X_ts[split]
            labels = y[split]
        else:
            raise ValueError("extraction_method deve ser 'time_domain' ou 'tsfresh'")

        # =========================
        # LONG → WIDE (TSFRESH apenas)  removi porque não faz muito sentido, quando uso o extract do tsfresh ele ja vem nesse formato
        # =========================
        if extraction_method == "tsfresh":

            if "channel" in feats_df.columns and "window" in feats_df.columns:

                feats_df = feats_df.pivot(
                    index="window",
                    columns="channel"
                )

                feats_df.columns = [
                    f"{feat}_chn{ch}"
                    for feat, ch in feats_df.columns
                ]

                feats_df = feats_df.reset_index(drop=True)

        # =========================
        # GUARDA SPLIT
        # =========================
        X_features[split] = feats_df
        y_features[split] = labels

    # =========================
    # TSFRESH FEATURE SELECTION (CORRETO)
    # =========================
    if extraction_method == "tsfresh":

        # Verificação se os indices estão batendo
        assert len(X_features["train"]) == len(y_features["train"]), (
            f"Train: {len(X_features['train'])} amostras "
            f"e {len(y_features['train'])} labels"
        )

        assert len(X_features["test"]) == len(y_features["test"]), (
            f"Test: {len(X_features['test'])} amostras "
            f"e {len(y_features['test'])} labels"
        )

        # -------------------------
        # SELECT SOMENTE NO TRAIN
        # -------------------------
        X_train_sel = select_ts_features(
            X_features["train"],
            y_features["train"],
            mode=mode,
            folder_name=folder_name,
            return_dataframe=True
        )

        # -------------------------
        # ALINHAR TEST COM TRAIN
        # -------------------------
        X_test_sel = X_features["test"].reindex(
            columns=X_train_sel.columns,
            fill_value=0
        )

        X_features = {
            "train": X_train_sel,
            "test": X_test_sel
        }

    # =========================
    # EXPORTA FEATURES PARA CSV
    # =========================
    if mode == "benchmark":
        save_path = (
            f"{output_root()}/"
            f"{folder_name}/[BENCHMARK]Zero-Phase/Data/{extraction_method_path}_features"
        )
    elif mode == "deploy":
        save_path = (
            f"{output_root()}/"
            f"{folder_name}/[DEPLOY]Causal_IIR/Data/{extraction_method_path}_features"
        )

    os.makedirs(save_path, exist_ok=True)

    # DataFrame de treino
    df_train = X_features["train"].copy()
    df_train["label"] = y_features["train"]
    df_train["split"] = "train"

    # DataFrame de teste
    df_test = X_features["test"].copy()
    df_test["label"] = y_features["test"]
    df_test["split"] = "test"

    # Train primeiro, test depois
    df_features = pd.concat([df_train, df_test], ignore_index=True)

    output_file = os.path.join(save_path, "features.csv")
    df_features.to_csv(output_file, index=False)

    # print(f"Features de treino salvas em:\n{train_file}")
    # print(f"Features de teste salvas em:\n{test_file}")

    # =========================
    # CONVERSÃO FINAL OPCIONAL
    # =========================
    if not return_dataframe:
        X_features["train"] = X_features["train"].values
        X_features["test"] = X_features["test"].values

    return X_features, y_features


def normalize_data(X, y, mode="", folder_name="SubjectX", extraction_method="", scale_y=False, verbose=True):
    """
    Normaliza os dados usando apenas estatísticas do conjunto de treino.

    Parameters
    ----------
    X : dict
        {'train': X_train, 'test': X_test}

    y : dict, optional
        {'train': y_train, 'test': y_test}

    scale_y : bool, default=False
        Se True, normaliza y com StandardScaler.

    Returns
    -------
    X_norm : dict
        {'train': X_train_scaled, 'test': X_test_scaled}

    y_norm : dict or None
        {'train': y_train_scaled, 'test': y_test_scaled}
        ou None se scale_y=False
        -> Normalmente só necessário em problemas de REGRESSÃO.

    scaler : sklearn scaler
        Scaler utilizado em X
    """
    mode = mode.lower()
    extraction_method = extraction_method.lower()

    if extraction_method not in ("time_domain", "tsfresh"):
        raise ValueError("extraction_method deve ser 'time_domain' ou 'tsfresh'")

    if extraction_method == "tsfresh":
        extraction_method_path = "TSFRESH"
    elif extraction_method == "time_domain":
        extraction_method_path = "TD"

    if mode not in ("benchmark", "deploy"):
        raise ValueError(
            f"Modo inválido: '{mode}'. O parâmetro 'mode' deve ser 'benchmark' ou 'deploy'."
        )

    X_train = X["train"]
    X_test = X["test"]

    # 1. medir skew apenas no treino
    skew_values = skew(X_train, axis=0, nan_policy="omit")
    mean_skew = np.mean(np.abs(skew_values))

    if verbose:
        print(f"\nSkewness média absoluta: {mean_skew:.4f}")

    # 2. escolher scaler
    if mean_skew > 0.5:
        scaler = RobustScaler()
        if verbose:
            print("Usando RobustScaler (dados assimétricos)")
    else:
        scaler = StandardScaler()
        if verbose:
            print("Usando StandardScaler (distribuição mais normal)")

    # 3. normalizar X
    X_norm = {
        "train": scaler.fit_transform(X_train),
        "test": scaler.transform(X_test),
        "scaler": scaler
    }

    # 4. normalizar y (opcional)
    y_norm = None

    if mode == "benchmark":
      save_path = f"{output_root()}/{folder_name}/[BENCHMARK]Zero-Phase/Data/{extraction_method_path}_features"
    elif mode == "deploy":
      save_path = f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/Data/{extraction_method_path}_features"
    os.makedirs(save_path, exist_ok=True)

    feature_names = X["train"].columns
    df_train = pd.DataFrame(X_norm["train"], columns=feature_names)
    df_test = pd.DataFrame(X_norm["test"], columns=feature_names)
    df_train["split"] = "train"
    df_test["split"] = "test"

    if y is not None and scale_y:
        y_scaler = StandardScaler()

        y_norm = {
            "train": y_scaler.fit_transform(
                np.asarray(y["train"]).reshape(-1, 1)
            ).ravel(),

            "test": y_scaler.transform(
                np.asarray(y["test"]).reshape(-1, 1)
            ).ravel()
        }
        # treino primeiro, teste abaixo
        df_train["label"] = y_norm["train"]
        df_test["label"] = y_norm["test"]
        cols = list(feature_names) + ["label", "split"]
        df_train = df_train[cols]
        df_test = df_test[cols]

        df_norm = pd.concat([df_train, df_test], ignore_index=True)
        df_norm.to_csv(f"{save_path}/normalized_features.csv", index=False)

        if verbose:
            print("y também foi normalizado (StandardScaler)")
        return X_norm, y_norm, y_scaler

    # y fornecido mas sem normalização
    # if y is not None:
    #     df_train["label"] = y["train"]
    #     df_test["label"] = y["test"]
    #     cols = list(feature_names) + ["label", "split"]
    #     df_train = df_train[cols]
    #     df_test = df_test[cols]

    #     df_norm = pd.concat([df_train, df_test], ignore_index=True)
    #     df_norm.to_csv(f"{save_path}/normalized_features.csv", index=False)

    #     return X_norm, y

    df_train["label"] = y["train"]
    df_test["label"] = y["test"]
    cols = list(feature_names) + ["label", "split"]
    df_train = df_train[cols]
    df_test = df_test[cols]

    df_norm = pd.concat([df_train, df_test], ignore_index=True)
    df_norm.to_csv(f"{save_path}/normalized_features.csv", index=False)
    # df_norm = pd.concat([df_train, df_test], ignore_index=True)
    # df_norm.to_csv(f"{save_path}/normalized_features.csv", index=False)

    return X_norm
