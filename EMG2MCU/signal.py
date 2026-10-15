from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, hilbert, tf2sos, sosfilt
from ._paths import output_root


def emg_filter(signals,
               label=None,
               mode="benchmark",
               folder_name="SubjectX",
               fs=1000,
               lowcut=30,
               highcut=450,
               notch_freqs=(60,),
               Q=30,
               padding=0.2,
               butter_order=4):

    mode = mode.lower()

    if mode not in ("benchmark", "deploy"):
        raise ValueError(
            f"Modo inválido: '{mode}'. Use 'benchmark' ou 'deploy'."
        )

    signals = np.atleast_2d(signals).astype(np.float32)

    raw_all = []
    filtered_all = []
    label_aligned = None
    cut = int(padding * fs)

    # =====================================================
    # BENCHMARK (Ground Truth - Zero Phase)
    # =====================================================
    if mode == "benchmark":

        nyq = fs / 2

        for ch in range(signals.shape[0]):

            signal = signals[ch].copy()

            # Notch
            for f0 in notch_freqs:
                b, a = iirnotch(f0 / nyq, Q)
                signal = filtfilt(b, a, signal, method="gust")

            # Butterworth
            b, a = butter(
                butter_order,
                [lowcut / nyq, highcut / nyq],
                btype="band"
            )

            signal = filtfilt(b, a, signal, method="gust")

            if 2 * cut < len(signal):
                raw = signals[ch][cut:-cut]
                filtered = signal[cut:-cut]
            else:
                raw = signals[ch]
                filtered = signal

            raw_all.append(raw)
            filtered_all.append(filtered)

        save_path = (
            f"{output_root()}/{folder_name}/[BENCHMARK]Zero-Phase/Data"
        )

        sos = None
        filter_specs = None

    # =====================================================
    # DEPLOY (Causal SOS)
    # =====================================================
    else:

        filter_specs = f"""
// =====================================================
// FILTER SPECS
// =====================================================
// {butter_order}-order Butterworth passband filter
// Sampling frequency = {fs} Hz
// Lowcut = {lowcut} Hz
// Highcut = {highcut} Hz
// Notch filter at frequencies = {notch_freqs}
// Notch quality factor (Q) = {Q}
""".strip()

        sos_band = butter(
            butter_order,
            [lowcut, highcut],
            fs=fs,
            btype="bandpass",
            output="sos"
        )

        sos_list = [sos_band]

        for f0 in notch_freqs:
            b, a = iirnotch(f0, Q=Q, fs=fs)
            sos_list.append(tf2sos(b, a))

        sos = np.vstack(sos_list)

        for ch in range(signals.shape[0]):

            signal = signals[ch]

            filtered = sosfilt(sos, signal)

            if 2 * cut < len(filtered):
                raw = signal[cut:-cut]
                filtered = filtered[cut:-cut]
            else:
                raw = signal
                filtered = filtered

            raw_all.append(raw)
            filtered_all.append(filtered)

        save_path = (
            f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/Data"
        )

    # =====================================================
    # LABEL
    # =====================================================
    if label is not None:
        if 2 * cut < len(label):
            label_aligned = label[cut:-cut]
        else:
            label_aligned = label

    # =====================================================
    # EXPORT CSV
    # =====================================================
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame()

    for i in range(len(filtered_all)):
        df[f"raw_ch{i+1}"] = raw_all[i]
        df[f"filtered_ch{i+1}"] = filtered_all[i]

    if label_aligned is not None:
        df["label"] = label_aligned

    file_path = f"{save_path}/filtered_signal.csv"
    df.to_csv(file_path, index=False)

    print(f"CSV salvo em:\n{file_path}")

    return raw_all, filtered_all, label_aligned, sos, filter_specs


def sos_to_biquad_c(sos):
    """
    Converte SOS (SciPy) para formato ESP32:
    {b0, b1, b2, -a1, -a2}
    """
    coeffs = []

    for section in sos:
        b0, b1, b2, a0, a1, a2 = section

        # normaliza (a0 geralmente = 1, ESP32 assume que sim, portanto, normalizar)
        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0

        coeffs.append([b0, b1, b2, -a1, -a2])

    return np.array(coeffs)


def emu_esp32(signal, coeffs):
    w = np.zeros((len(coeffs), 2))
    y_out = np.zeros_like(signal, dtype=np.float32)

    for n in range(len(signal)):
        x = signal[n]
        y = x

        for i, c in enumerate(coeffs):
            b0, b1, b2, a1, a2 = c

            # diferença (forma direta do biquad)
            y0 = b0 * y + w[i][0]

            w[i][0] = b1 * y + w[i][1] + a1 * y0
            w[i][1] = b2 * y + a2 * y0

            y = y0

        y_out[n] = y

    return y_out


def evaluate_error(filtered_py, filtered_c):
    """
    Sinais EMG costumam estar na ordem de:
        ~1e-3 a 1e-1 (dependendo da normalização)
    Então:
        erro 1e-6 → praticamente irrelevante
        erro 1e-4 → pequeno, mas pode aparecer
        erro > 1e-3 → já começa a distorcer
    """
    error = np.abs(filtered_py - filtered_c)

    max_error = np.max(error)
    mean_error = np.mean(error)

    # --- critérios ---
    if mean_error <= 1e-6 and max_error <= 1e-5:
        status = "ALTA"
        mean_tolerance = 1e-6
        max_tolerance = 1e-5

    elif mean_error <= 1e-4 and max_error <= 1e-4:
        status = "TOLERÁVEL"
        mean_tolerance = 1e-4
        max_tolerance = 1e-4

    else:
        status = "NÃO TOLERÁVEL"
        mean_tolerance = "N/A"
        max_tolerance = "N/A"

    error_report = f"""
CORRESPONDÊNCIA (SCIPY x ESP32) DO FILTRO: {status}

Max error : {max_error:.3e} <= {max_tolerance}
Mean error: {mean_error:.3e} <= {mean_tolerance}
"""

    return error_report, max_error, mean_error


def debug_esp32_filter(signals, sos_configs, folder_name="SubjectX"):
    """
    Aplica filtro SOS em cada canal e compara:
        SciPy vs ESP32 simulado

    Parameters
    ----------
    signals : list[np.array]
        Lista com sinais raw sem padding de cada canal, i.e., o sinal importado no passo 1.

    sos_configs : list[np.array]
        [sos1, sos2, ...]

    Returns
    -------
    results : list[dict]
        Resultados por canal.
    """
    # ---------------------------------------------------
    # AJUSTE SOS
    # ---------------------------------------------------

    if isinstance(sos_configs, np.ndarray) and sos_configs.ndim == 2:
        sos_configs = [sos_configs] * len(signals)

    results = []

    # ---------------------------------------------------
    # STRING FINAL
    # ---------------------------------------------------

    filter_comparison = ""

    # ---------------------------------------------------
    # LOOP DOS CANAIS
    # ---------------------------------------------------

    for i, raw_signal in enumerate(signals):

        if isinstance(sos_configs[i], dict):
            sos = sos_configs[i]["sos"]
        else:
            sos = sos_configs[i]

        # -----------------------------------------------
        # CONVERTE PARA FORMATO ESP32
        # -----------------------------------------------

        coeffs_c = sos_to_biquad_c(sos)

        # -----------------------------------------------
        # FILTRO SCIPY
        # -----------------------------------------------

        filtered_py = sosfilt(sos, raw_signal)

        # -----------------------------------------------
        # FILTRO ESP32
        # -----------------------------------------------

        filtered_c = emu_esp32(
            raw_signal.astype(np.float32),
            coeffs_c
        )

        # -----------------------------------------------
        # ERRO
        # -----------------------------------------------

        error_report, max_error, mean_error = evaluate_error(
            filtered_py,
            filtered_c
        )

        # -----------------------------------------------
        # ACUMULA STRING
        # -----------------------------------------------

        filter_comparison += f"""
========================================================================
{'Canal ' + str(i):^72}
========================================================================

=== COEFICIENTES ESP32 ===
{coeffs_c}

=== SAÍDA SCIPY ===
{filtered_py[:10]}

=== SAÍDA ESP32 ===
{filtered_c[:10]}

Max error : {max_error}
Mean error: {mean_error}

{error_report}

"""

        # -----------------------------------------------
        # RESULTS
        # -----------------------------------------------

        results.append({
            "channel": i,
            "filtered_py": filtered_py,
            "filtered_c": filtered_c,
            "max_error": max_error,
            "mean_error": mean_error
        })

    # ---------------------------------------------------
    # PRINT FINAL
    # ---------------------------------------------------

    print(filter_comparison)

    # ---------------------------------------------------
    # SAVE TXT
    # ---------------------------------------------------

    save_path = f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/esp32_filter"

    os.makedirs(save_path, exist_ok=True)

    file_path = f"{save_path}/esp32_scipy_comparison_.txt"

    with open(file_path, "w") as f:
        f.write(filter_comparison)

    print(f"\nTXT salvo em:\n{file_path}")

    return results


def generate_esp32_code(sos, fs=1000, adc_pin=34, folder_name="SubjectX", filter_specs=""):
    """
    Generate an ESP32 Arduino (.ino) firmware implementing an EMG IIR filter
    based on a SciPy SOS (Second-Order Sections) representation.

    The function converts the input SOS matrix into a cascaded Direct Form II
    Transposed (DF2T) biquad implementation suitable for real-time execution
    on ESP32 microcontrollers.

    Parameters
    ----------
    sos : array-like (n_sections, 6)
        Second-order sections representation of the filter (SciPy format).
        Each row is expected to follow:
        [b0, b1, b2, a0, a1, a2] where a0 is typically 1.

    fs : int, optional (default=1000)
        Sampling frequency in Hz used to configure timing in the ESP32 loop.
        Must match the sampling rate used during filter design.

    adc_pin : int, optional (default=34)
        GPIO pin used for analog input (EMG signal acquisition).

    Returns
    -------
    None
        The function writes a complete Arduino .ino file under the configured
        output directory's [DEPLOY]Causal_IIR/esp32_filter folder.

    Notes
    -----
    - The SOS matrix is converted internally to DF2T biquad coefficients.
    - The implementation uses fixed-rate sampling via micros() for stability.
    - ADC is configured to 12-bit resolution with 11dB attenuation.
    - Input signal is normalized to approximately [-1, 1] to improve
      numerical stability and avoid DC bias issues in the IIR filter.
    - This implementation is intended for validation against SciPy's sosfilt.

    Warning
    -------
    This implementation prioritizes numerical equivalence with SciPy over
    hardware-specific optimizations. It does NOT use ESP-DSP libraries.

    """

    coeffs = sos_to_biquad_c(sos)
    num_sections = len(coeffs)

    c_code = ""

    # =========================================================
    # CONFIGURATION
    # =========================================================

    c_code += f"#define NUM_SECTIONS {num_sections}\n"
    c_code += f"#define EMG_PIN {adc_pin}\n"
    c_code += f"#define FS {fs}\n"
    c_code += f"#define TS_US (1000000 / FS)\n\n"

    # =========================================================
    # SOS COEFFICIENTS (DF2T FORMAT)
    # =========================================================

    c_code += f"{filter_specs}\n\n"
    c_code += "const float coeffs[NUM_SECTIONS][5] = {\n"

    for c in coeffs:
        c_code += (
            f"    {{"
            f"{c[0]:.10f}f, "
            f"{c[1]:.10f}f, "
            f"{c[2]:.10f}f, "
            f"{c[3]:.10f}f, "
            f"{c[4]:.10f}f}},\n"
        )

    c_code += "};\n\n"

    # =========================================================
    # FILTER STATES (DF2T MEMORY)
    # =========================================================

    c_code += "static float w[NUM_SECTIONS][2] = {0};\n\n"

    # =========================================================
    # TIMING CONTROL
    # =========================================================

    c_code += "uint32_t last_t = 0;\n"

    # =========================================================
    # FILTER FUNCTION (DIRECT FORM II TRANSPOSED)
    # =========================================================

    c_code += r"""
float biquad_process(float x) {

    float y = x;

    // Cascade all SOS sections
    for (int i = 0; i < NUM_SECTIONS; i++) {

        float w0 =
            y
            - coeffs[i][3] * w[i][0]
            - coeffs[i][4] * w[i][1];

        y =
            coeffs[i][0] * w0
            + coeffs[i][1] * w[i][0]
            + coeffs[i][2] * w[i][1];

        w[i][1] = w[i][0];
        w[i][0] = w0;
    }

    return y;
}

void setup() {

    Serial.begin(115200);

    // =====================================================
    // ADC CONFIGURATION (IMPORTANT FOR EMG QUALITY)
    // =====================================================

    // ESP32 ADC resolution:
    // 12 bits -> 0 to 4095
    analogReadResolution(12);

    // ADC attenuation:
    // ADC_11db -> allows full range up to ~3.3V
    // critical for EMG signal conditioning
    analogSetAttenuation(ADC_11db);

    // Optional: stabilize ADC before start
    delay(1000);

    Serial.println("ESP32 EMG DF2T filter initialized");
}

void loop() {

    // =========================================================
    // FIXED SAMPLING RATE CONTROL
    // =========================================================
    // Ensures consistent DSP behavior (critical for IIR filters)
    if ((uint32_t)(micros() - last_t) >= TS_US) {

        last_t += TS_US;

        // =====================================================
        // RAW ADC READING
        // =====================================================

        int adc = analogRead(EMG_PIN);

        // =====================================================
        // SIGNAL NORMALIZATION
        // Convert ADC [0..4095] -> [-1..1]
        //
        // WHY:
        // - avoids DC bias inside IIR filter
        // - improves numerical stability
        // - matches DSP assumptions better
        // =====================================================

        float x =
            ((float)adc - 2048.0f)
            / 2048.0f;

        // =====================================================
        // FILTER PROCESSING (DF2T CASCADE SOS)
        // =====================================================

        float y = biquad_process(x);

        // =====================================================
        // SERIAL OUTPUT (DEBUG / VALIDATION)
        // Format: raw,filtered
        // =====================================================

        Serial.print(x, 6);
        Serial.print(",");
        Serial.println(y, 6);
    }
}
"""

    print(c_code)
    save_path = f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/esp32_filter"
    os.makedirs(save_path, exist_ok=True)
    file_path = f"{save_path}/esp32_filter.ino"

    with open(file_path, "w") as f:
      f.write(c_code)


def envelope_smooth(input1, window_size, mode='valid'):
    # Sinal analítico (Hilbert transform)
    analytic_signal = hilbert(input1)

    # Envelope (magnitude)
    envelope = np.abs(analytic_signal)

    # Filtro de média móvel (convolução)
    kernel = np.ones(window_size) / window_size
    smoothed_envelope = np.convolve(envelope, kernel, mode=mode)

    return smoothed_envelope


def estimate_baseline_envelope(envelope_signal):
    # Converte a entrada (lista, array, etc.) para um array NumPy
    x = np.asarray(envelope_signal)

    # Calcula o percentil de 10% (valores mais baixos)
    p10 = np.percentile(x, 10)

    # Calcula o percentil de 90% (valores mais altos)
    p90 = np.percentile(x, 90)

    # Remove valores extremos:
    # mantém apenas os valores entre o 10º e o 90º percentil
    # (isso reduz influência de picos e ruídos)
    trimmed = x[(x >= p10) & (x <= p90)]

    # Retorna a mediana dos valores filtrados
    # (estimativa robusta do "baseline" do envelope)
    return np.median(trimmed)


def detect_onset(results,                 # Sinal filtrado
                 folder_name="SubjectX",  # Nome da pasta
                 mode="",                 # Modo benchmark ou deploy
                 fs=1000,                 # Frequência de amostragem do sinal (Hz)
                 window_size=500,         # Transformada Hilbert-Huang
                 sustain_ms=1500,         # Tempo mínimo sustentado acima do threshold
                 tolerance_ratio=0.20,    # Tolerância de falhas dentro da janela
                 plot_result=True):      # Mostrar gráfico

    mode = mode.lower()
    if mode not in ("benchmark", "deploy"):
        raise ValueError(
            f"Modo inválido: '{mode}'. O parâmetro 'mode' deve ser 'benchmark' ou 'deploy'."
        )

    if mode == "benchmark":
      save_path = f"{output_root()}/{folder_name}/[BENCHMARK]Zero-Phase/Plots/Onset"
    elif mode == "deploy":
      save_path = f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/Plots/Onset"
    os.makedirs(save_path, exist_ok=True)

    onsets = []

    for idx, signal in enumerate(results):

        # 1. normalização
        signal = signal / np.max(np.abs(signal))

        # 2. envelope
        env = envelope_smooth(signal, window_size=window_size)

        # 3. threshold
        threshold = estimate_baseline_envelope(env)

        above = env > threshold

        # 4. janela sustentada
        sustain_samples = int(fs * sustain_ms / 1000)

        # tolerância: quantos "zeros" posso aceitar
        max_gaps = int(sustain_samples * tolerance_ratio)

        onset = None

        for i in range(len(above) - sustain_samples):

            window = above[i:i + sustain_samples]

            # Quantas amostras dentro da janela estão ABAIXO do threshold
            gaps = np.sum(~window)

            if gaps <= max_gaps:
                onset = i
                break

        # ---------------- PLOT ----------------
        if plot_result:

            x = np.arange(len(env)) / fs

            plt.figure(figsize=(15, 4))

            plt.plot(x, env, label="Envelope")

            plt.axhline(
                threshold,
                color='g',
                linestyle='--',
                label="Threshold"
            )

            if onset is not None:

                plt.axvline(
                    onset / fs,
                    color='r',
                    alpha=0.5,
                    label="Onset"
                )

            plt.title(f"Canal {idx}")

            plt.xlabel("Tempo (s)")
            plt.ylabel("Amplitude")

            plt.grid(True)
            plt.legend()

            plt.tight_layout()

            plt.savefig(f"{save_path}/onset_ch{idx+1}.svg",
                        format="svg",
                        bbox_inches="tight"
            )

            plt.show()

        # --------------------------------------

        onsets.append({
            "onset": onset
        })

    return onsets


def analyze_onsets(onsets,
                   zero_threshold=10,# Valor (# de amostras) máximo para considerar que um onset está "próximo de zero"
                   tolerance=0.3): # Desvio relativo máximo permitido em relação à mediana antes de considerar um onset como outlier

    # extrai os valores
    valores = np.array([o["onset"] for o in onsets])

    # remove possíveis None
    valores = valores[valores != None]

    # verifica se algum está próximo de 0
    tem_proximo_zero = np.any(valores <= zero_threshold)

    # calcula mediana
    mediana = np.median(valores)

    # calcula desvio relativo
    desvio_relativo = np.abs(valores - mediana) / (mediana + 1e-9)

    # máscara de valores válidos (não outliers)
    mask_validos = desvio_relativo <= tolerance

    # verifica se há outlier
    tem_outlier = np.any(~mask_validos)

    # filtra valores válidos
    valores_validos = valores[mask_validos]

    # pega o menor valor válido
    # (mais próximo de zero sem ser outlier)
    if len(valores_validos) > 0:

        onset_min_valido = np.min(valores_validos)

    else:
        # caso extremo: todos são outliers
        onset_min_valido = None

    # -------- PRINTS --------

    print("Valores encontrados:", valores)

    print("Mediana:", mediana)

    print("Tem onset próximo de zero?",
          tem_proximo_zero)

    print("Tem outlier?",
          tem_outlier)

    print("Valores válidos:",
          valores_validos)

    print("Onset mínimo válido:",
          onset_min_valido)

    # ------------------------

    return onset_min_valido


def remove_standby(filtered_signals,   # sinais filtrados
                   raw_signals,        # sinais brutos
                   labels,             # labels alinhadas
                   onset_min,          # onset mínimo válido
                   pre_onset_ratio=0.25,
                   plot_result=True):

    # mantém uma porcentagem do standby antes do onset
    standby = onset_min

    cut = int((1 - pre_onset_ratio) * standby)

    print("Cut:", cut)

    # corta labels
    labels_cut = labels[cut:]

    # corta sinais brutos
    raw_cut = [r[cut:] for r in raw_signals]

    # corta sinais filtrados
    filtered_cut = [sig[cut:] for sig in filtered_signals]

    print("Novo tamanho:", len(filtered_cut[0]))

    # -------- PLOT --------

    if plot_result:

        for idx, sig in enumerate(filtered_cut):

            plt.figure(figsize=(15, 4))

            plt.plot(sig, label="Filtered")

            plt.plot(
                labels_cut * np.max(np.abs(sig)),
                label="Label",
                alpha=0.7
            )

            plt.title(f"Canal {idx}")

            plt.legend()
            plt.grid(True)

            plt.show()

    # ----------------------

    return raw_cut, filtered_cut, labels_cut


def find_events(signal, label, class_labels, class_names=None, fs=1000, pad_ms=0):
    """
    Encontra eventos multicanais no sinal a partir das labels.

    Parameters
    ----------
    signal : np.ndarray

        Shape:
            (n_channels, n_samples)

        ou:
            (n_samples,) para 1 canal

    label : np.ndarray

        Labels do sinal.

        Shape:
            (n_samples,)

    class_labels : list

        Lista de labels alvo.

        Ex:
            [0, 1]

    class_names : list or None

        Nome das classes.

        Ex:
            ["relax", "contraction"]

        Se None:
            ["class_0", "class_1", ...]

    fs : int

        Frequência de amostragem (Hz)

    pad_ms : int, float, list ou np.ndarray

        Pode ser:

        - escalar:
            mesmo padding para todas classes

        - lista:
            padding específico por classe

    Returns
    -------
    events_dict : dict

        Estrutura:

        {
            "relax": [

                {
                    "signals": np.ndarray,
                        Shape:
                            (n_channels, n_samples_event)

                    "event_id": int,

                    "start_event": int,
                    "end_event": int,

                    "start_idx": int,
                    "end_idx": int,

                    "event_label": int,
                    "event_name": str,

                    "event_duration_samples": int,
                    "event_duration_ms": float
                },

                ...
            ]
        }
    """

    signal = np.asarray(signal)
    label = np.asarray(label)

    # ==================================================
    # garante shape (n_channels, n_samples)
    # ==================================================
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    n_channels, n_samples = signal.shape

    # ==================================================
    # nomes padrão
    # ==================================================
    if class_names is None:
        class_names = [f"class_{c}" for c in class_labels]

    if len(class_labels) != len(class_names):
        raise ValueError(
            "class_labels e class_names precisam ter mesmo tamanho"
        )

    # ==================================================
    # pad por classe
    # ==================================================
    if np.isscalar(pad_ms):

        pad_list = [pad_ms] * len(class_labels)

    else:

        if len(pad_ms) != len(class_labels):
            raise ValueError(
                "pad_ms deve ter mesmo tamanho de class_labels"
            )

        pad_list = pad_ms

    # ==================================================
    # dicionário final
    # ==================================================
    events_dict = {}

    # ==================================================
    # percorre classes
    # ==================================================
    for target_label, class_name, class_pad in zip(
        class_labels,
        class_names,
        pad_list
    ):

        # padding em samples
        pad = int((class_pad / 1000) * fs)

        # ==============================================
        # máscara da classe
        # ==============================================
        mask = (label == target_label).astype(int)

        diff = np.diff(mask)

        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        # evento começa no início
        if mask[0] == 1:
            starts = np.insert(starts, 0, 0)

        # evento termina no final
        if mask[-1] == 1:
            ends = np.append(ends, len(mask))

        class_events = []

        # ==============================================
        # percorre eventos
        # ==============================================
        for event_id, (s, e) in enumerate(zip(starts, ends)):

            start_idx = max(0, s - pad)
            end_idx = min(n_samples, e + pad)

            # ==========================================
            # extrai TODOS os canais juntos
            # ==========================================
            event_signals = signal[:, start_idx:end_idx]

            duration_samples = e - s
            duration_ms = (duration_samples / fs) * 1000

            class_events.append({

                # multicanal
                "signals": event_signals,

                # id do evento
                "event_id": int(event_id),

                # índices do evento real
                "start_event": int(s),
                "end_event": int(e),

                # índices incluindo padding
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),

                # infos da classe
                "event_label": int(target_label),
                "event_name": str(class_name),

                # duração
                "event_duration_samples": int(duration_samples),
                "event_duration_ms": float(duration_ms),
            })

        events_dict[class_name] = class_events

    return events_dict
