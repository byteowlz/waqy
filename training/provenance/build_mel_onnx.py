"""Build melspectrogram.onnx from first principles and prove it matches the shipped front end.

The upstream file (openWakeWord v0.5.1, re-shipped by livekit-wakeword) contains only fixed
constants: a 512-point DFT with a periodic Hann(400) window centred in the frame, hop 160,
and a librosa Slaney mel filterbank (32 bands, 60-3800 Hz), followed by 10*log10 with an 80 dB
floor. This script regenerates those constants itself, writes an equivalent graph, and checks
outputs against the upstream file on random and speech-like audio.

Usage: python -I build_mel_onnx.py <out.onnx> [<upstream melspectrogram.onnx>]
  also writes <out>_raw.onnx (no floor) for streaming runtimes
"""
from __future__ import annotations

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

N_FFT, WIN, HOP, N_MELS, SR, FMIN, FMAX = 512, 400, 160, 32, 16000, 60.0, 3800.0


def hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-12) / min_log_hz) / logstep, mels)


def mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def slaney_mel() -> np.ndarray:
    """(n_fft/2+1, n_mels) Slaney-style, area-normalised triangular filters."""
    fft_freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_freqs[None, :]
    weights = np.zeros((N_MELS, len(fft_freqs)))
    for i in range(N_MELS):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2:N_MELS + 2] - mel_f[:N_MELS]))[:, None]
    return weights.T.astype(np.float32)


def dft_kernels() -> tuple[np.ndarray, np.ndarray]:
    n = np.arange(N_FFT)
    k = np.arange(N_FFT // 2 + 1)[:, None]
    window = np.zeros(N_FFT)
    start = (N_FFT - WIN) // 2
    window[start:start + WIN] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)  # periodic Hann
    real = np.cos(2 * np.pi * k * n / N_FFT) * window
    imag = -np.sin(2 * np.pi * k * n / N_FFT) * window
    return real[:, None, :].astype(np.float32), imag[:, None, :].astype(np.float32)


def build(raw: bool = False) -> onnx.ModelProto:
    """raw=True stops before the 80 dB floor; the caller applies the floor over a trailing window."""
    real, imag = dft_kernels()
    c = lambda name, value: numpy_helper.from_array(np.asarray(value, dtype=np.float32 if not isinstance(value, np.ndarray) or value.dtype != np.int64 else np.int64), name)
    inits = [numpy_helper.from_array(real, "dft_real"), numpy_helper.from_array(imag, "dft_imag"),
             numpy_helper.from_array(slaney_mel(), "mel_filters"), numpy_helper.from_array(np.array([1], np.int64), "axis1"),
             c("eps", 1e-10), c("ten", 10.0), c("ln10", np.log(10.0)), c("two", 2.0), c("top_db", 80.0),
             c("inf", np.inf)]
    nodes = [
        helper.make_node("Unsqueeze", ["input", "axis1"], ["x"]),
        helper.make_node("Conv", ["x", "dft_real"], ["re"], kernel_shape=[N_FFT], strides=[HOP], pads=[0, 0]),
        helper.make_node("Conv", ["x", "dft_imag"], ["im"], kernel_shape=[N_FFT], strides=[HOP], pads=[0, 0]),
        helper.make_node("Unsqueeze", ["re", "axis1"], ["re4"]),
        helper.make_node("Unsqueeze", ["im", "axis1"], ["im4"]),
        helper.make_node("Transpose", ["re4"], ["re_t"], perm=[0, 1, 3, 2]),
        helper.make_node("Transpose", ["im4"], ["im_t"], perm=[0, 1, 3, 2]),
        helper.make_node("Pow", ["re_t", "two"], ["re2"]),
        helper.make_node("Pow", ["im_t", "two"], ["im2"]),
        helper.make_node("Add", ["re2", "im2"], ["power"]),
        helper.make_node("MatMul", ["power", "mel_filters"], ["mel"]),
        helper.make_node("Clip", ["mel", "eps", "inf"], ["mel_c"]),
        helper.make_node("Log", ["mel_c"], ["ln"]),
        helper.make_node("Mul", ["ln", "ten"], ["ln10x"]),
        helper.make_node("Div", ["ln10x", "ln10"], ["output" if raw else "db"]),
    ]
    if not raw:
        nodes += [helper.make_node("ReduceMax", ["db"], ["peak"], keepdims=0),
                  helper.make_node("Sub", ["peak", "top_db"], ["floor"]),
                  helper.make_node("Clip", ["db", "floor", "inf"], ["output"])]
    else:
        inits = [i for i in inits if i.name != "top_db"]
    graph = helper.make_graph(
        nodes, "melspectrogram",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch_size", "samples"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch_size", 1, "time", N_MELS])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="oqto-wake build_mel_onnx")
    model.ir_version = 7
    model.doc_string = ("Log-mel front end regenerated from formulas (periodic Hann 400 in 512-pt DFT, hop 160, "
                        "Slaney mel 32x 60-3800 Hz, 10log10" + (", no floor: apply max(x, trailing 2 s peak - 80) outside"
                        if raw else ", 80 dB floor; numerically equivalent to openWakeWord v0.5.1 melspectrogram.onnx")
                        + "). Contains no learned parameters.")
    onnx.checker.check_model(model)
    return model


def compare(ours: str, upstream: str) -> dict:
    import onnxruntime as ort
    a = ort.InferenceSession(ours, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(upstream, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    t = np.arange(16000 * 2) / 16000
    cases = {
        "white_noise": rng.normal(0, 0.1, (2, 32000)),
        "chirp_harmonics": np.stack([sum(np.sin(2 * np.pi * f * (1 + 0.3 * t) * t) / (i + 1) for i, f in enumerate((180, 360, 540, 900))) * 0.2] * 2),
        "quiet": rng.normal(0, 1e-4, (1, 20000)),
        "int16_scale": rng.normal(0, 3000, (1, 24000)),
    }
    worst = 0.0
    out = {}
    for name, audio in cases.items():
        audio = audio.astype(np.float32)
        ya = a.run(None, {"input": audio})[0]
        yb = b.run(None, {b.get_inputs()[0].name: audio})[0]
        assert ya.shape == yb.shape, (name, ya.shape, yb.shape)
        err = float(np.max(np.abs(ya - yb)))
        out[name] = {"shape": list(ya.shape), "max_abs_db_diff": err}
        worst = max(worst, err)
    out["worst_max_abs_db_diff"] = worst
    return out


if __name__ == "__main__":
    import json
    onnx.save(build(), sys.argv[1])
    raw_path = sys.argv[1].replace(".onnx", "_raw.onnx")
    onnx.save(build(raw=True), raw_path)
    print("wrote", sys.argv[1], raw_path)
    if len(sys.argv) > 2:
        print(json.dumps(compare(sys.argv[1], sys.argv[2]), indent=1))
