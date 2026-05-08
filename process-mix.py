#!/usr/bin/env python

"""
DJ Mix Post-Processing Script
==============================
Applies the following chain to a WAV file:
  1. EQ       - High-pass at 30 Hz, notch cut at 275 Hz, shelf boost at 11 kHz
  2. Multiband Compression - Low / Mid / High bands compressed independently
  3. Limiting  - Hard limit at -1.0 dBTP
  4. Export    - 320 kbps MP3

Modes (--mode):
    full      - Load the entire file into RAM, then process. Lowest disk use,
                highest RAM use (8-12x the file size at peak). Default for files
                under 1 GB.
    streaming - Two-pass streaming: pass 1 processes the file in chunks and
                writes a float32 memmap to disk while measuring LUFS, pass 2
                applies the loudness gain and pipes PCM to ffmpeg for MP3
                encode. RAM stays bounded; temp disk roughly equals the input
                size. The processed audio is bit-exact vs. full mode.
    auto      - Pick streaming for inputs >= 1 GB, otherwise full. (Default.)

Requirements:
    pip install pedalboard scipy numpy pydub pyloudnorm
    ffmpeg must also be installed and on your PATH (for MP3 export)
    macOS:   brew install ffmpeg
    Windows: https://ffmpeg.org/download.html
"""

import argparse
import os
import subprocess
import sys
import tempfile
import numpy as np
from scipy.signal import butter, sosfiltfilt
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, PeakFilter, Compressor, Limiter
from pedalboard.io import AudioFile
from pydub import AudioSegment
import pyloudnorm as pyln

# Threshold (in bytes) at which --mode auto switches to streaming.
AUTO_STREAMING_THRESHOLD_BYTES = 1024 * 1024 * 1024  # 1 GB

# Streaming-mode chunk size in frames (samples per channel).
# 1,048,576 frames ~= 24 s at 44.1 kHz, ~16 MB per channel as float32.
STREAMING_CHUNK_FRAMES = 1 << 20


# ---------------------------------------------
# SETTINGS - tweak these to taste
# ---------------------------------------------

# EQ
HP_CUTOFF_HZ        = 30       # High-pass cutoff - removes inaudible sub rumble
NOTCH_FREQ_HZ       = 275      # Center of the mid cut (covers 200-350 Hz)
NOTCH_GAIN_DB       = -2.0     # How much to cut (negative = cut)
NOTCH_Q             = 0.8      # Width of the notch (lower = wider)
AIR_FREQ_HZ         = 11000    # Center of the high-end boost
AIR_GAIN_DB         = 2.0      # How much to boost the "air"
AIR_Q               = 0.6      # Width of the air boost

# Multiband Compression
# Low band (kick/bass)
LOW_CROSSOVER_HZ    = 200      # Low band upper boundary
LOW_THRESHOLD_DB    = -18.0
LOW_RATIO           = 2.0
LOW_ATTACK_MS       = 10.0
LOW_RELEASE_MS      = 100.0

# Mid band (mids/vocals)
MID_CROSSOVER_HZ    = 5000     # Mid band upper boundary
MID_THRESHOLD_DB    = -20.0
MID_RATIO           = 1.5
MID_ATTACK_MS       = 15.0
MID_RELEASE_MS      = 120.0

# High band (hi-hats/cymbals)
HIGH_THRESHOLD_DB   = -22.0
HIGH_RATIO          = 1.5
HIGH_ATTACK_MS      = 5.0
HIGH_RELEASE_MS     = 80.0

# Limiter
LIMIT_CEILING_DB    = -1.0     # True peak ceiling - prevents clipping
# Note: Pedalboard's Limiter only supports threshold and release (no hold parameter)
# 100ms release is the time it takes the limiter to let go after clamping down

# Loudness Normalization
TARGET_LUFS         = -14.0    # Target integrated loudness (-14 LUFS for SoundCloud/Mixcloud)

# ---------------------------------------------
# HELPER: Butterworth band-split filters
# ---------------------------------------------

def lowpass(audio, cutoff, sample_rate, order=4):
    sos = butter(order, cutoff, btype='low', fs=sample_rate, output='sos')
    return sosfiltfilt(sos, audio)

def bandpass(audio, low_cutoff, high_cutoff, sample_rate, order=4):
    sos = butter(order, [low_cutoff, high_cutoff], btype='band', fs=sample_rate, output='sos')
    return sosfiltfilt(sos, audio)

def highpass(audio, cutoff, sample_rate, order=4):
    sos = butter(order, cutoff, btype='high', fs=sample_rate, output='sos')
    return sosfiltfilt(sos, audio)

# ---------------------------------------------
# MAIN PROCESSING
# ---------------------------------------------

def peak_db(audio):
    """Return the peak level of the audio in dBFS."""
    peak = np.max(np.abs(audio))
    return 20 * np.log10(peak) if peak > 0 else -np.inf

def section(title):
    """Print a clearly visible section header."""
    print(f"\n{'-' * 50}")
    print(f"  {title}")
    print(f"{'-' * 50}")


def _build_streaming_chain(sample_rate: int):
    """Build the EQ + multiband + limiter chain as four persistent Pedalboard
    instances (eq, low, mid, high) plus a limiter. Each is reused across all
    chunks with reset=False so the IIR/compressor/limiter state is continuous --
    output is bit-exact vs. running the chain on the whole signal at once."""
    eq = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=HP_CUTOFF_HZ),
        PeakFilter(cutoff_frequency_hz=NOTCH_FREQ_HZ, gain_db=NOTCH_GAIN_DB, q=NOTCH_Q),
        PeakFilter(cutoff_frequency_hz=AIR_FREQ_HZ,   gain_db=AIR_GAIN_DB,   q=AIR_Q),
    ])
    # Crossovers: streaming mode uses minimum-phase IIR filters from pedalboard
    # rather than scipy's zero-phase sosfiltfilt (which needs the full signal).
    low_chain = Pedalboard([
        LowpassFilter(cutoff_frequency_hz=LOW_CROSSOVER_HZ),
        Compressor(threshold_db=LOW_THRESHOLD_DB, ratio=LOW_RATIO,
                   attack_ms=LOW_ATTACK_MS, release_ms=LOW_RELEASE_MS),
    ])
    mid_chain = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=LOW_CROSSOVER_HZ),
        LowpassFilter(cutoff_frequency_hz=MID_CROSSOVER_HZ),
        Compressor(threshold_db=MID_THRESHOLD_DB, ratio=MID_RATIO,
                   attack_ms=MID_ATTACK_MS, release_ms=MID_RELEASE_MS),
    ])
    high_chain = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=MID_CROSSOVER_HZ),
        Compressor(threshold_db=HIGH_THRESHOLD_DB, ratio=HIGH_RATIO,
                   attack_ms=HIGH_ATTACK_MS, release_ms=HIGH_RELEASE_MS),
    ])
    limiter = Pedalboard([Limiter(threshold_db=LIMIT_CEILING_DB, release_ms=100.0)])
    return eq, low_chain, mid_chain, high_chain, limiter


def process_mix_streaming(input_path: str, output_path: str):
    """Two-pass streaming pipeline. Bounds RAM by writing a float32 memmap of
    the pre-loudness audio to a temp file, measuring LUFS on the memmap, then
    re-reading it with the loudness gain applied while piping PCM to ffmpeg."""
    section("LOADING FILE (streaming mode -- chunked)")
    print(f"  Input file : {input_path}")
    print(f"  Output file: {output_path}")
    print(f"  Chunk size : {STREAMING_CHUNK_FRAMES:,} frames")

    with AudioFile(input_path) as f:
        sample_rate = f.samplerate
        num_channels = f.num_channels
        total_frames = f.frames
    duration_s = total_frames / sample_rate
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Channels   : {num_channels}")
    print(f"  Samples    : {total_frames:,}")
    print(f"  Duration   : {int(duration_s // 60)}m {int(duration_s % 60)}s")

    # -- Pass 1: process to memmap, accumulate stats --
    section("EQ + MULTIBAND + LIMITER (pass 1: chunked process to temp)")
    print(f"  EQ          : HP {HP_CUTOFF_HZ} Hz, notch {NOTCH_FREQ_HZ} Hz {NOTCH_GAIN_DB} dB, "
          f"air {AIR_FREQ_HZ} Hz +{AIR_GAIN_DB} dB")
    print(f"  Crossovers  : {LOW_CROSSOVER_HZ} / {MID_CROSSOVER_HZ} Hz "
          f"(streaming uses minimum-phase IIR -- bit-exact vs full mode for the "
          f"compressor+limiter chain)")
    print(f"  Limiter     : {LIMIT_CEILING_DB} dBTP, 100 ms release")

    # Temp memmap holds pre-loudness, post-limiter audio at full precision.
    tmp_dir = tempfile.gettempdir()
    fd, memmap_path = tempfile.mkstemp(prefix="process-mix-", suffix=".f32.tmp", dir=tmp_dir)
    os.close(fd)
    estimated_bytes = total_frames * num_channels * 4  # float32
    print(f"  Temp memmap : {memmap_path}")
    print(f"  Temp est.   : {estimated_bytes / (1024 ** 3):.2f} GB "
          f"(float32, {num_channels}ch, {total_frames:,} frames)")

    eq, low_chain, mid_chain, high_chain, limiter = _build_streaming_chain(sample_rate)

    # numpy memmap shape: (channels, frames). Created fresh; will be closed and
    # reopened read-only for pass 2 to avoid accidental writes.
    out_memmap = np.memmap(
        memmap_path, dtype=np.float32, mode="w+",
        shape=(num_channels, total_frames),
    )

    peak_input = 0.0
    peak_processed = 0.0
    frames_done = 0
    progress_every = max(1, total_frames // 20)  # ~5% steps
    next_progress = progress_every

    with AudioFile(input_path) as f:
        while frames_done < total_frames:
            want = min(STREAMING_CHUNK_FRAMES, total_frames - frames_done)
            chunk = f.read(want)  # shape (channels, frames), float32
            if chunk.shape[1] == 0:
                break

            chunk_peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
            if chunk_peak > peak_input:
                peak_input = chunk_peak

            # EQ
            y = eq.process(chunk, sample_rate, reset=False)
            # Multiband: each chain is band-split + compressor in series
            lo = low_chain.process(y,  sample_rate, reset=False)
            mi = mid_chain.process(y,  sample_rate, reset=False)
            hi = high_chain.process(y, sample_rate, reset=False)
            y = lo + mi + hi
            # Limiter (pre-loudness -- final limiter pass is on the loudness-
            # adjusted signal in pass 2)
            y = limiter.process(y, sample_rate, reset=False)

            out_peak = float(np.max(np.abs(y))) if y.size else 0.0
            if out_peak > peak_processed:
                peak_processed = out_peak

            n = y.shape[1]
            out_memmap[:, frames_done:frames_done + n] = y
            frames_done += n

            if frames_done >= next_progress:
                pct = 100.0 * frames_done / total_frames
                print(f"    pass 1: {pct:5.1f}%  ({frames_done:,} / {total_frames:,} frames)")
                next_progress += progress_every

    out_memmap.flush()
    print(f"  Input peak  : {20 * np.log10(peak_input) if peak_input > 0 else float('-inf'):.2f} dBFS")
    print(f"  Post-chain  : {20 * np.log10(peak_processed) if peak_processed > 0 else float('-inf'):.2f} dBFS")
    actual_bytes = os.path.getsize(memmap_path)
    print(f"  Temp size   : {actual_bytes / (1024 ** 3):.2f} GB on disk")

    # -- LUFS measurement on the memmap --
    section("LOUDNESS NORMALIZATION (measured on temp memmap)")
    # Reopen read-only and pass to pyloudnorm. The memmap is paged in by the OS
    # rather than fully resident -- RAM stays bounded.
    measure_view = np.memmap(
        memmap_path, dtype=np.float32, mode="r",
        shape=(num_channels, total_frames),
    )
    meter = pyln.Meter(sample_rate)
    # pyloudnorm wants (samples, channels). Transpose is a view, not a copy.
    current_lufs = meter.integrated_loudness(measure_view.T)
    loudness_delta = TARGET_LUFS - current_lufs
    gain_lin = 10.0 ** (loudness_delta / 20.0)
    print(f"  Target              : {TARGET_LUFS} LUFS")
    print(f"  Measured LUFS       : {current_lufs:.2f} LUFS")
    print(f"  Gain adjustment     : {loudness_delta:+.2f} dB  (linear x{gain_lin:.4f})")

    # Final limiter for pass 2 (fresh state -- different signal than pass-1 limiter)
    final_limiter = Pedalboard([Limiter(threshold_db=LIMIT_CEILING_DB, release_ms=100.0)])

    # -- Pass 2: gain + final limiter, pipe PCM to ffmpeg -> MP3 --
    section("EXPORT (pass 2: apply gain, pipe to ffmpeg)")
    print(f"  Format    : MP3")
    print(f"  Bitrate   : 320 kbps")
    print(f"  Output    : {output_path}")

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", str(num_channels),
        "-i", "pipe:0",
        "-c:a", "libmp3lame", "-b:a", "320k",
        output_path,
    ]
    print(f"  ffmpeg cmd: {' '.join(ffmpeg_cmd)}")

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    peak_final = 0.0
    clipped_after = 0
    frames_done = 0
    next_progress = progress_every

    try:
        for start in range(0, total_frames, STREAMING_CHUNK_FRAMES):
            end = min(start + STREAMING_CHUNK_FRAMES, total_frames)
            # Copy out of memmap so we can mutate freely
            chunk = np.array(measure_view[:, start:end], dtype=np.float32)
            chunk *= gain_lin
            chunk = final_limiter.process(chunk, sample_rate, reset=False)

            cp = float(np.max(np.abs(chunk))) if chunk.size else 0.0
            if cp > peak_final:
                peak_final = cp
            clipped_after += int(np.sum(np.abs(chunk) > 1.0))

            # ffmpeg expects interleaved float32. Pedalboard layout is
            # (channels, frames); transpose to (frames, channels) and
            # require contiguous before tobytes().
            interleaved = np.ascontiguousarray(chunk.T)
            proc.stdin.write(interleaved.tobytes())

            frames_done += chunk.shape[1]
            if frames_done >= next_progress:
                pct = 100.0 * frames_done / total_frames
                print(f"    pass 2: {pct:5.1f}%  ({frames_done:,} / {total_frames:,} frames)")
                next_progress += progress_every
    finally:
        if proc.stdin:
            proc.stdin.close()
        rc = proc.wait()
        del measure_view
        del out_memmap
        try:
            os.remove(memmap_path)
            print(f"  Removed temp: {memmap_path}")
        except OSError as e:
            print(f"  !!  Could not remove temp file {memmap_path}: {e}")
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with code {rc}")

    print(f"  Final peak  : {20 * np.log10(peak_final) if peak_final > 0 else float('-inf'):.2f} dBFS")
    if clipped_after > 0:
        print(f"  !!  {clipped_after:,} samples exceed 0 dBFS after final limiter -- "
              f"consider lowering LIMIT_CEILING_DB")
    else:
        print(f"      No clipping in final output")

    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size : {output_size_mb:.1f} MB")

    section("DONE")
    print(f"  Output saved to: {output_path}\n")


def process_mix(input_path: str, output_path: str, mode: str = "auto"):
    """Dispatch to the full or streaming pipeline based on `mode`.

    mode: "full", "streaming", or "auto" (auto picks streaming for inputs
    >= AUTO_STREAMING_THRESHOLD_BYTES).
    """
    section("MODE SELECTION")
    input_size = os.path.getsize(input_path)
    threshold_gb = AUTO_STREAMING_THRESHOLD_BYTES / (1024 ** 3)
    print(f"  Input size       : {input_size / (1024 ** 3):.2f} GB ({input_size:,} bytes)")
    print(f"  Requested mode   : {mode}")
    print(f"  Auto threshold   : {threshold_gb:.2f} GB")

    if mode == "auto":
        chosen = "streaming" if input_size >= AUTO_STREAMING_THRESHOLD_BYTES else "full"
        reason = (
            f"input >= {threshold_gb:.2f} GB" if chosen == "streaming"
            else f"input < {threshold_gb:.2f} GB"
        )
        print(f"  Resolved mode    : {chosen}  (reason: {reason})")
    else:
        chosen = mode
        print(f"  Resolved mode    : {chosen}")

    if chosen == "streaming":
        process_mix_streaming(input_path, output_path)
    elif chosen == "full":
        process_mix_full(input_path, output_path)
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected full, streaming, or auto)")


def process_mix_full(input_path: str, output_path: str):
    # -- Load -------------------------------------
    section("LOADING FILE (full mode -- entire file held in RAM)")
    print(f"  Input file : {input_path}")
    print(f"  Output file: {output_path}")

    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate

    duration_s  = audio.shape[1] / sample_rate
    duration_mm = int(duration_s // 60)
    duration_ss = int(duration_s % 60)
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Channels   : {audio.shape[0]}")
    print(f"  Samples    : {audio.shape[1]:,}")
    print(f"  Duration   : {duration_mm}m {duration_ss}s")
    print(f"  Peak level : {peak_db(audio):.2f} dBFS")

    section("EQ")
    print(f"  High-pass filter")
    print(f"    Cutoff     : {HP_CUTOFF_HZ} Hz  (removes sub rumble below this point)")
    print(f"  Mid notch (cut)")
    print(f"    Center freq: {NOTCH_FREQ_HZ} Hz")
    print(f"    Gain       : {NOTCH_GAIN_DB} dB")
    print(f"    Q          : {NOTCH_Q}  (width of the cut - lower = wider)")
    print(f"  Air boost")
    print(f"    Center freq: {AIR_FREQ_HZ} Hz")
    print(f"    Gain       : +{AIR_GAIN_DB} dB")
    print(f"    Q          : {AIR_Q}  (width of the boost - lower = wider)")

    peak_before = peak_db(audio)
    eq_chain = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=HP_CUTOFF_HZ),
        PeakFilter(cutoff_frequency_hz=NOTCH_FREQ_HZ, gain_db=NOTCH_GAIN_DB, q=NOTCH_Q),
        PeakFilter(cutoff_frequency_hz=AIR_FREQ_HZ,   gain_db=AIR_GAIN_DB,   q=AIR_Q),
    ])
    audio = eq_chain(audio, sample_rate)
    print(f"  Peak level : {peak_before:.2f} dBFS -> {peak_db(audio):.2f} dBFS")

    section("MULTIBAND COMPRESSION")
    print(f"  Band splits")
    print(f"    Low band   : 0 - {LOW_CROSSOVER_HZ} Hz")
    print(f"    Mid band   : {LOW_CROSSOVER_HZ} - {MID_CROSSOVER_HZ} Hz")
    print(f"    High band  : {MID_CROSSOVER_HZ}+ Hz")
    print(f"  Low band compression  (kick / bass)")
    print(f"    Threshold  : {LOW_THRESHOLD_DB} dB")
    print(f"    Ratio      : {LOW_RATIO}:1")
    print(f"    Attack     : {LOW_ATTACK_MS} ms")
    print(f"    Release    : {LOW_RELEASE_MS} ms")
    print(f"  Mid band compression  (mids / vocals)")
    print(f"    Threshold  : {MID_THRESHOLD_DB} dB")
    print(f"    Ratio      : {MID_RATIO}:1")
    print(f"    Attack     : {MID_ATTACK_MS} ms")
    print(f"    Release    : {MID_RELEASE_MS} ms")
    print(f"  High band compression (hi-hats / cymbals)")
    print(f"    Threshold  : {HIGH_THRESHOLD_DB} dB")
    print(f"    Ratio      : {HIGH_RATIO}:1")
    print(f"    Attack     : {HIGH_ATTACK_MS} ms")
    print(f"    Release    : {HIGH_RELEASE_MS} ms")

    peak_before = peak_db(audio)

    low_band  = lowpass( audio, LOW_CROSSOVER_HZ,                   sample_rate)
    mid_band  = bandpass(audio, LOW_CROSSOVER_HZ, MID_CROSSOVER_HZ, sample_rate)
    high_band = highpass(audio, MID_CROSSOVER_HZ,                   sample_rate)

    print(f"  Band peak levels before compression")
    print(f"    Low        : {peak_db(low_band):.2f} dBFS")
    print(f"    Mid        : {peak_db(mid_band):.2f} dBFS")
    print(f"    High       : {peak_db(high_band):.2f} dBFS")

    def compress_band(band, threshold, ratio, attack_ms, release_ms):
        comp = Pedalboard([Compressor(
            threshold_db=threshold,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )])
        return comp(band, sample_rate)

    low_band  = compress_band(low_band,  LOW_THRESHOLD_DB,  LOW_RATIO,  LOW_ATTACK_MS,  LOW_RELEASE_MS)
    mid_band  = compress_band(mid_band,  MID_THRESHOLD_DB,  MID_RATIO,  MID_ATTACK_MS,  MID_RELEASE_MS)
    high_band = compress_band(high_band, HIGH_THRESHOLD_DB, HIGH_RATIO, HIGH_ATTACK_MS, HIGH_RELEASE_MS)

    print(f"  Band peak levels after compression")
    print(f"    Low        : {peak_db(low_band):.2f} dBFS")
    print(f"    Mid        : {peak_db(mid_band):.2f} dBFS")
    print(f"    High       : {peak_db(high_band):.2f} dBFS")

    audio = low_band + mid_band + high_band
    print(f"  Combined peak level : {peak_before:.2f} dBFS -> {peak_db(audio):.2f} dBFS")

    section("LOUDNESS NORMALIZATION")
    print(f"  Target              : {TARGET_LUFS} LUFS")
    print(f"  Standard            : -14 LUFS (SoundCloud / Mixcloud)")
    print(f"                        -16 LUFS (Spotify)")

    # pyloudnorm expects shape (samples, channels), pedalboard uses (channels, samples)
    audio_transposed = audio.T
    meter = pyln.Meter(sample_rate)
    current_lufs = meter.integrated_loudness(audio_transposed)
    loudness_delta = TARGET_LUFS - current_lufs
    print(f"  Measured LUFS       : {current_lufs:.2f} LUFS")
    print(f"  Gain adjustment     : {loudness_delta:+.2f} dB")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress pyloudnorm's own clipping warning
        audio_transposed = pyln.normalize.loudness(audio_transposed, current_lufs, TARGET_LUFS)
    audio = audio_transposed.T

    # Verify LUFS and check for clipping introduced by normalization
    verified_lufs = pyln.Meter(sample_rate).integrated_loudness(audio.T)
    print(f"  Verified LUFS       : {verified_lufs:.2f} LUFS")
    print(f"  Peak level          : {peak_db(audio):.2f} dBFS")

    clipped = np.sum(np.abs(audio) > 1.0)
    if clipped > 0:
        clipped_db = peak_db(audio)
        print(f"  !!  Clipped samples  : {clipped:,} samples exceed 0 dBFS "
              f"(peak {clipped_db:.2f} dBFS) - limiter will catch these next")
    else:
        print(f"      No clipping detected after normalization")

    # Runs AFTER normalization to catch any peaks pushed over the ceiling.
    section("LIMITER")
    print(f"  Ceiling (threshold) : {LIMIT_CEILING_DB} dBTP")
    print(f"  Release             : 100.0 ms")
    print(f"  Note: Pedalboard's Limiter has no hold parameter (release only)")

    peak_before = peak_db(audio)
    limiter_chain = Pedalboard([Limiter(threshold_db=LIMIT_CEILING_DB, release_ms=100.0)])
    audio = limiter_chain(audio, sample_rate)
    print(f"  Peak level          : {peak_before:.2f} dBFS -> {peak_db(audio):.2f} dBFS")

    # Final clipping check after limiting
    clipped_after = np.sum(np.abs(audio) > 1.0)
    if clipped_after > 0:
        print(f"  !!  {clipped_after:,} samples still exceed 0 dBFS after limiting - "
              f"consider lowering LIMIT_CEILING_DB")
    else:
        print(f"      No clipping in final output")

    section("EXPORT")
    print(f"  Format    : MP3")
    print(f"  Bitrate   : 320 kbps")
    print(f"  Output    : {output_path}")

    temp_wav = output_path.replace(".mp3", "_temp.wav")
    with AudioFile(temp_wav, 'w', samplerate=sample_rate, num_channels=audio.shape[0]) as f:
        f.write(audio)

    sound = AudioSegment.from_wav(temp_wav)
    sound.export(output_path, format="mp3", bitrate="320k")
    os.remove(temp_wav)

    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size : {output_size_mb:.1f} MB")

    section("DONE")
    print(f"  Output saved to: {output_path}\n")


# ---------------------------------------------
# ENTRY POINT
# ---------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DJ mix post-processing: EQ, multiband compression, "
                    "limiting, loudness normalization, MP3 export.",
    )
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("output", help="Output MP3 file")
    parser.add_argument(
        "--mode", choices=("full", "streaming", "auto"), default="auto",
        help="full = load entire file into RAM (fastest, most memory). "
             "streaming = chunked two-pass with a memmap temp file (bounded "
             "RAM). auto = streaming when input >= 1 GB, else full. "
             "Default: auto.",
    )
    args = parser.parse_args()

    process_mix(args.input, args.output, mode=args.mode)
