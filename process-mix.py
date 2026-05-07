#!/usr/bin/env python

"""
DJ Mix Post-Processing Script
==============================
Applies the following chain to a WAV file:
  1. EQ       - High-pass at 30 Hz, notch cut at 275 Hz, shelf boost at 11 kHz
  2. Multiband Compression - Low / Mid / High bands compressed independently
  3. Limiting  - Hard limit at -1.0 dBTP
  4. Export    - 320 kbps MP3

Requirements:
    pip install pedalboard scipy numpy pydub pyloudnorm
    ffmpeg must also be installed and on your PATH (for MP3 export)
    macOS:   brew install ffmpeg
    Windows: https://ffmpeg.org/download.html
"""

import sys
import numpy as np
from scipy.signal import butter, sosfiltfilt
from pedalboard import Pedalboard, HighpassFilter, PeakFilter, Compressor, Limiter
from pedalboard.io import AudioFile
from pydub import AudioSegment
import pyloudnorm as pyln


# ─────────────────────────────────────────────
# SETTINGS - tweak these to taste
# ─────────────────────────────────────────────

# EQ
HP_CUTOFF_HZ        = 30       # High-pass cutoff - removes inaudible sub rumble
NOTCH_FREQ_HZ       = 275      # Center of the mid cut (covers 200–350 Hz)
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

# ─────────────────────────────────────────────
# HELPER: Butterworth band-split filters
# ─────────────────────────────────────────────

def lowpass(audio, cutoff, sample_rate, order=4):
    sos = butter(order, cutoff, btype='low', fs=sample_rate, output='sos')
    return sosfiltfilt(sos, audio)

def bandpass(audio, low_cutoff, high_cutoff, sample_rate, order=4):
    sos = butter(order, [low_cutoff, high_cutoff], btype='band', fs=sample_rate, output='sos')
    return sosfiltfilt(sos, audio)

def highpass(audio, cutoff, sample_rate, order=4):
    sos = butter(order, cutoff, btype='high', fs=sample_rate, output='sos')
    return sosfiltfilt(sos, audio)

# ─────────────────────────────────────────────
# MAIN PROCESSING
# ─────────────────────────────────────────────

def peak_db(audio):
    """Return the peak level of the audio in dBFS."""
    peak = np.max(np.abs(audio))
    return 20 * np.log10(peak) if peak > 0 else -np.inf

def section(title):
    """Print a clearly visible section header."""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def process_mix(input_path: str, output_path: str):
    import os

    # ── Load ─────────────────────────────────────
    section("LOADING FILE")
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
    print(f"    Low band   : 0 – {LOW_CROSSOVER_HZ} Hz")
    print(f"    Mid band   : {LOW_CROSSOVER_HZ} – {MID_CROSSOVER_HZ} Hz")
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
        print(f"  ⚠️  Clipped samples  : {clipped:,} samples exceed 0 dBFS "
              f"(peak {clipped_db:.2f} dBFS) - limiter will catch these next")
    else:
        print(f"  ✓  No clipping detected after normalization")

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
        print(f"  ⚠️  {clipped_after:,} samples still exceed 0 dBFS after limiting - "
              f"consider lowering LIMIT_CEILING_DB")
    else:
        print(f"  ✓  No clipping in final output")

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


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process-mix.py input.wav output.mp3")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2]

    process_mix(input_file, output_file)
