# process-mix

A DJ mix post-processing script that applies a mastering chain to a WAV file and exports it as a 320 kbps MP3.

## Processing Chain

1. **EQ** - High-pass filter at 30 Hz, notch cut at 275 Hz, air shelf boost at 11 kHz
2. **Multiband Compression** - Low, mid, and high bands compressed independently
3. **Loudness Normalization** - Normalizes to -14 LUFS (SoundCloud/Mixcloud standard)
4. **Limiter** - Hard limit at -1.0 dBTP to prevent clipping
5. **Export** - 320 kbps MP3

## Requirements

Install Python dependencies:

```
pip install pedalboard scipy numpy pydub pyloudnorm
```

`ffmpeg` must also be installed and on your PATH for MP3 export:

- macOS:   `brew install ffmpeg`
- Windows: https://ffmpeg.org/download.html

## Usage

```
python process-mix.py input.wav output.mp3
```

## Default Settings

### EQ

| Parameter       | Value    | Description                        |
|-----------------|----------|------------------------------------|
| High-pass       | 30 Hz    | Removes inaudible sub rumble       |
| Notch center    | 275 Hz   | Mid cut to clean up muddy mids     |
| Notch gain      | -2.0 dB  | Depth of the cut                   |
| Notch Q         | 0.8      | Width of the cut (lower = wider)   |
| Air boost       | 11000 Hz | High-end brightness boost          |
| Air gain        | +2.0 dB  | Depth of the boost                 |
| Air Q           | 0.6      | Width of the boost (lower = wider) |

### Multiband Compression

| Band | Range           | Threshold  | Ratio | Attack  | Release |
|------|-----------------|------------|-------|---------|---------|
| Low  | 0 - 200 Hz      | -18.0 dB   | 2:1   | 10 ms   | 100 ms  |
| Mid  | 200 - 5000 Hz   | -20.0 dB   | 1.5:1 | 15 ms   | 120 ms  |
| High | 5000+ Hz        | -22.0 dB   | 1.5:1 | 5 ms    | 80 ms   |

### Loudness and Limiting

| Parameter       | Value      | Description                               |
|-----------------|------------|-------------------------------------------|
| Target loudness | -14.0 LUFS | SoundCloud/Mixcloud standard (-16 Spotify)|
| Limiter ceiling | -1.0 dBTP  | True peak ceiling to prevent clipping     |
| Limiter release | 100 ms     |                                           |

All settings are defined as constants at the top of `process-mix.py` and can be adjusted to taste.
