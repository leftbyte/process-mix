# process-mix

A DJ mix post-processing script. It applies a mastering chain to a recording and exports a 320 kbps MP3 ready for SoundCloud or Mixcloud.

The input can be:

- **A video**, such as an OBS recording (MP4, MOV, M4V, MKV). The audio track is extracted automatically; there is no need to pull it out with VLC first.
- **An audio file**, such as a WAV or MP3.

## Quick start

```
git clone https://github.com/berkley/process-mix
cd process-mix
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python process-mix.py ~/Music/Recordings/OBS/"2026-11-14 20-06-14.mp4"
```

This writes `2026-11-14 20-06-14.mp3` next to the video. `./run-process-mix.sh` is a shortcut that runs the script with the `.venv` Python and takes the same arguments:

```
./run-process-mix.sh ~/Music/Recordings/OBS/"2026-11-14 20-06-14.mp4"
```

 An hour-long mix takes about 2-3 minutes.

## Requirements

- Python 3 with the packages in `requirements.txt` (`pedalboard`, `scipy`, `numpy`, `pyloudnorm`). Installing them into a `.venv` as above keeps them separate from the system Python.
- `ffmpeg` and `ffprobe` on your PATH. They extract the audio from video, run the limiter and encode the MP3.
  - macOS: `brew install ffmpeg`
  - Windows: https://ffmpeg.org/download.html

## Usage

```
.venv/bin/python process-mix.py INPUT [OUTPUT] [--audio-track N] [--mode full|streaming|auto]
```

Examples:

```
.venv/bin/python process-mix.py recording.mp4                         # writes recording.mp3
.venv/bin/python process-mix.py recording.mp4 "Friday Night.mp3"      # choose the output name
.venv/bin/python process-mix.py recording.mp4 --audio-track 2         # use OBS audio track 2
.venv/bin/python process-mix.py mix.wav mix-mastered.mp3              # audio input
```

| Option | Description |
|--------|-------------|
| `OUTPUT` | Optional. Defaults to the input's name with `.mp3`, in the same folder. The script refuses to overwrite its own input. |
| `--audio-track N` | Video input only: which audio track to use, numbered from 1 as in OBS. Default 1. The log lists every track in the file and marks the one used. |
| `--mode` | `full` holds the whole mix in RAM (8-12x the input size at peak). `streaming` processes in chunks with a temp file on disk, keeping RAM low. `auto` (default) streams inputs of 1 GB or more. |

## Processing chain

1. **Extract** (video only): the chosen audio track is copied to a temporary 16-bit WAV with ffmpeg. It is deleted when the run finishes, even if it fails.
2. **EQ**: high-pass at 30 Hz, cut at 275 Hz, "air" boost at 11 kHz.
3. **Multiband compression**: low, mid and high bands compressed independently.
4. **Loudness normalization**: the whole mix is turned up or down to -14 LUFS.
5. **Limiter**: a lookahead peak limiter (ffmpeg `alimiter`, run at 4x the sample rate) catches any peaks pushed over the ceiling. It only turns peaks down; it adds no gain.
6. **Export**: 320 kbps MP3.
7. **Output check**: the finished MP3 is measured with ffmpeg's EBU R128 meter, and its loudness and true peak are printed.

## Reading the output check

The end of the log shows what was actually delivered:

```
  OUTPUT CHECK (measured on the MP3)
  Integrated loudness : -14.0 LUFS  (target -14.0)
  True peak           : -1.0 dBTP  (ceiling -1.0)
```

A `!!` line appears if the loudness is more than 1 dB off target or the true peak is over the ceiling. If the peak is over, raise `LIMIT_HEADROOM_DB`.

## Default settings

All settings are constants at the top of `process-mix.py` and can be adjusted to taste.

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

### Multiband compression

| Band | Range           | Threshold  | Ratio | Attack  | Release |
|------|-----------------|------------|-------|---------|---------|
| Low  | 0 - 200 Hz      | -18.0 dB   | 2:1   | 10 ms   | 100 ms  |
| Mid  | 200 - 5000 Hz   | -20.0 dB   | 1.5:1 | 15 ms   | 120 ms  |
| High | 5000+ Hz        | -22.0 dB   | 1.5:1 | 5 ms    | 80 ms   |

### Loudness and limiting

| Setting | Constant | Value | Description |
|---------|----------|-------|-------------|
| Target loudness | `TARGET_LUFS` | -14.0 LUFS | SoundCloud/Mixcloud standard (Spotify uses -16) |
| Ceiling | `LIMIT_CEILING_DB` | -1.0 dBTP | Highest true peak allowed in the final MP3 |
| Headroom | `LIMIT_HEADROOM_DB` | 1.0 dB | The limiter works at -2.0 dB because MP3 encoding adds about 0.6 dB of peak |
| Oversampling | `LIMIT_OVERSAMPLE` | 4x | Catches peaks that fall between samples |
| Attack | `LIMIT_ATTACK_MS` | 5 ms | Lookahead time |
| Release | `LIMIT_RELEASE_MS` | 100 ms | Time to let go after clamping down |

## Note on older output

Versions before November 2026 used pedalboard's `Limiter` as the final stage. That limiter also compresses everything above about -10 dB and adds about 4.75 dB of gain, so those MP3s came out at roughly -10.5 LUFS with true peaks above 0 dBTP (audible as clipping on some players). The script's log still reported -14 LUFS, because it measured before the limiter. Mixes processed now are quieter and less squashed than those earlier ones; this is the intended -14 LUFS result.
