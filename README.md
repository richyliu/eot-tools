# EOT tools

## Setup

To set up, I recommend using python venv:
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You will also need to install [hackrf tools](https://hackrf.readthedocs.io/en/latest/installing_hackrf_software.html). It is available as `hackrf` on most package managers.

## Usage

Run `./main.py -h` to see the available options. Running `./main.py` without any arguments will output a sample EOT packet to stdout as a bitstring.

Run `./main.py -a <audio_file>` to generate an audio WAV file that can be sent over FM.

Use `./sample_decoder/decode.py <audio_file>` to decode the audio files generated.

## Best progress so far

This generates an EOT packet which is reliably detected by SoftEOT, but still is decoded as invalid:
```sh
./main.py -u 32312 -p 73 --no-motion --no-marker-light --no-marker-battery-weak --turbine -b 3 -c 0 -m normal --no-confirm --valve-circuit -a out.wav
```
