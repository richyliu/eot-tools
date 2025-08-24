# EOT/HOT/DPU parser

Data types:
- EOT/DPU packet: parsed EOT/DPU data into individual fields
- bitstring: string of 1's and 0's that represent AFSK demodulated data
- modulation signal: signal wave that modulates the carrier

There are functions that can convert between the different data types.

## Use with airspy

It is recommended to use some sort of buffering tool.

```sh
airspy_rx -f 455.5 -g 15 -r - 2>/dev/null | mbuffer -q -m 16M | ./main.py 455500000 6000000 i16c >> log.txt
```
