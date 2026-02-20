# Secure EOT/HOT implementation

This is an upgraded protocol for communication between end-of-train devices (EOTD) and head-of-train devices (HOTD) that supports message authentication using ECDH and HMAC-SHA256, as well as a legacy mode for backward compatibility.

To ensure minimal operator changes, the pairing process is designed to be as similar to the old process as possible. The old pairing procedure is described below:
1. Railman installs the EOTD to the end of the train and relays to the engineer at the head of the train the EOT ID.
2. The engineer inputs the 5-digit EOT ID into the HOTD.
3. The railman pushes the TEST button on the EOTD, which broadcasts an ARM message.
4. The engineer at the head of the train has 5 seconds to press the ARM button to confirm pairing.

Sources:
- <https://www.youtube.com/watch?v=UI4a9ygz_pI&t=316>
- <https://vimeo.com/groups/310557/videos/124589083>
