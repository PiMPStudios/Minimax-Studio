"""Model-license notices both the GUI and the worker may import.

Quote these; do not paraphrase. Verified against the upstream LICENSE files
2026-09-02 (MiniMaxAI/MiniMax-Music3, MiniMaxAI/MiniMax-H3).

Music 3 §3.1 requires “MiniMax-Music3” shown *prominently* in the UI of any
commercial product or service that *uses* the weights — shipping them is not
the trigger. The window header and the Music page both carry the name today.
§3.2 needs written authorization above US$20M aggregate yearly revenue; §4
binds anyone offering generation to third parties. Whether Studio is a
commercial product or service is a business/legal call — confirm with counsel,
do not settle it in a comment.

Upstream notices (§ tail of the Music 3 license): Qwen3-8B Apache-2.0,
Stable Audio MIT, DAC MIT.
"""

H3_TERRITORY = (
    "The MiniMax H3 Community License does not authorize using the open weights "
    "(or their outputs) in the US, EU, UK, or South Korea unless MiniMax grants "
    "a separate license. The MiniMax hosted API remains globally available."
)

MUSIC_CREDIT = (
    "The MiniMax-Music3 Community License requires UI credit on commercial "
    "products. There is no geographic carve-out. Over USD 20 million/year "
    "needs MiniMax authorization (api@minimax.io)."
)

MUSIC_UPSTREAMS = (
    "MiniMax-Music3 was fine-tuned from Qwen3-8B (Apache License 2.0). "
    "DiT-2B was modified from the Stable Audio code (MIT). The VAE was "
    "modified from the DAC code (MIT)."
)
