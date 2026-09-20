"""Model-license notices both the GUI and the worker may import.

Quote these; do not paraphrase. Verified against the upstream LICENSE files
2026-09-02 (MiniMaxAI/MiniMax-Music3, MiniMaxAI/MiniMax-H3).

Music 3 §3.1 requires “MiniMax-Music3” shown *prominently* in the UI of any
commercial product or service that *uses* the weights — shipping them is not
the trigger. The window header and the Music page both carry the name today.
§3.2 needs written authorization above US$20M aggregate yearly revenue; §4
binds anyone offering generation to third parties.

Studio is not a commercial product (owner's decision, 2026-09-20 — a hobby
repo, Apache-2.0, published for whoever wants it), so §3.1 is not triggered by
shipping it. Keep the credit anyway: it is one line, and a downstream user
might be building something commercial. §4 is that downstream user's problem
to read, which is why these strings ship verbatim and Help quotes them.

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
