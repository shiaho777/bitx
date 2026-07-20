"""KEF — Knowledge-Externalization Framework (research reference implementation).

Thesis (validated in RESULTS.md): don't compress a saturated model (information
theory forbids it); relocate high-entropy FACTS out of the weights into an
editable, retrievable external store, and keep only low-entropy reusable SKILLS
in the weights. Savings on accelerator bytes are a by-product; the real value is
knowledge maintainability + unified recall/derive.

Three decoupled modules (the real-data-corrected architecture):
  - RetrievalEncoder : a DEDICATED small encoder produces the lookup key.
                       (the generation model cannot self-retrieve — proven.)
  - FactStore        : add/edit/delete + semantic search + sublinear index.
  - ReasoningCore    : a frozen LM that thinks/derives when no fact is hit.

Positioning: research prototype, synthetic + small-real validation. NOT a
"lossless compression" framework.
"""

__version__ = "0.1.0"

from kef.config import Config, set_seed, count_params, fmt_bits, banner
from kef.model import KEFModel

__all__ = ["Config", "KEFModel", "set_seed", "count_params", "fmt_bits",
           "banner", "__version__"]
