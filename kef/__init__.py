"""KEF — Knowledge-Externalization Framework.

Relocate high-entropy facts into an editable external store; keep reusable
reasoning in weights. Not a lossless compression framework.

Modules:
  - RetrievalEncoder: dedicated small encoder for lookup keys
  - FactStore: add/edit/delete + gated semantic search
  - ReasoningCore: frozen LM for miss / derive path
"""

__version__ = "0.1.0"

from kef.config import Config, set_seed, count_params, fmt_bits, banner
from kef.model import KEFModel

__all__ = ["Config", "KEFModel", "set_seed", "count_params", "fmt_bits",
           "banner", "__version__"]
