"""Makes the lab pipeline modules (soc_common, loader, canonical_tactics, sessionize_and_mine) importable
from the prototype folder without copying them."""
import os, sys
_lab = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lab")
if _lab not in sys.path:
    sys.path.insert(0, _lab)
