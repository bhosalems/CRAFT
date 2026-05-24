import os
from typing import List, Dict, Any, Tuple




def _sanity_check_sources(
    sources: List[str],
):
    valid_source = [False] * len(sources)
    for i, source in enumerate(sources):
        valid_source[i] = os.path.exists(source)

    unvalidated_sources = [s for i, s in enumerate(sources) if not valid_source[i]]
    if len(unvalidated_sources) > 0:
        raise ValueError(f"The following sources do not exist: {unvalidated_sources}")