from __future__ import annotations

import logging

from a_share_mtf_accumulation_core import run_profile


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(run_profile("aggressive"))
