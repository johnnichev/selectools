#!/usr/bin/env python3
"""Report public symbols whose stability marker disagrees with their module.

Walks ``selectools.__all__`` plus every public submodule's ``__all__`` and, for
each symbol, compares its own marker (``@stable``/``@beta``/``@deprecated`` via
``selectools.stability.get_stability``) against its module's ``__stability__``
promise. Two mismatch classes matter for the v1.0 freeze:

- **promotion candidate**: a ``beta`` symbol in a ``stable``-promised module —
  a mature symbol that hasn't been frozen yet.
- **lagging module**: a ``stable`` symbol in a ``beta``-promised module — the
  symbol was frozen but the module promise never caught up.

Usage::

    python scripts/stability_audit.py            # full mismatch report
    python scripts/stability_audit.py --summary   # counts only
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
from collections import defaultdict
from typing import Dict, List, Tuple

import selectools
from selectools.stability import get_stability


def _public_modules() -> Dict[str, object]:
    mods: Dict[str, object] = {"selectools": selectools}
    for info in pkgutil.walk_packages(selectools.__path__, "selectools."):
        short = info.name[len("selectools.") :]
        if any(part.startswith("_") for part in short.split(".")):
            continue
        try:
            module = importlib.import_module(info.name)
        except Exception:  # noqa: BLE001 - optional-dep modules may not import
            continue
        if hasattr(module, "__all__"):
            mods[info.name] = module
    return mods


def audit() -> List[Tuple[str, str, str, str]]:
    """Return (module, symbol, symbol_marker, module_promise) for every symbol."""
    rows: List[Tuple[str, str, str, str]] = []
    for modname, module in _public_modules().items():
        short = modname[len("selectools.") :] if modname != "selectools" else "(top)"
        promise = getattr(module, "__stability__", "?")
        for sym in getattr(module, "__all__", []):
            obj = getattr(module, sym, None)
            if obj is None:
                continue
            marker = get_stability(obj, f"{modname}.{sym}") or get_stability(obj, sym) or "unmarked"
            rows.append((short, sym, marker, promise))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true", help="print counts only")
    args = parser.parse_args()

    rows = audit()
    counts = defaultdict(int)
    for _, _, marker, _ in rows:
        counts[marker] += 1
    print(
        f"public symbol entries: {len(rows)}  |  "
        + "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )

    promote = [r for r in rows if r[2] == "beta" and r[3] == "stable"]
    lagging = [r for r in rows if r[2] == "stable" and r[3] == "beta"]
    print(f"promotion candidates (beta in stable module): {len(promote)}")
    print(f"lagging modules (stable symbol in beta module): {len(lagging)}")

    if args.summary:
        return 0

    for title, group in (("PROMOTION CANDIDATES", promote), ("LAGGING MODULES", lagging)):
        bymod: Dict[str, List[str]] = defaultdict(list)
        for short, sym, _, _ in group:
            bymod[short].append(sym)
        print(f"\n=== {title} ===")
        for short in sorted(bymod):
            print(f"  {short}: {', '.join(sorted(bymod[short]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
