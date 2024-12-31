from __future__ import annotations

def get_min_vertex_cover(
    U: set[str], E: set[tuple[str, str]], max_matching: dict[str, str]
) -> list[str]: ...
def get_maximal_matching(
    U: set[str], E: set[tuple[str, str]]
) -> dict[str, str]: ...
