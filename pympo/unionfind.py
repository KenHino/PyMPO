"""
Union-Find data structure for disjoint set operations.
"""


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.group_leaders = list(range(n))

    def _find(self, x: int) -> int:
        assert 0 <= x < len(self.parent)
        if self.parent[x] != x:
            self.parent[x] = self._find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        x_root = self._find(x)
        y_root = self._find(y)
        if x_root == y_root:
            return
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1

    def is_same(self, x: int, y: int) -> bool:
        return self._find(x) == self._find(y)

    def is_root(self, x: int) -> bool:
        return self._find(x) == x

    def __repr__(self) -> str:
        return f"UnionFind({self.parent})"

    def update_group_leaders(self):
        new_group_leaders = []
        for leader in self.group_leaders:
            if self.is_root(leader):
                new_group_leaders.append(leader)
        self.group_leaders = new_group_leaders
        return self.group_leaders

    def copy(self):
        uf = UnionFind(len(self.parent))
        uf.parent = self.parent.copy()
        uf.rank = self.rank.copy()
        uf.group_leaders = self.group_leaders.copy()
        return uf
