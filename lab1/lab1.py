import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import random

matplotlib.use("TkAgg")


class RoadGraph:
    def __init__(self, size: int = 25, remove_edges: int = 5):
        self.size = size
        self.grid_size = int(size ** 0.5)
        if self.grid_size ** 2 != size:
            raise ValueError("Кількість вершин повинна утворювати квадратну сітку")
        self.remove_edges = remove_edges
        self.graph = nx.Graph()

        self._generate_graph()
        self._remove_edges()

    def _generate_graph(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                node = i * self.grid_size + j
                self.graph.add_node(node, pos=(j, -i))

                if j < self.grid_size - 1:
                    self.graph.add_edge(node, node + 1)
                if i < self.grid_size - 1:
                    self.graph.add_edge(node, node + self.grid_size)

    def _remove_edges(self):
        removed = 0
        while removed < self.remove_edges:
            bridges = set(nx.bridges(self.graph))
            candidates = [i for i in self.graph.edges() if i not in bridges]
            if not candidates:
                break
            edg = random.choice(candidates)
            self.graph.remove_edge(*edg)
            removed += 1

    def draw(self):
        pos = nx.get_node_attributes(self.graph, 'pos')
        plt.figure(figsize=(6, 6))
        nx.draw(self.graph, pos, with_labels=False, node_size=500, node_color="white", edgecolors="black", linewidths=2)
        plt.show()


# road = RoadGraph(size=25, remove_edges=10)
# road.draw()
