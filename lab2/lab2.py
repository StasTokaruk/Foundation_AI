import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from lab1.lab1 import RoadGraph


class CarAgent:
    def __init__(self, graph: nx.Graph, start: int, finish: int):
        self.graph = graph
        self.start = start
        self.finish = finish
        self.current_node = start
        self.visited = set([start])
        self.history = [start]

        self.positions = nx.get_node_attributes(graph, 'pos')

    def perceive(self):
        return list(self.graph.neighbors(self.current_node))

    def manhattan_distance(self, node1, node2):
        x1, y1 = self.positions[node1]
        x2, y2 = self.positions[node2]
        return abs(x1 - x2) + abs(y1 - y2)

    def decide_way(self):
        neighbors = self.perceive()
        if not neighbors:
            return None

        unvisited = [n for n in neighbors if n not in self.visited]

        if unvisited:
            best_neighbor = min(unvisited, key=lambda n: self.manhattan_distance(n, self.finish))
        else:
            best_neighbor = min(neighbors, key=lambda n: self.manhattan_distance(n, self.finish))

        return best_neighbor

    def move(self, next_node):
        if next_node is not None:
            self.current_node = next_node
            self.visited.add(next_node)
            self.history.append(next_node)

    def get_colors(self):
        colors = []
        for node in self.graph.nodes():
            if node == self.start:
                colors.append("green")
            elif node == self.current_node:
                colors.append("blue")
            elif node == self.finish:
                colors.append("gold")
            elif node in self.visited:
                colors.append("lightgrey")
            else:
                colors.append("white")
        return colors


road = RoadGraph(size=25, remove_edges=10)
start, goal = 0, 24
agent = CarAgent(road.graph, start, goal)

pos = nx.get_node_attributes(agent.graph, 'pos')
fig, ax = plt.subplots(figsize=(6, 6))

nx.draw(agent.graph, pos, node_color=agent.get_colors(),with_labels=True, node_size=500, edgecolors="black", ax=ax)


def update(frame):
    ax.clear()

    if agent.current_node == agent.finish:
        ani.event_source.stop()

    if frame > 0 and agent.current_node != agent.finish:
        next_node = agent.decide_way()
        agent.move(next_node)

    nx.draw(agent.graph, pos, node_color=agent.get_colors(),with_labels=True, node_size=500, edgecolors="black", ax=ax)

    path_edges = list(zip(agent.history[:-1], agent.history[1:]))
    nx.draw_networkx_edges(agent.graph, pos, edgelist=path_edges, edge_color="red", width=3, ax=ax)

    ax.set_title(f"Крок {frame}: агент у вершині {agent.current_node}")


ani = FuncAnimation(fig, update, frames=range(0, 100), interval=1000, repeat=False)


plt.show()
