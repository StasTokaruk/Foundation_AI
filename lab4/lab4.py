import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from lab1_sl import RoadGraph
# --- ІМПОРТ ФУНКЦІОНАЛУ CNN ---
from speed_cnn import get_sign_image, recognize_sign, get_cnn_model, load_sign_images


# ---

class KnowledgeBase:
    def __init__(self):
        self.world = {}

    def tell(self, node, neighbors):
        if node not in self.world:
            self.world[node] = list(neighbors)

    def ask(self, node):
        return self.world.get(node, [])


class CarAgent:
    def __init__(self, graph: nx.Graph, start: int, finish: int):
        self.graph = graph
        self.start = start
        self.finish = finish
        self.current_node = start
        self.visited = {start}
        self.history = [start]
        self.stack = [start]
        self.kb = KnowledgeBase()
        self.positions = nx.get_node_attributes(graph, 'pos')

        self.speed_history = []

        get_cnn_model()
        load_sign_images()

        self.kb.tell(start, self.perceive())

    def perceive(self):
        return list(self.graph.neighbors(self.current_node))

    def manhattan_distance(self, node1, node2):
        x1, y1 = self.positions[node1]
        x2, y2 = self.positions[node2]
        return abs(x1 - x2) + abs(y1 - y2)

    def set_speed(self, from_node, to_node):
        edge_data = self.graph.get_edge_data(from_node, to_node)
        sign_digit = edge_data.get('speed_limit', self.graph.get_edge_data(to_node, from_node).get('speed_limit'))

        if sign_digit is None:
            return 0

        simulated_image = get_sign_image(sign_digit)

        recognized_speed = recognize_sign(simulated_image)

        return recognized_speed

    def decide_next(self):
        known_neighbors = self.kb.ask(self.current_node)
        if not known_neighbors:
            return None

        unvisited = [n for n in known_neighbors if n not in self.visited]

        if unvisited:
            return min(unvisited, key=lambda n: self.manhattan_distance(n, self.finish))
        else:
            return None

    def move(self):
        next_node = self.decide_next()

        if next_node is not None:
            max_speed = self.set_speed(self.current_node, next_node)
            self.speed_history.append((self.current_node, next_node, max_speed))

            self.current_node = next_node
            self.visited.add(next_node)
            self.history.append(next_node)
            self.stack.append(next_node)
            self.kb.tell(next_node, self.perceive())

        else:
            if len(self.stack) > 1:
                self.stack.pop()
                self.current_node = self.stack[-1]
                self.history.append(self.current_node)

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


def draw_graph(agent, ax):
    nx.draw(agent.graph, agent.positions, node_color=agent.get_colors(),
            with_labels=False, node_size=500, edgecolors="black",linewidths=2, ax=ax)

    edge_labels = {
        (a,b): f"{agent.graph.get_edge_data(a, b).get('speed_limit', agent.graph.get_edge_data(b, a).get('speed_limit')) * 10} км/год"
        for a, b in agent.graph.edges()
    }

    nx.draw_networkx_edge_labels(agent.graph, agent.positions, edge_labels=edge_labels, font_color='purple',
                                 font_size=8, ax=ax)

    path_edges = list(zip(agent.history[:-1], agent.history[1:]))
    nx.draw_networkx_edges(agent.graph, agent.positions, edgelist=path_edges, edge_color="red", width=1, ax=ax)


road = RoadGraph(size=25, remove_edges=10)
start, goal = 0, 24
agent = CarAgent(road.graph, start, goal)

fig, ax = plt.subplots(figsize=(6, 6))


def update(frame):
    ax.clear()

    if agent.current_node == agent.finish:
        ani.event_source.stop()

        print(f"Пройдений шлях: {agent.history}\n")
        print("Швидкість, встановлена агентом на кожному відрізку:")
        for a, b, speed in agent.speed_history:
            print(f"    Дорога ({a} --> {b}): {speed} км/год")

    if frame > 0 and agent.current_node != agent.finish:
        agent.move()

    draw_graph(agent, ax)


ani = FuncAnimation(fig, update, frames=range(0, 100), interval=1000, repeat=False)
plt.show()