import networkx as nx
import matplotlib.pyplot as plt

if __name__ == '__main__':
    G = nx.random_lobster(100,0.9,0.9)
    nx.draw(G)
    plt.show()