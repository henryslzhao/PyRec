import  matplotlib.pyplot as plt
import networkx as nx

G = nx.random_lobster(100,0.6,0.9)
nx.draw(G)
plt.show()