import numpy as np
import networkx as nx
from pgmpy.base import DAG
from pgmpy.readwrite import XMLBIFWriter
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork

if __name__ == "__main__":
    i = 0
    while i <= 99:  # change to indicate amount of wanted BNs
        i += 1
        n_nodes = 12  # amount of nodes
        edge_chance = 0.5  # probability
        n_states = 2  # binary
        n_states_dict = {str(i): n_states for i in range(n_nodes)}

        adj_mat = np.random.choice(
            [0, 1], size=(n_nodes, n_nodes), p=[1 - edge_chance, edge_chance]
        )

        nodes = [str(n) for n in range(n_nodes)]
        mat = np.triu(adj_mat, k=1)

        graph = nx.convert_matrix.from_numpy_matrix(mat, create_using=nx.DiGraph)
        graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        edges = graph.edges()

        dag = DAG(edges)
        dag.add_nodes_from(nodes)

        bn_model = BayesianNetwork(dag.edges())
        bn_model.add_nodes_from(dag.nodes())

        cpds = []
        for node in bn_model.nodes():
            predecessors = list(bn_model.predecessors(node))
            cpds.append(
                TabularCPD.get_random(
                    variable=node, evidence=predecessors, cardinality=n_states_dict
                )
            )

        bn_model.add_cpds(*cpds)

        filename = "random" + str(n_nodes) + "-" + str(i) + ".BIFXML"
        XMLBIFWriter(bn_model).write_xmlbif(filename)
        print(f"wrote '{filename}'")

