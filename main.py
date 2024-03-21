import graphviz as gr

from utils import display_graph


def main():
    g = gr.Digraph()
    
    g.edge("C", "A")
    g.edge("C", "B")

    g.edge("X", "Y")
    g.edge("X", "Z")
    g.node("X", "X", color="red")

    g.edge("statistics", "causal inference")
    g.edge("statistics", "machine learning")

    display_graph(g)


if __name__ == '__main__':
    main()
