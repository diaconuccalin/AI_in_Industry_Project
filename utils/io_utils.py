import io

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def display_graph(the_graph):
    the_graph = the_graph.pipe(format="png")

    sio = io.BytesIO()
    sio.write(the_graph)
    sio.seek(0)
    img = mpimg.imread(sio)

    plt.imshow(img, aspect='equal')
    plt.show()
