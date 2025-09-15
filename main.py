from plotter import Plotter

if __name__ == "__main__":
    plotter = Plotter()
    plotter.contour([[1, 2],[1, 2]], [[1, 2],[1]],[[1, 2],[1, 2]])
    plotter.show()
