from SimulationFramework.Framework_objects import frameworkElement


class apcontour(frameworkElement):
    """
    Class defining a contour.
    """

    resolution: float = 0.001
    """z resolution of finding intersection"""

    filename: str = ""
    """Name of file containing contour data """

    xcolumn: str = ""
    """Name of column containing x data """

    ycolumn: str = ""
    """Name of column containing y data """

    def __init__(self, *args, **kwargs):
        super(apcontour, self).__init__(*args, **kwargs)
