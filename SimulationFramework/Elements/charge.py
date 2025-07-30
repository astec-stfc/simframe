from SimulationFramework.Framework_objects import frameworkElement


class charge(frameworkElement):
    """
    Class defining a charge element.
    """

    total: float = None
    """Bunch charge [C]"""

    def __init__(self, *args, **kwargs):
        super(charge, self).__init__(*args, **kwargs)
