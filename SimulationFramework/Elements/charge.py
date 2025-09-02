from SimulationFramework.Framework_objects import frameworkElement


class charge(frameworkElement):
    """
    Class defining a charge element.
    """

    total: float | None = None
    """Bunch charge [C]"""
