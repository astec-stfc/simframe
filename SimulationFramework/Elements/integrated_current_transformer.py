from SimulationFramework.Elements.wall_current_monitor import wall_current_monitor


class integrated_current_transformer(wall_current_monitor):

    def __init__(self, name=None, type="integrated_current_transformer", **kwargs):
        super().__init__(name, type, **kwargs)
