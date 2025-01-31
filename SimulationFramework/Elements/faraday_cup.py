from SimulationFramework.Elements.wall_current_monitor import wall_current_monitor


class faraday_cup(wall_current_monitor):

    def __init__(self, name=None, type="faraday_cup", **kwargs):
        super().__init__(name, type, **kwargs)
