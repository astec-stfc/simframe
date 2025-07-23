from SimulationFramework.Elements.wall_current_monitor import wall_current_monitor


class integrated_current_transformer(wall_current_monitor):

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super(integrated_current_transformer, self).__init__(
            *args,
            **kwargs,
        )
