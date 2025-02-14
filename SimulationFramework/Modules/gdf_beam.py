from munch import Munch
import easygdf


class gdf_beam(Munch):

    def __init__(self, filename) -> None:
        super().__init__(self)
        self.screens_touts = easygdf.load_screens_touts(filename)
        self.sort_screens()
        self._create_positions_dictionary()
        self.sort_touts()
        self._create_times_dictionary()

    @property
    def screens(self) -> dict:
        return self.screens_touts['screens']

    @property
    def touts(self) -> dict:
        return self.screens_touts['touts']

    def _create_positions_dictionary(self) -> None:
        self._positions = {s['position']: s for s in self.screens}

    @property
    def positions(self) -> dict:
        return self._positions

    def _create_times_dictionary(self) -> None:
        self._times = {s['time']: s for s in self.touts}

    @property
    def times(self) -> dict:
        return self._times

    def sort_screens(self, **kwargs) -> None:
        self.screens_touts['screens'] = sorted(self.screens_touts['screens'], key=lambda s: s['position'], **kwargs)

    def sort_touts(self, **kwargs) -> None:
        self.screens_touts['touts'] = sorted(self.screens_touts['touts'], key=lambda s: s['time'], **kwargs)

    def get_position(self, position: float) -> dict | None:
        if position in self._positions.keys():
            return Munch(self._positions[position])
        return None

    def get_time(self, time: float) -> dict | None:
        if time in self._times.keys():
            return Munch(self._times[time])
        return None
