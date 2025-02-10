import numpy as np
from .sdds import write_SDDS_field_file
from warnings import warn
from collections import Counter


def write_opal_field_file(
    self,
    frequency: float = None,
    radius: float = None,
    fourier: int = 100,
    orientation: str = None,
):
    length = self.length
    opal_file = self.filename.replace(".hdf5", ".opal")
    data = None
    header = None
    fourier_ratio = fourier / length
    if self.field_type is not "2DElectroDynamic":
        if fourier_ratio > 1.0:
            warn("Too many fourier components provided; defaulting to 1/100")
            fourier = int(length / 100)
        elif fourier_ratio < 0.01:
            warn("Not enough fourier components provided; defaulting to 1/100")
            fourier = int(length / 100)
    if self.field_type == "1DMagnetoStatic":
        zmin = min(self.z.value.val) * 1e-2
        zmax = max(self.z.value.val) * 1e-2
        data = self.Bz.value.val
        head = ["1DMagnetoStatic", str(fourier)]
        if radius is None:
            warn("Magnet radius not provided; defaulting to 10cm")
            radius = 0.1
        rvals = [str(0), str(radius * 100), str(length), fourier]
        zvals = [str(zmin), str(zmax), str(length)]
        header = [head, rvals, zvals]
    elif self.field_type == "1DElectroDynamic":
        if frequency is None:
            warn("RF Frequency not provided to field class")
            return
        head = ["ASTRADynamic", str(fourier)]
        freq = [str(frequency * 1e-6)]
        zdata = self.z.value.val
        ezdata = self.Ez.value.val
        data = np.transpose([zdata, ezdata])
        header = [head, freq]
    elif self.field_type == "2DElectroDynamic":
        if frequency is None:
            warn("RF Frequency not provided to field class")
            return
        if radius is None:
            warn("Radius not provided to field class")
            return
        if orientation is None:
            warn("Orientation not provided to field class")
            return
        orient = self.orientation
        ezvals = self.Ez.value.val
        ervals = self.Er.value.val
        eabsvals = self.Ex.value.val
        brvals = self.Br.value.val
        shape = len(ezvals)
        count = Counter(ezvals)
        repeating_floats = {key: value for key, value in count.items() if value > 1}
        zlen = int(len(repeating_floats) - 1)
        rlen = int((shape / len(repeating_floats)) - 1)
        head = ["2DDynamic", orient]
        leng = ["0.0", str(self.length * 10), str(zlen)]
        freq = [str(frequency * 1e-6)]
        rad = ["0.0", str(self.radius * 10), str(rlen)]
        data = np.transpose([ezvals, ervals, eabsvals, brvals])
        header = [head, leng, freq, rad]
    elif "wake" in self.field_type:
        write_SDDS_field_file(self)
        warn(f"Field type {self.field_type} defaulting to SDDS type; use with caution")
    else:
        warn(f"Field type {self.field_type} not supported for OPAL")
    if data is not None:
        with open(f"{opal_file}", "w") as f:
            for h in header:
                f.write(" ".join([str(x) for x in h]) + "\n")
            for d in data:
                f.write(" ".join([str(x) for x in d]) + "\n")
