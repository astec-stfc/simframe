import easygdf
from warnings import warn


def write_gdf_field_file(self):
    gdf_file = self._output_filename(extension=".gdf")
    data = None
    blocks = []
    if self.field_type == "LongitudinalWake":
        zdata = self.z.value.val
        wzdata = self.Wz.value.val
        blocks = [
            {"name": "z", "value": zdata},
            {"name": "Wz", "value": wzdata},
        ]
    elif self.field_type == "TransverseWake":
        zdata = self.z.value.val
        wxdata = self.Wx.value.val
        wydata = self.Wy.value.val
        blocks = [
            {"name": "z", "value": zdata},
            {"name": "Wx", "value": wxdata},
            {"name": "Wy", "value": wydata},
        ]
    elif self.field_type == "3DWake":
        zdata = self.z.value.val
        wxdata = self.Wx.value.val
        wydata = self.Wy.value.val
        wzdata = self.Wz.value.val
        blocks = [
            {"name": "z", "value": zdata},
            {"name": "Wx", "value": wxdata},
            {"name": "Wy", "value": wydata},
            {"name": "Wz", "value": wzdata},
        ]
    elif self.field_type == "1DMagnetoStatic":
        zdata = self.z.value.val
        bzdata = self.Bz.value.val
        blocks = [
            {"name": "z", "value": zdata},
            {"name": "Bz", "value": bzdata},
        ]
    elif self.field_type == "1DElectroDynamic":
        zdata = self.z.value.val
        ezdata = self.Ez.value.val
        blocks = [
            {"name": "z", "value": zdata},
            {"name": "Ez", "value": ezdata},
        ]
    else:
        warn(f"Field type {self.field_type} not supported for GPT")
    if data is not None:
        easygdf.save(gdf_file, blocks)
    return gdf_file
