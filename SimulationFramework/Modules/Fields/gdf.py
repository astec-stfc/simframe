import easygdf
from warnings import warn

def write_gdf_beam_file(self):
    gdf_file = self.filename.replace(".hdf5", ".gdf")
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
    else:
        warn(f"Field type {self.field_type} not supported for GPT")
    if data is not None:
        easygdf.save(gdf_file, blocks)