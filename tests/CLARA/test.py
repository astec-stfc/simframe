from src.FEBE_Simple_NM import FEBE_Mode_1
from argparse import Namespace

if __name__ == "__main__":

    opt = FEBE_Mode_1(argparse=Namespace(sample=1, charge=250, subdir="1"), charge=250)
    opt.base_files = '../../Basefiles/Base_' + str(250) + 'pC/'  # This is where to look for the input files (in this case CLA-S02-APER-01.hdf5)
    opt.deleteFolders = False
    opt.sample_interval = 2**(3*1)
    opt.set_start_file('FEBE')
    opt.verbose = True
    opt.Example(dir='Setups/Setup_'+str(1)+'_'+str(250)+'pC/')
