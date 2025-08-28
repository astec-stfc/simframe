import os
from SimulationFramework import Framework as fw
from SimulationFramework.Modules.optimisation.constraints import constraintsClass
from SimulationFramework.Modules import Beams as rbf
from SimulationFramework.Modules import Twiss as rtf
import shutil
import uuid
import yaml as yaml


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


class TemporaryDirectory(object):
    """Context manager for tempfile.mkdtemp() so it's usable with "with" statement."""

    def __init__(self, dir=os.getcwd(), *args, **kwargs):
        self.dir = dir
        self.args = args
        self.kwargs = kwargs

    def tempname(self):
        return "tmp" + str(uuid.uuid4())

    def __enter__(self):
        exists = True
        while exists:
            self.name = self.dir + "/" + self.tempname()
            if not os.path.exists(self.name):
                exists = False
                os.makedirs(self.name)
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            shutil.rmtree(self.name)
        except Exception:
            pass


class fitnessFunc(object):

    def __init__(
        self, *args, lattice="Lattices/clara400_v12_v3_elegant_jkj.def", **kwargs
    ):
        self.CLARA_dir = os.path.relpath(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )  # if CLARA_dir is None else CLARA_dir
        self.cons = constraintsClass()
        self.beam = rbf.beam()
        self.twiss = rtf.twiss()
        self.lattice_file = lattice
        self.base_files = None
        self.verbose = False
        self.start_lattice = None
        self.scaling = 6
        self.overwrite = True
        self.clean = False
        self.post_injector = True
        self.startcharge = None
        self.changes = None
        self.sample_interval = 1
        self.npart = 2 ** (3 * self.scaling)
        self.ncpu = self.scaling * 3
        self.elegant_ncpu = 1
        self.genesis_ncpu = 2
        self.doTracking = True
        self.change_to_elegant = True
        self.framework = fw.Framework(
            ".", *args, clean=False, overwrite=False, verbose=False, **kwargs
        )

    def set_CLARA_directory(self, clara_dir):
        self.CLARA_dir = clara_dir

    def set_start_file(self, file):
        self.start_lattice = file

    def load_best(self, datadict):
        if isinstance(datadict, str):
            with open(datadict, "r") as infile:
                data = dict(yaml.load(infile, Loader=yaml.UnsafeLoader))
        elif isinstance(datadict, dict):
            data = datadict
        else:
            raise ValueError(
                "load_best requires a filename or a dictionary of key/values"
            )
        best = []
        for n, p in [r[:2] for r in self.parameter_names]:
            if n in data:
                # print("Loading from data dict", n, p)
                best.append(data[n][p])
            elif n == "bunch_compressor" and p == "set_angle":
                # print("Loading bunch_compressor set_angle", n, p)
                best.append(data["CLA-VBC-MAG-DIP-01"]["angle"])
            elif n == "startcharge":
                # print("Loading start charge from data dict", n, p)
                best.append(self.startcharge)
            else:
                print("Loading from lattice file", n, p)
                if not hasattr(self, "framework"):
                    self.framework = fw.Framework(None)
                if len(self.framework.elementObjects.keys()) < 1:
                    self.framework.loadSettings(self.lattice_file)
                try:
                    print(n, p, self.framework[n][p])
                    best.append(self.framework[n][p])
                except Exception:
                    raise ValueError("load_best error", os.path.abspath(datadict), n, p)
            self.best = best
        return best

    def setup_lattice(self, inputargs, tempdir, *args, **kwargs):
        self.dirname = tempdir
        self.input_parameters = list(inputargs)
        self.framework.setSubDirectory(self.dirname)

        # print('self.ncpu = ', self.ncpu)
        if not os.name == "nt":
            self.framework.defineASTRACommand(ncpu=self.ncpu)
            self.framework.defineCSRTrackCommand(ncpu=self.ncpu)
        self.framework.defineElegantCommand(ncpu=self.elegant_ncpu)
        # if os.name == 'nt':
        #     self.framework.defineElegantCommand(['mpiexec','-np','10','pelegant'])
        self.framework.loadSettings(self.lattice_file)
        if hasattr(self, "change_to_elegant") and self.change_to_elegant:
            self.framework.change_Lattice_Code(
                "All", "elegant", exclude=["injector400"]
            )
        elif hasattr(self, "change_to_astra") and self.change_to_astra:
            self.framework.change_Lattice_Code("All", "ASTRA", exclude=["injector400"])
        elif hasattr(self, "change_to_gpt") and self.change_to_gpt:
            self.framework.change_Lattice_Code("All", "GPT", exclude=["injector400"])

        # ## Define starting lattice
        if self.start_lattice is None:
            self.start_lattice = self.framework[
                list(self.framework.latticeObjects)[0]
            ].objectname

        # ## Apply any pre-tracking changes to elements
        if self.changes is not None:
            if isinstance(self.changes, (tuple, list)):
                for c in self.changes:
                    self.framework.load_changes_file(c, verbose=False)
            else:
                self.framework.load_changes_file(self.changes)

        # ## Apply input arguments to element definitions
        """ Apply arguments: [[element, parameter, value], [...]] """
        for e, p, v in self.input_parameters:
            if e == "startcharge":
                print("Setting start charge to ", v)
                self.startcharge = v
            else:
                self.framework.modifyElement(e, p, v)

        # ## Save the changes to the run directory
        self.framework.save_changes_file(
            filename=self.framework.subdirectory + "/changes.yaml",
            elements=self.input_parameters,
        )

    def before_tracking(self):
        pass

    def track(self, endfile=None, **kwargs):
        # ## Have we defined where the base files are?
        if self.base_files is not None:
            self.framework[self.start_lattice].prefix = self.base_files
            if self.verbose:
                print("Using base_files = ", self.framework[self.start_lattice].prefix)
        # ## If not, use the defaults based on the location of the CLARA example directory
        elif self.CLARA_dir is not None:
            self.framework[self.start_lattice].prefix = self.CLARA_dir + "/basefiles_" + str(self.scaling) + "/"
            if self.verbose:
                print("Using CLARA_dir base_files = ", self.framework[self.start_lattice].prefix)

        # ## Are we are setting the charge?
        if self.startcharge is not None:
            if self.verbose:
                print("Starting Charge =", self.startcharge)
            self.framework[self.start_lattice].bunch_charge = 1e-12 * self.startcharge

        # ## Are we are sub-sampling the distribution?
        if self.sample_interval is not None:
            if self.verbose:
                print("Sampling at ", self.sample_interval)
            self.framework[self.start_lattice].sample_interval = (
                self.sample_interval
            )  # 2**(3*4)

        # ## TRACKING
        if self.doTracking:
            if self.verbose:
                print("Tracking from", self.start_lattice, "to", endfile)
            self.framework.track(
                startfile=self.start_lattice, endfile=endfile, **kwargs
            )


def optfunc(inputargs, dir=None, *args, **kwargs):
    global bestfit
    if dir is None:
        with TemporaryDirectory(dir=os.getcwd()) as tmpdir:
            fit = fitnessFunc(inputargs, tmpdir, *args, **kwargs)
            fitvalue = fit.calculateBeamParameters()
    else:
        fit = fitnessFunc(inputargs, dir, *args, **kwargs)
        fitvalue = fit.calculateBeamParameters()
    return fitvalue
