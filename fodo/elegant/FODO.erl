SDDS1
&description text="Error log--input: FODO.ele  lattice: FODO.lte", contents="error log, elegant output" &end
&associate filename="FODO.ele", path="/home/xkc85723/Documents/simframe", contents="elegant input, parent" &end
&associate filename="FODO.lte", path="/home/xkc85723/Documents/simframe", contents="elegant lattice, parent" &end
&parameter name=Step, type=long, description="simulation step" &end
&parameter name=When, type=string, description="phase of simulation when errors were asserted" &end
&column name=ParameterValue, type=double, description="Perturbed value" &end
&column name=ParameterError, type=double, description="Perturbation value" &end
&column name=ElementParameter, type=string, description="Parameter name" &end
&column name=ElementName, type=string, description="Element name" &end
&column name=ElementOccurence, type=long, description="Element occurence" &end
&column name=ElementType, type=string, description="Element type" &end
&data mode=ascii, lines_per_row=1, no_row_counts=1 &end
0              ! simulation step
pre-correction
-2.000000000000000e+01 0.000000000000000e+00         K1      QUAD1 1      KQUAD
4.000000000000000e+01 0.000000000000000e+00         K1      QUAD2 1      KQUAD
-1.000000000000000e+01 0.000000000000000e+00         K1      QUAD3 1      KQUAD

