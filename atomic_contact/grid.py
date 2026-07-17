# prepare database
import pickle

import numpy as np
from mar_fit import MARGrid, prepare_mar_database

grid = MARGrid(
    V_mV=np.linspace(-0.9, 0.9, 1801),
    tau=np.linspace(0, 1, 1001),
    T_K=0.0,
    Delta_meV=np.linspace(0.187, 0.191, 9),
    gamma_meV=1e-6,
    sigmaV_mV=np.linspace(0, 0.05, 101),
)
database = prepare_mar_database(grid)

with open("atomic_contact/grid.pkl", "wb") as f:
    pickle.dump(database, f)
