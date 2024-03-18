# Dihedrals

This contains rust and python code to predict whether two dihedral configurations are connected by a single TS, and if they are, what that TS is.

Python notebooks (will change to .py) require safetensors and pytorch

Rust code requires rust toolchain to compile.

`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

There are 5 rust executables:

* `dihedrals2csv` reformats the dihedral data into a csv file
* `setup-connections` sets up the training data to predict connections (requires around 50 GB of RAM)
* `setup-train` sets up the training data to predict the TS
* `test-connected` takes the trained weights from pytorch and uses it to see which minima may be connected
* `train` is training code for the model in rust (pytorch may be quicker)
