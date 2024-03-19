# Dihedrals

This repository contains rust and python code to predict whether two dihedral configurations are connected by a single transition state (TS), and if they are, what that TS is.

## Architecture

The input of the model to predict connections is a tensor of dimensions $[n_{\text{inputs}}, 1, 2, 178]$, encoding both the 178 dihedrals of the starting state and the 178 dihedrals of the final state. This is put through a convolutional layer with kernel size $3\times 3$ , input channels 1, output channels 1 and [half padding](https://github.com/vdumoulin/conv_arithmetic/tree/master) to produce another tensor of dimensions $[n_{\text{inputs}}, 1, 2, 178]$ (the convolutional layer helps directly encode local effects, specifically there is a relation between a dihedral in the start and the same dihedral in the end state).

The convolutional layer is then flattened to a tensor of dimensions $[n_{\text{inputs}}, 356]$, which is then put through a fully connected linear layer with 356 inputs and 356 outputs and a $\tanh$ activation function, to produce a tensor of dimensions $[n_{\text{inputs}}, 356]$. This output is then put through a linear layer with 356 inputs and 1 output, and a sigmoid activation function to produce a probability of connection for each of the inputs.

### Training

For training the loss function used is binary cross entropy loss between the predicted and actual connections.

In training [dropout layers](https://arxiv.org/abs/1207.0580) are used to preventing overfitting and co-adpatation: a dropout layer with dropout rate 0.2 is used on the training inputs and a layer with dropout rate 0.6 is used between the two linear layers.

As this is stochastic, the optimiser used is a stochastic optimiser, specifically [NAdam](https://docs.rs/candle-optimisers/latest/candle_optimisers/nadam/index.html): an incorporation of Nesterov momentum into Adam.

## Usage

### Setup

#### Rust Setup

Install rust compiler:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### Python Setup

This presumes that the python environment is python 3: otherwise all python commands should be replaced with `python3`.

* `git clone` this repository and `cd` into it.

* Create a virtual environment

```sh
python -m venv .venv
```

* Activate the virtual environment

```bash
source .venv/bin/activate
```

* install pytorch, following instructions at <https://pytorch.org/> using se lecting the appropriate options for your system (namely cuda version / CPU / ROCm: only tested on CUDA 12.1 however, which requires an NVIDIA GPU)

* install [safetensors](https://huggingface.co/docs/safetensors/index), for (de)serialization of the data and weights

#### PATHSAMPLE

Have a recently compiled version of [PATHSAMPLE](https://www-wales.ch.cam.ac.uk/PATHSAMPLE/) so that the dihedral data can be generated.

### Running

* create the dihedral data: from the root of the repository:

```sh
cd PATHSAMPLE
path/to/PATHSAMPLE # run PATHSAMPLE executable
cd ..
```

* build all rust code

```sh
cargo build --release
```

* run the rust code to generate the input tensors for training (this will not be the exact same as my results as I've changed the PRNG from when I orginally ran the code: from now on however this should be deterministic)

```sh
cargo r -r --bin setup-connections
```

* train the network to predict connections (this will take a long time if not accelerated and is not deterministic)

```sh
python predict_connected.py
```

* run the rust code to evaluate the model across all predictions

```sh
cargo r -r --bin eval-cx > connectpairs
```

 (this is also non deterministic due to parallelism affecting the order in which minima are put in connectpairs. The degree of parallelism can be controlled via the `RAYON_NUM_THREADS` environmental variable: by default it uses all available threads (it also does this if the variable is set to 0)): on a i7-12700K this takes about 17 s to evalaute all 9110046 unique pairs of minima.

## Future Work

A possible direction to consider for future improvements, may be rearchitecturing the network to use a transformer, in order to get the correct permutational invariance between the start and end states, better non locality for transferring the model to larger systems and to allow for the use of attention mechanisms to better encode the relations between dihedrals (see <https://huggingface.co/blog/intro-graphml> and Microsoft's Graphormers): there may also be some advantage in splitting the minima by amino acid type and having 20 channels in the input corresponding to the 20 amino acids, as this would allow the model to better encode the differences between amino acids (although this may need an ML architecture that can deal with sparsity better due to how sparse said encoding would be).
