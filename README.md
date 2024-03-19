# Dihedrals

This contains rust and python code to predict whether two dihedral configurations are connected by a single TS, and if they are, what that TS is.

Python notebooks (will change to .py) require safetensors and pytorch

Rust code requires rust toolchain to compile.

There are 5 rust executables:

* `dihedrals2csv` reformats the dihedral data into a csv file
* `setup-connections` sets up the training data to predict connections (requires around 50 GB of RAM)
* `setup-train` sets up the training data to predict the TS
* `test-connected` takes the trained weights from pytorch and uses it to see which minima may be connected
* `train` is training code for the model in rust (pytorch may be quicker)
* `eval-cx` evaluates the trained connection model on the train / test data that was provided to it

and two python files

* `predict_connected.py` trains the model to predict direct connections in pytorch
* `predict_ts.py` trains the model to predict transition states in pytorch

## Architecture

The input of the model to predict connections is a tensor of deminsions $[n_{\text{inputs}}, 1, 2, 178]$, encoding both the 178 dihedrals of the starting state and the 178 dihedrals of the final state. This is put through a convolutional layer with kernel size $3\times 3$ , input channels 1, output channels 1 anf [half padding](https://github.com/vdumoulin/conv_arithmetic/tree/master) to produce another tensor of dimensions $[n_{\text{inputs}}, 1, 2, 178]$ (the convolutional layer helps directly encode local effects, importantly that there is a relation between a dihedral in the start and the same dihedral in the end state).

The convolutional layer is then flattened to a tensor of dimensions $[n_{\text{inputs}}, 356]$, which is then put through a fully connected linear layer with 356 inputs and 356 outputs and a $\tanh$ activation function, to produce a tensor of dimensions $[n_{\text{inputs}}, 356]$,. This output is then put through a linear layer with 356 inputs and 1 output, and a sigmoid activation function to get a probability of connection for each of the inputs.

### Training

For training the loss function used is binary cross entropy loss between the predicted and actual connections.

In training [droupout layers](https://arxiv.org/abs/1207.0580) are used to preventing overfitting and co-adpatation: a dropout layer with dropout rate 0.2 is used on the training inputs and a layer with dropout rate 0.6 is used between the two linear layers.

As this is obviously stochastic, the optimiser used is also a stochastic optimiser, specifically [NAdam](https://docs.rs/candle-optimisers/latest/candle_optimisers/nadam/index.html), and incorporation of Nesterov momentum into Adam.

## Usage

### Setup

#### Rust Setup

Install rust compiler:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Build all rust code

```sh
cargo build --release
```

#### Python Setup

* Create a virtual environment

```sh
python -m venv .venv
```

* Activate the virtual environment

```bash
source .venv/bin/activate
```

* install pytorch, following instructions at <https://pytorch.org/> using se lecting the appropriate options for your system (namely cuda version / cpu / ROCm: only tested on CUDA 12.1 however, which requires an NVIDIA GPU)

* install [safetensors](https://huggingface.co/docs/safetensors/index), for (de)serialization of the data and weights
