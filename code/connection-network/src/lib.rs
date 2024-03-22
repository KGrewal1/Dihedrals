use candle_core::{Device, Result, Tensor};
use candle_nn::{ops::sigmoid, Linear, Module, VarBuilder};
use log::info;

const NDIHEDRALS: usize = 178;
const NINPUTS: usize = 2 * NDIHEDRALS;
const NHIDDEN: usize = 356;

#[derive(Debug)]
pub struct CheckCxModel {
    ln1: Linear,
    ln2: Linear,
}

impl CheckCxModel {
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(NINPUTS, NHIDDEN, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(NHIDDEN, 1, vs.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }
}

impl Module for CheckCxModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input
            .flatten_from(1)?
            .apply(&self.ln1)?
            .tanh()?
            .apply(&self.ln2)?
            .apply(&sigmoid)
    }
}

pub fn setup_connection(dev: &Device) -> anyhow::Result<CheckCxModel> {
    // check to see if cuda device availabke
    // let dev = candle_core::Device::Cpu;
    info!("Training on device {dev:?}");
    // create a new variable builder
    let weights = candle_core::safetensors::load("connection_pred_weights.st", dev)?;
    let vs = VarBuilder::from_tensors(weights, candle_core::DType::F32, dev);
    let model = CheckCxModel::new(vs.clone())?;

    Ok(model)
}
