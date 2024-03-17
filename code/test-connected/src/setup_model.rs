use candle_nn::VarBuilder;
use log::info;

use crate::nn_arch::CheckCxModel;

pub fn setup_connection() -> anyhow::Result<CheckCxModel> {
    // check to see if cuda device availabke
    let dev = candle_core::Device::cuda_if_available(0)?;
    info!("Training on device {dev:?}");
    // create a new variable builder
    let weights = candle_core::safetensors::load("connection_pred_Weights.st", &dev)?;
    let vs = VarBuilder::from_tensors(weights, candle_core::DType::F32, &dev);
    let model = CheckCxModel::new(vs.clone())?;

    Ok(model)
}
