use candle_core::Device;
use candle_nn::VarBuilder;
use connection_network::CheckCxModel;
use log::info;

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
