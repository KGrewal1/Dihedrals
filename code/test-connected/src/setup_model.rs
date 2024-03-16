use std::path::Path;

use candle_nn::{VarBuilder, VarMap};

use crate::nn_arch::CheckCxModel;

pub fn setup_connection() -> anyhow::Result<(CheckCxModel, VarMap)> {
    // check to see if cuda device availabke
    let dev = candle_core::Device::cuda_if_available(0)?;
    println!("Training on device {dev:?}");
    let mut varmap = VarMap::new();
    // create a new variable builder
    let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);

    // create model from variables
    let model = CheckCxModel::new(vs.clone())?;

    if Path::new("connection_pred_Weights.st").exists() {
        println!("Weights loaded");
        varmap.load("connection_pred_Weights.st")?;
    }
    Ok((model, varmap))
}
