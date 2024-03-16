use std::path::Path;

use anyhow::Context;
use candle_nn::{VarBuilder, VarMap};
use log::info;

use crate::{nn_arch::MyModel, DATATYPE};

pub fn setup_training() -> anyhow::Result<(MyModel, VarMap)> {
    // check to see if cuda device availabke
    let dev = candle_core::Device::cuda_if_available(0)?;
    info!("Training on device {dev:?}");
    // dev.set_seed(0)?;

    let mut data = candle_core::safetensors::load("dihedral_data.st", &dev)?;

    // let iter = DatasetRandomIter::new(ds, valid, seq_len, device)
    // let batches = Batcher::new2((train_images, train_labels));

    // get the labels from the dataset
    let train_input = data
        .remove("traininput")
        .context("key 'input' not found")?
        .to_dtype(DATATYPE)?;

    let train_output = data
        .remove("trainoutput")
        .context("key 'output' not found")?
        .to_dtype(DATATYPE)?;

    let test_input = data
        .remove("testinput")
        .context("key 'input' not found")?
        .to_dtype(DATATYPE)?;

    let test_output = data
        .remove("testoutput")
        .context("key 'output' not found")?
        .to_dtype(DATATYPE)?;

    // let test_labels = test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    // let test_images = test_images.to_device(&dev)?;

    // create a new variable store
    let mut varmap = VarMap::new();
    // create a new variable builder
    let vs = VarBuilder::from_varmap(&varmap, DATATYPE, &dev);

    let setup = crate::nn_arch::MySetupVars {
        train_data: train_input,
        train_labels: train_output,
        test_data: test_input,
        test_labels: test_output,
    };
    // create model from variables
    let model = MyModel::new(vs.clone(), setup)?;

    if Path::new("weights.st").exists() {
        varmap.load("weights.st")?;
    }
    Ok((model, varmap))
}
