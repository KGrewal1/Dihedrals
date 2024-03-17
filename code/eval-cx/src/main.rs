use anyhow::Context;

mod setup_model;
fn main() -> anyhow::Result<()> {
    let dev = candle_core::Device::cuda_if_available(0)?;
    let model = setup_model::setup_connection(&dev)?;
    let input_tensors = candle_core::safetensors::load("dihedral_class_data.st", &dev)?;

    let train_input = input_tensors
        .get("traininput")
        .context("Missing training data")?;
    let train_output = input_tensors
        .get("trainoutput")
        .context("Missing training data")?;

    let test_input_cx = input_tensors
        .get("testinputcx")
        .context("Missing training data")?;

    let test_input_ucx = input_tensors
        .get("testinputucx")
        .context("Missing training data")?;

    let preds_train = model.forward(train_input)?;

    let preds = preds_train.gt(0.5)?;
    let n_preds = preds.flatten_all()?.dims1()?;
    let n_pred_cx = preds
        .to_dtype(candle_core::DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let pred_correct = &preds
        .eq(&train_output.to_dtype(candle_core::DType::U8)?)?
        .to_dtype(candle_core::DType::F32)?;
    let n_pred_correct = pred_correct.sum_all()?.to_scalar::<f32>()?;
    let n_pred_correct_cx = (pred_correct * preds.to_dtype(candle_core::DType::F32)?)?
        .sum_all()?
        .to_scalar::<f32>()?;

    // println!("Predicted: {}", n_preds);
    // println!("Predicted cx: {}", n_pred_cx);
    let n_pred_incorrect = n_preds as f32 - n_pred_correct;
    let tp = n_pred_correct_cx;
    let flp = n_pred_cx - n_pred_correct_cx;
    let tn = n_pred_correct - n_pred_correct_cx;
    let fln = n_pred_incorrect - flp;
    println!("-----------For Training Data-----------");
    println!("Correctly predicted: {}", n_pred_correct);
    println!("True Positives: {}", tp);
    println!("False Positives: {}", flp);
    println!("True Negatives: {}", tn);
    println!("False Negatives: {}", fln);

    let preds_test_cx = model.forward(test_input_cx)?;
    let n_cx = preds_test_cx.flatten_all()?.dims1()? as f32;
    let preds_test_cx = preds_test_cx.ge(0.5)?;
    let n_pred_cx = preds_test_cx
        .to_dtype(candle_core::DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let tp = n_pred_cx;
    let fln = n_cx - n_pred_cx;

    let preds_test_ucx = model.forward(test_input_ucx)?;
    let n_ucx = preds_test_ucx.flatten_all()?.dims1()? as f32;
    let preds_test_ucx = preds_test_ucx.gt(0.5)?;
    let n_pred_cx = preds_test_ucx
        .to_dtype(candle_core::DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let flp = n_pred_cx;
    let tn = n_ucx - flp;

    println!("-----------For Testing Data-----------");
    println!("Correctly predicted: {}", tp + tn);
    println!("True Positives: {}", tp);
    println!("False Positives: {}", flp);
    println!("True Negatives: {}", tn);
    println!("False Negatives: {}", fln);

    Ok(())
}
