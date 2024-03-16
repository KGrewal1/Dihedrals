use anyhow::anyhow;
use candle_core::Tensor;
use parse_dihedrals::{Dihedral, Dihedrals};
use parse_tsdata::TransitionStates;
use std::{collections::BTreeMap, fs, path::Path};

mod nn_arch;
mod setup_model;
const NDIHEDRALS: usize = 178;
const NCX: usize = 100;

fn main() -> anyhow::Result<()> {
    let pathsample = Path::new("PATHSAMPLE");

    let dev = candle_core::Device::cuda_if_available(0)?;
    let model = setup_model::setup_connection()?;
    let minima: Vec<Dihedral> = fs::read_to_string(pathsample.join("min.dihedrals"))?
        .parse::<Dihedrals>()
        .map_err(|err| anyhow!(err))?
        .into();

    let connections: BTreeMap<(usize, usize), usize> =
        fs::read_to_string(pathsample.join("ts.data"))?
            .parse::<TransitionStates>()
            .map_err(|err| anyhow!(err))?
            .into();

    // let raw = Vec::from_raw_parts();
    let mut count = 0;
    let mut count_cx = 0;
    let mut count_cx_miss = 0;
    let mut count_ucx = 0;
    let mut count_ucx_miss = 0;
    'outer: for (i, min_1) in minima.iter().enumerate() {
        for min_2 in minima.iter().skip(i + 1) {
            let cx = connections.get(&(min_1.id, min_2.id));
            let mut angles: Vec<f64> = Vec::with_capacity(2 * NDIHEDRALS);
            angles.extend(min_1.dihedrals.iter());
            angles.extend(min_2.dihedrals.iter());
            let test_input = Tensor::from_slice(&angles, (1, 1, 2, NDIHEDRALS), &dev)?
                .to_dtype(candle_core::DType::F32)?;
            let pred = model.forward(&test_input)?;
            let pred = pred.reshape(())?;
            let pred: f32 = pred.to_scalar()?;
            if cx.is_none() {
                count_ucx += 1;

                if pred > 0.5 {
                    count_ucx_miss += 1;
                }
                if pred > 0.9999 {
                    println!("{} {} {}", min_1.id, min_2.id, pred);
                    count += 1;
                    if count > NCX {
                        break 'outer;
                    }
                }
            } else {
                count_cx += 1;
                if pred < 0.5 {
                    count_cx_miss += 1;
                }
            }
        }
        // println!("{}", i);
        // if i == 200 {
        //     break;
        // }
    }
    println!(
        "count_cx: {}, count_ucx: {}, count_ucx_miss: {}, count_cx_miss: {}",
        count_cx, count_ucx, count_ucx_miss, count_cx_miss
    );
    Ok(())
}
