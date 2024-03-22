use anyhow::anyhow;
use candle_core::Tensor;
use candle_nn::Module;
use env_logger::Builder;
use log::{info, warn, LevelFilter};
use parse_dihedrals::{Dihedral, Dihedrals};
use parse_tsdata::TransitionStates;
use rayon::prelude::*;
use std::{collections::BTreeMap, fs, path::Path};

// mod nn_arch;
mod setup_model;
const NDIHEDRALS: usize = 178;

fn main() -> anyhow::Result<()> {
    let mut builder = Builder::new();
    builder.format_target(false);
    builder.format_timestamp(None);
    builder.format_level(false);
    builder.filter(None, LevelFilter::Info);
    builder.init();

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

    let (count_cx, count_ucx, count_cx_miss, count_ucx_miss): (u32, u32, u32, u32) = minima
        .par_iter()
        .enumerate()
        .flat_map(|(i, min_1)| {
            minima.par_iter().skip(i + 1).map(|min_2| {
                let mut angles: Vec<f64> = Vec::with_capacity(2 * NDIHEDRALS);
                angles.extend(min_1.dihedrals.iter());
                angles.extend(min_2.dihedrals.iter());
                let pred = Tensor::from_slice(&angles, (1, 1, 2, NDIHEDRALS), &dev)
                    .and_then(|tensor| tensor.to_dtype(candle_core::DType::F32))
                    .and_then(|input| model.forward(&input))
                    .and_then(|pred| pred.reshape(()))
                    .and_then(|pred| pred.to_scalar::<f32>());
                (min_1.id, min_2.id, pred)
            })
        })
        .map(|(min_1, min_2, pred)| {
            if let Ok(pred) = pred {
                let cx = connections.get(&(min_1, min_2));
                // unconnected
                if cx.is_none() {
                    // unconnected misidentified
                    if 0.5 < pred {
                        // unconnected w 'high' likelihood of being connected
                        if 0.999 < pred {
                            info!("prob {}", pred);
                            println!("{} {}", min_1, min_2);
                        }
                        (0, 1, 0, 1)
                    } else {
                        // unconnected identified as unconnected
                        (0, 1, 0, 0)
                    }
                } else {
                    // connected identified as connected
                    if 0.5 < pred {
                        (1, 0, 0, 0)
                    } else {
                        // connected misidentified as unconnected
                        (1, 0, 1, 0)
                    }
                }
            } else {
                // prediction is in error state
                warn!("failed prediction for {} {} {:?}", min_1, min_2, pred);
                (0, 0, 0, 0)
            }
        })
        .reduce(
            || (0, 0, 0, 0),
            |(count_cx, count_ucx, count_ucx_miss, count_cx_miss), (cx, ucx, ucx_miss, cx_miss)| {
                (
                    count_cx + cx,
                    count_ucx + ucx,
                    count_ucx_miss + ucx_miss,
                    count_cx_miss + cx_miss,
                )
            },
        );

    info!(
        "count_cx: {}, count_ucx: {}, count_ucx_miss: {}, count_cx_miss: {}",
        count_cx, count_ucx, count_ucx_miss, count_cx_miss
    );
    Ok(())
}
