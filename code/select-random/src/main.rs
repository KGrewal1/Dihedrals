use anyhow::anyhow;
use env_logger::Builder;
use log::{info, LevelFilter};
use parse_dihedrals::{Dihedral, Dihedrals};
use parse_tsdata::TransitionStates;
use rand::{seq::SliceRandom, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use std::{collections::BTreeMap, fs, path::Path};

fn main() -> anyhow::Result<()> {
    let mut builder = Builder::new();
    builder.format_target(false);
    builder.format_timestamp(None);
    builder.format_level(false);
    builder.filter(None, LevelFilter::Info);
    builder.init();

    let pathsample = Path::new("PATHSAMPLE");
    let minima: Vec<Dihedral> = fs::read_to_string(pathsample.join("min.dihedrals"))?
        .parse::<Dihedrals>()
        .map_err(|err| anyhow!(err))?
        .into();

    let connections: BTreeMap<(usize, usize), usize> =
        fs::read_to_string(pathsample.join("ts.data"))?
            .parse::<TransitionStates>()
            .map_err(|err| anyhow!(err))?
            .into();

    let mut connected_mins: Vec<(usize, usize)> = Vec::with_capacity(2 * connections.len());
    let mut unconnected_mins: Vec<(usize, usize)> = Vec::with_capacity(minima.len() * minima.len());

    (1..=minima.len()).for_each(|i| {
        (i + 1..=minima.len()).for_each(|j| {
            let cx = connections.get(&(i, j));
            if cx.is_some() {
                connected_mins.push((i, j));
            } else {
                unconnected_mins.push((i, j));
            }
        });
    });

    info!("Connected: {}", connected_mins.len());
    info!("Unconnected: {}", unconnected_mins.len());

    let ntrain_ucx = 10_000;

    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(42);
    rng.long_jump();

    unconnected_mins.shuffle(&mut rng);

    for (min_1, min_2) in unconnected_mins.into_iter().take(ntrain_ucx) {
        println!("{} {}", min_1, min_2);
    }
    Ok(())
}
