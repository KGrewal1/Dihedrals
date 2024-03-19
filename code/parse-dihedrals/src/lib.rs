use std::{collections::BTreeMap, str::FromStr};
use winnow::{
    ascii::{digit1, float, multispace0, space0},
    combinator::{alt, repeat},
    error::{AddContext, StrContext},
    stream::Stream,
    PResult, Parser,
};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Dihedral {
    pub id: usize,
    pub dihedrals: Vec<f64>,
}

pub struct Dihedrals(Vec<Dihedral>);

impl std::ops::Deref for Dihedrals {
    type Target = Vec<Dihedral>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Dihedrals {
    // type Target = Vec<TransitionState>;

    fn deref_mut(&mut self) -> &mut Vec<Dihedral> {
        &mut self.0
    }
}

impl IntoIterator for Dihedrals {
    type Item = Dihedral;
    type IntoIter = <Vec<Dihedral> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl From<Dihedrals> for Vec<Dihedral> {
    fn from(dihedrals: Dihedrals) -> Self {
        dihedrals.0
    }
}

impl From<Dihedrals> for BTreeMap<usize, Dihedral> {
    fn from(dihedrals: Dihedrals) -> Self {
        let mut map = BTreeMap::new();
        for dihedral in dihedrals.0 {
            let id = dihedral.id;
            let _entry = map.insert(id, dihedral);
        }
        map
    }
}

fn parse_id(input: &mut &str) -> PResult<usize> {
    let start = input.checkpoint();
    let _system = alt(("minimum", "ts")).parse_next(input).map_err(|err| {
        err.add_context(
            input,
            &start,
            StrContext::Expected(winnow::error::StrContextValue::Description(
                "Valid system (minimum or ts)",
            )),
        )
    })?;
    space0.parse_next(input)?;
    let id: usize = digit1.parse_to().parse_next(input)?;
    Ok(id)
}

fn parse_dihedrals(input: &mut &str) -> PResult<usize> {
    let start = input.checkpoint();
    "dihedrals".parse_next(input).map_err(|err| {
        err.add_context(
            input,
            &start,
            StrContext::Expected(winnow::error::StrContextValue::Description(
                "Expected dihedrals keyword",
            )),
        )
    })?;
    space0.parse_next(input)?;
    let dihedrals: usize = digit1.parse_to().parse_next(input)?;
    Ok(dihedrals)
}

fn parse_angle(input: &mut &str) -> PResult<f64> {
    let angle: f64 = float.parse_next(input)?;
    multispace0.parse_next(input)?;
    Ok(angle)
}

fn parse_system(input: &mut &str) -> PResult<Dihedral> {
    let id = parse_id
        .context(winnow::error::StrContext::Label("Parsing ID"))
        .parse_next(input)?;
    multispace0.parse_next(input)?;
    let number_dihedrals = parse_dihedrals
        .context(winnow::error::StrContext::Label(
            "Parsing number of dihedrals",
        ))
        .parse_next(input)?;
    multispace0.parse_next(input)?;
    let dihedrals = repeat(number_dihedrals..=number_dihedrals, parse_angle)
        .context(winnow::error::StrContext::Label("Parsing dihedral angles"))
        .parse_next(input)?;
    Ok(Dihedral { id, dihedrals })
}

impl FromStr for Dihedral {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut input = s;
        parse_system(&mut input).map_err(|e| e.to_string())
    }
}

impl FromStr for Dihedrals {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut input = s;
        Ok(Dihedrals(
            repeat(0.., parse_system)
                .context(winnow::error::StrContext::Label("Parsing systems"))
                .parse_next(&mut input)
                .map_err(|e| e.to_string())?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_pasre() {
        #[rustfmt::skip]
        let system =
        "minimum 5
        dihedrals 3
        1E3 2.0 3.0";
        let expected = Dihedral {
            id: 5,
            dihedrals: vec![1E3, 2.0, 3.0],
        };
        let system: Dihedral = system.parse().unwrap();
        assert_eq!(system, expected);
    }

    #[test]
    fn test_multi_pasre() {
        #[rustfmt::skip]
        let system =
        "minimum 5
        dihedrals 3
        1E3 2.0 3.0
        minimum 8
        dihedrals 3
        5E3 9.0 12";
        let expected_1 = Dihedral {
            id: 5,
            dihedrals: vec![1E3, 2.0, 3.0],
        };
        let expected_2 = Dihedral {
            id: 8,
            dihedrals: vec![5E3, 9.0, 12.0],
        };
        let system: Dihedrals = system.parse().unwrap();
        assert_eq!(system[0], expected_1);
        assert_eq!(system[1], expected_2);
    }
}
