use std::{collections::BTreeMap, str::FromStr};
use winnow::{
    ascii::{digit1, float, multispace0, space0, till_line_ending},
    combinator::repeat,
    error::{AddContext, StrContext},
    stream::Stream,
    PResult, Parser,
};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct TransitionState {
    pub ts: usize,
    pub min1: usize,
    pub min2: usize,
}

#[derive(Debug)]
pub struct TransitionStates(Vec<TransitionState>);

impl std::ops::Deref for TransitionStates {
    type Target = Vec<TransitionState>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for TransitionStates {
    // type Target = Vec<TransitionState>;

    fn deref_mut(&mut self) -> &mut Vec<TransitionState> {
        &mut self.0
    }
}

impl From<TransitionStates> for BTreeMap<(usize, usize), usize> {
    fn from(transition_states: TransitionStates) -> Self {
        let mut map = BTreeMap::new();
        for dihedral in transition_states.0 {
            let (min_1, min_2) = if dihedral.min1 < dihedral.min2 {
                (dihedral.min1, dihedral.min2)
            } else {
                (dihedral.min2, dihedral.min1)
            };
            let id = (min_1, min_2);
            map.entry(id).or_insert(dihedral.ts);
        }
        map
    }
}

impl IntoIterator for TransitionStates {
    type Item = TransitionState;
    type IntoIter = <Vec<TransitionState> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl From<TransitionStates> for Vec<TransitionState> {
    fn from(transition_state: TransitionStates) -> Self {
        transition_state.0
    }
}

pub fn parse_line(input: &mut &str) -> PResult<(usize, usize)> {
    multispace0.parse_next(input)?;
    let start = input.checkpoint();
    space0.parse_next(input)?;
    let _float_1: f32 = float.parse_next(input)?;
    space0.parse_next(input)?;
    let _float_2: f32 = float.parse_next(input)?;
    space0.parse_next(input)?;
    let _sym: u8 = digit1.parse_to().parse_next(input).map_err(|err| {
        err.add_context(
            input,
            &start,
            StrContext::Expected(winnow::error::StrContextValue::Description(
                "Unknown point group",
            )),
        )
    })?;
    space0.parse_next(input)?;
    let min1 = digit1.parse_to().parse_next(input).map_err(|err| {
        err.add_context(
            input,
            &start,
            StrContext::Expected(winnow::error::StrContextValue::Description(
                "Unknown first min",
            )),
        )
    })?;
    space0.parse_next(input)?;
    let min2 = digit1.parse_to().parse_next(input).map_err(|err| {
        err.add_context(
            input,
            &start,
            StrContext::Expected(winnow::error::StrContextValue::Description(
                "Unknown second min",
            )),
        )
    })?;
    till_line_ending(input)?;
    Ok((min1, min2))
}

pub fn parse_lines(input: &mut &str) -> PResult<Vec<(usize, usize)>> {
    repeat(0.., parse_line)
        .context(winnow::error::StrContext::Label("Parsing systems"))
        .parse_next(input)
}

impl FromStr for TransitionStates {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut input = s;
        Ok(TransitionStates(
            parse_lines
                .parse_next(&mut input)
                .map_err(|e| e.to_string())?
                .into_iter()
                .enumerate()
                .map(|(i, (min1, min2))| TransitionState {
                    ts: i + 1,
                    min1,
                    min2,
                })
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        #[rustfmt::skip]
        let input =
        "     -265.3681603180    14234.1103288837         1        67       218     1113.5998995034     1254.7791348255     2199.1723510143
              -265.5616256367    14234.6056494403         1        67       180     1160.4501320693     1338.7236517612     2218.3066378356";
        let ts_states = input.parse::<TransitionStates>().unwrap();
        let ts_1 = TransitionState {
            ts: 1,
            min1: 67,
            min2: 218,
        };
        let ts_2 = TransitionState {
            ts: 2,
            min1: 67,
            min2: 180,
        };

        println!("{:?}", ts_states);
        assert_eq!(ts_states.len(), 2);
        assert_eq!(ts_states[0], ts_1);
        assert_eq!(ts_states[1], ts_2);
    }
}
