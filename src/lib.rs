//! Wake-phrase detection for Oqto, library-first.
//!
//! This crate owns wake model artifacts, evaluation and streaming detection only.
//! A detection is evidence that a phrase was heard: it never authenticates a speaker,
//! selects a Session or authorizes an action. Callers (eaRS, Oqto, Foxline) decide what
//! it means.
//!
//! The product backend is [`detector`] (a frozen speech-embedding front end plus one
//! small classifier per phrase, loaded from a release folder). The legacy rustpotter
//! backend is kept behind the non-default `rustpotter` feature for the pilot record.

pub mod detector;
pub mod protocol;

#[cfg(feature = "rustpotter")]
mod legacy_rustpotter;
#[cfg(feature = "rustpotter")]
pub use legacy_rustpotter::*;
