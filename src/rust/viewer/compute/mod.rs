//! Browser-side numerical pipelines.

#[path = "../../compute/browser/mod.rs"]
mod browser;
#[path = "../../compute/source/mod.rs"]
mod source;

pub(crate) use browser::ComputeEngine;
