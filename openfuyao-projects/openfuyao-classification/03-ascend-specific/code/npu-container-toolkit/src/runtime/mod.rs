// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

mod containerd;
mod endpoint;

use std::{fmt::Display, path::Path};

use anyhow::Result;

use crate::cli::CRIRuntimes;
pub use endpoint::Endpoint;

pub fn new(runtime: &CRIRuntimes, endpoint: &str, sysroot: &str) -> Result<impl Runtime> {
    match runtime {
        CRIRuntimes::Containerd => containerd::Runtime::new(endpoint, sysroot),
    }
}

pub trait Runtime {
    fn new(endpoint: &str, sysroot: &str) -> Result<Self>
    where Self: Sized;

    fn config(&self) -> Result<impl RuntimeConfig>;

    fn restart(&self) -> Result<&Self>;
}

pub trait RuntimeConfig: Display {
    fn new<P1: AsRef<Path>, P2: AsRef<Path>>(sysroot: P1, path: P2) -> Result<Self>
    where Self: Sized;

    fn path(&self) -> &Path;

    fn register_oci_runtime<P: AsRef<Path>>(&mut self, name: &str, binary: P) -> Result<&Self>;
}
