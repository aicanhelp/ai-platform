// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// path to root directory
    #[arg(short = 'R', long, value_name = "DIR", default_value = "/")]
    pub sysroot: String,
}

#[derive(Subcommand)]
#[clap(rename_all = "kebab_case")]
pub enum Commands {
    /// prepare for NPU container runtime injection
    Prepare {
        #[command(flatten)]
        cri_runtime: CRIRuntimeArgs,

        #[command(flatten)]
        oci_runtime: OCIRuntimeArgs,

        #[command(flatten)]
        output: OutputArgs,
    },

    /// inject NPU container runtime
    Inject {
        #[command(flatten)]
        cri_runtime: CRIRuntimeArgs,

        /// path to the modified CRI runtime config
        #[arg(value_name = "CONFIG")]
        config: String,
    },
}

#[derive(Args)]
pub struct CRIRuntimeArgs {
    /// CRI runtime name
    #[arg(value_name = "CRI_RUNTIME", id = "cri.name")]
    pub name: CRIRuntimes,

    /// URI to CRI runtime endpoint
    #[arg(value_name = "CRI_ENDPOINT", id = "cri.endpoint")]
    pub endpoint: String,
}

#[derive(ValueEnum, Clone)]
#[clap(rename_all = "kebab_case")]
pub enum CRIRuntimes {
    Containerd,
}

#[derive(Args)]
pub struct OCIRuntimeArgs {
    /// OCI runtime name
    #[arg(value_name = "OCI_RUNTIME", id = "oci.name")]
    pub name: String,

    /// path to OCI runtime binary
    #[arg(value_name = "OCI_BINARY", id = "oci.binary")]
    pub binary: String,
}

#[derive(Args)]
#[group(required = true, multiple = true)]
pub struct OutputArgs {
    /// output directory to the modified CRI runtime config
    #[arg(short = 'C', long = "out-dir", value_name = "DIR")]
    pub dir: Option<String>,

    /// output path to the modified CRI runtime config
    #[arg(short = 'o', long = "output", value_name = "FILE")]
    pub file: Option<String>,
}

impl OutputArgs {
    pub fn check(&self) -> Result<(Option<&str>, Option<&str>)> {
        if self.file.as_ref().is_some_and(|x| x == "-") {
            if self.dir.is_some() {
                bail!("--out-dir is not supported when output to stdout")
            }
            return Ok((None, self.file.as_deref()));
        }
        Ok((self.dir.as_deref().or(Some(".")), self.file.as_deref()))
    }
}
