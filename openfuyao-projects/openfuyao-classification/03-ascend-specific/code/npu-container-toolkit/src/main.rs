// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

mod cli;
mod runtime;
mod util;

use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
    thread,
    time::Duration,
};

use anyhow::Result;
use clap::Parser;
use nix::mount::{self, MsFlags};

use cli::{CRIRuntimeArgs, Cli, Commands, OCIRuntimeArgs, OutputArgs};
use runtime::{Runtime, RuntimeConfig};

const INTERVAL: Duration = Duration::from_secs(1);

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Prepare {
            cri_runtime: cri,
            oci_runtime: oci,
            output,
        } => prepare(&cri, &oci, &cli.sysroot, &output),
        Commands::Inject {
            cri_runtime: cri,
            config,
        } => inject(cri, cli.sysroot, &config),
    }
}

#[inline]
fn prepare(cri: &CRIRuntimeArgs, oci: &OCIRuntimeArgs, sysroot: &str, output: &OutputArgs) -> Result<()> {
    let (dir, file) = output.check()?;
    let runtime = runtime::new(&cri.name, &cri.endpoint, sysroot)?;
    let mut config = runtime.config()?;
    let mut writer = BufWriter::<Box<dyn Write>>::new(if file == Some("-") {
        Box::new(io::stdout())
    } else {
        Box::new(File::create(
            Path::new(dir.unwrap_or(".")).join(file.or(config.path().file_name().unwrap().to_str()).unwrap()),
        )?)
    });
    writer.write_all(
        config
            .register_oci_runtime(&oci.name, &oci.binary)?
            .to_string()
            .as_bytes(),
    )?;
    Ok(())
}

#[inline]
fn inject(cri: CRIRuntimeArgs, sysroot: String, path: &str) -> Result<()> {
    let runtime = runtime::new(&cri.name, &cri.endpoint, &sysroot)?;
    let config = runtime.config()?;
    let dst = config.path().to_owned();
    mount::mount(Some(path), &dst, None::<&str>, MsFlags::MS_BIND, None::<&str>)?;
    thread::sleep(INTERVAL);
    runtime.restart()?;
    util::park(move || -> Result<()> {
        mount::umount(&dst)?;
        thread::sleep(INTERVAL);
        runtime::new(&cri.name, &cri.endpoint, &sysroot)?.restart()?;
        Ok(())
    })?;
    Ok(())
}
