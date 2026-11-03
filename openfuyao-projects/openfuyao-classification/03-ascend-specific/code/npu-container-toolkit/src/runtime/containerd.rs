// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

use std::{
    fmt::{self, Display, Formatter},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{bail, Result};
use clap::Parser;
use nix::{sys::signal, unistd::Pid};
use toml_edit as toml;

use crate::util;

use super::Endpoint;

pub struct Runtime {
    endpoint: Endpoint,
}

impl Runtime {
    const DEFAULT_PATH: &str = "/etc/containerd/config.toml";
}

impl super::Runtime for Runtime {
    fn new(endpoint: &str, sysroot: &str) -> Result<Self>
    where Self: Sized {
        let mut runtime = Runtime {
            endpoint: endpoint.parse()?,
        };
        runtime.endpoint.sysroot = sysroot.into();
        Ok(runtime)
    }

    fn config(&self) -> Result<impl super::RuntimeConfig> {
        #[derive(Parser)]
        #[command(ignore_errors = true)]
        struct Cli {
            #[arg(short, long)]
            config: String,
        }

        <RuntimeConfig as super::RuntimeConfig>::new(
            &self.endpoint.sysroot,
            Cli::try_parse_from(self.endpoint.proc()?.cmdline()?)
                .map(|cli| cli.config)
                .as_deref()
                .unwrap_or(Self::DEFAULT_PATH),
        )
    }

    fn restart(&self) -> Result<&Self> {
        signal::kill(Pid::from_raw(self.endpoint.proc()?.stat()?.pid), signal::SIGHUP)?;
        Ok(self)
    }
}

pub struct RuntimeConfig {
    path_: PathBuf,
    doc: toml::DocumentMut,
}

impl RuntimeConfig {
    #[inline]
    fn register_oci_runtime_v2(&mut self, name: &str, binary: &str) {
        let runtimes = self.doc.as_table_mut()["plugins"].or_insert(toml::table())["io.containerd.grpc.v1.cri"]
            .or_insert(toml::table())["containerd"]
            .or_insert(toml::table())["runtimes"]
            .or_insert(toml::table());
        let runtime = runtimes[name].or_insert(toml::table());
        runtime["runtime_type"] = "io.containerd.runc.v2".into();
        runtime["privileged_without_host_devices"] = false.into();
        runtime["options"].or_insert(toml::table())["BinaryName"] = binary.into();
    }

    fn register_oci_runtime_v3(&mut self, name: &str, binary: &str) {
            let runtimes = self.doc.as_table_mut()["plugins"].or_insert(toml::table())["io.containerd.cri.v1.runtime"]
                .or_insert(toml::table())["containerd"]
                .or_insert(toml::table())["runtimes"]
                .or_insert(toml::table());
            let runtime = runtimes[name].or_insert(toml::table());
            runtime["runtime_type"] = "io.containerd.runc.v2".into();
            runtime["privileged_without_host_devices"] = false.into();
            runtime["options"].or_insert(toml::table())["BinaryName"] = binary.into();
        }

}

impl super::RuntimeConfig for RuntimeConfig {
    fn new<P1: AsRef<Path>, P2: AsRef<Path>>(sysroot: P1, path: P2) -> Result<Self> {
        // NOTE: here we assume that `path` is always absolute
        let realpath = util::concat(sysroot, path);
        Ok(RuntimeConfig {
            path_: realpath.as_os_str().into(),
            doc: fs::read_to_string(realpath)?.parse()?,
        })
    }

    #[inline]
    fn path(&self) -> &Path {
        &self.path_
    }

    fn register_oci_runtime<P: AsRef<Path>>(&mut self, name: &str, binary: P) -> Result<&Self> {
        match self.doc["version"].as_integer() {
            Some(2) => {
                self.register_oci_runtime_v2(name, binary.as_ref().to_str().unwrap());
                Ok(self)
            }
            Some(3) => {
                self.register_oci_runtime_v3(name, binary.as_ref().to_str().unwrap());
                Ok(self)
            }
            _ => bail!("unsupported container runtime config"),
        }
    }
}

impl Display for RuntimeConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.doc)
    }
}
