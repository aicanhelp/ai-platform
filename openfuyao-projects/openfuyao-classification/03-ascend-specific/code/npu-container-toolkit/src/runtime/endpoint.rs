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
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{anyhow, bail, Error, Result};
use fluent_uri::{component::Scheme, UriRef};
use procfs::{
    net::{self, UnixNetEntry, UnixState},
    process::{self, FDTarget, Process},
};

use crate::util::{self, r#enum};

r#enum!(EndpointScheme: &'static str {
    UNIX => "unix",
});

#[derive(PartialEq)]
pub struct Endpoint {
    pub sysroot: PathBuf,
    uri: UriRef<String>,
    find_proc: fn(uri: UriRef<&str>, sysroot: &Path) -> Result<Process>,
}

impl Endpoint {
    #[inline]
    pub fn proc(&self) -> Result<Process> {
        (self.find_proc)(self.uri.borrow(), &self.sysroot)
    }
}

impl FromStr for Endpoint {
    type Err = Error;

    fn from_str(endpoint: &str) -> std::result::Result<Self, Self::Err> {
        let uri: UriRef<String> = endpoint.parse()?;
        let find_proc = match uri.scheme().map(Scheme::as_str) {
            Some(EndpointScheme::UNIX) => get_proc_from_unix,
            _ => bail!("unsupported container runtime endpoint: {endpoint}"),
        };
        Ok(Endpoint {
            sysroot: "/".into(),
            uri,
            find_proc,
        })
    }
}

#[inline]
fn get_proc_from_unix(uri: UriRef<&str>, sysroot: &Path) -> Result<Process> {
    // NOTE: here we assume that uri path is always absolute
    let sock_inode_in_fsns = util::concat(sysroot, uri.path().decode().into_string()?.to_string())
        .metadata()?
        .ino();

    #[cfg(debug_assertions)]
    eprintln!("inode: {sock_inode_in_fsns}");

    let sock_inode = net::unix()?
        .into_iter()
        .find_map(|UnixNetEntry { path, state, inode, .. }| {
            (state == UnixState::UNCONNECTED
                && match path {
                    Some(path) => {
                        #[cfg(debug_assertions)]
                        eprintln!("path: {path:?}");

                        match util::concat(sysroot, path).metadata() {
                            Ok(metadata) => {
                                #[cfg(debug_assertions)]
                                eprintln!("inode: {}", metadata.ino());

                                metadata.ino() == sock_inode_in_fsns
                            }
                            _ => false,
                        }
                    }
                    _ => false,
                })
            .then_some(inode)
        })
        .ok_or(anyhow!("failed to find the unix domain socket: {uri}"))?;
    process::all_processes()?
        .filter_map(Result::ok)
        .find(|proc| {
            proc.fd()
                .iter_mut()
                .flatten()
                .filter_map(Result::ok)
                .any(|fd| matches!(fd.target, FDTarget::Socket(i) if i == sock_inode))
        })
        .ok_or(anyhow!(
            "failed to obtain the owner process of unix domain socket: {uri}"
        ))
}
