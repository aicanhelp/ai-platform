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
    convert::Infallible,
    ffi::OsStr,
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    thread,
};

use anyhow::Result;
use signal_hook::{
    consts::{SIGHUP, SIGINT, SIGQUIT, SIGTERM},
    iterator::Signals,
    low_level::emulate_default_handler,
};

macro_rules! r#enum {
    ($vis:vis $name:ident: $type:ty {
        $($variant:ident => $value:expr),*
        $(,)?
    }) => {
        #[non_exhaustive]
        $vis struct $name;

        impl $name {
            $(pub const $variant: $type = $value;)*
        }
    };
}

pub(crate) use r#enum;

#[inline]
pub fn concat<P1: AsRef<Path>, P2: AsRef<Path>>(p1: P1, p2: P2) -> PathBuf {
    p1.as_ref()
        .join(OsStr::from_bytes(&p2.as_ref().as_os_str().as_bytes()[1..]))
}

#[inline]
pub fn park<F, T>(on_exit: F) -> Result<Infallible>
where
    F: (Fn() -> T) + Send + 'static,
    T: Send + 'static, {
    let mut signals = Signals::new([SIGHUP, SIGINT, SIGQUIT, SIGTERM])?;
    thread::spawn(move || -> Result<Infallible> {
        for signal in signals.forever() {
            on_exit();
            emulate_default_handler(signal)?;
        }
        unreachable!()
    });
    thread::park();
    unreachable!()
}
