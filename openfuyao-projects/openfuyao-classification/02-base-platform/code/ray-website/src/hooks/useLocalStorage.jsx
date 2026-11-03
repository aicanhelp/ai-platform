/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useState, useEffect } from 'openinula';

export default function useLocalStorage(key, initValue) {
  const [value, setValue] = useState(() => localStorage.getItem(key) ? localStorage.getItem(key) : initValue);

  const handleLocalStorageChange = (e) => {
    if (e.key === key) {
      setValue(e.newValue ? e.newValue : initValue);
    }
  };

  const handleMainFrameStorageChange = (e) => {
    if (e.detail.key === key) {
      setValue(e.detail.newValue ? e.detail.newValue : initValue);
    }
  };

  const setLocalStorage = (newLocalValue) => {
    localStorage.setItem(key, newLocalValue);
    setValue(newValue);
  };

  useEffect(() => {
    window.addEventListener('storage', handleLocalStorageChange);
    return () => {
      window.removeEventListener('storage', handleLocalStorageChange);
    };
  }, [key, initValue]);

  useEffect(() => {
    window.addEventListener('localStorageChanged', handleMainFrameStorageChange);
    return () => {
      window.removeEventListener('localStorageChanged', handleMainFrameStorageChange);
    };
  }, [key, initValue]);

  return [value, setLocalStorage];
}