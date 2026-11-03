/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useEffect, useState } from 'openinula';
import { createPatch } from 'diff';
import { html, parse } from 'diff2html';
import { Diff2HtmlUI } from 'diff2html/lib/ui/js/diff2html-ui';
import 'highlight.js/styles/googlecode.css';
import 'diff2html/bundles/css/diff2html.min.css';
const DiffComponent = ({ 
  diffDataList,
  outputFormat,
  isUseUi,
  id,
  fileListToggle,
}) => {
  const [diffDatas, setDiffDatas] = useState('');
  useEffect(() => {
    createDiffData(diffDataList);
  }, [diffDataList, outputFormat, isUseUi, id, fileListToggle]);

  const createDiffData = (fileList) => {
    let diffJsonList = [];
    fileList.forEach(item => {
      let { fileName, oldHeader, newHeader, oldData, newData, isJsons, isYamls } = item;
      let oldString = oldData || '';
      let newString = newData || '';
      if (isJsons) {
        // 格式化json
        oldString = JSON.stringify(oldData, null, 2);
        newString = JSON.stringify(newData, null, 2);
      } else if (!isYamls) {
        // isYamls 直接使用原始 yaml 数据
        return;
      }
      let args = [fileName || '', oldString, newString, oldHeader || '', newHeader || '', { context: 99999 }];
      // 对比差异
      const diffStr = createPatch(...args);
      // 差异json化
      const diffJsons = parse(diffStr);
      diffJsonList.push(diffJsons[0]);
    });
    if (isUseUi) {
      // 使用diff2html-ui
      const targetElement = document.getElementById(id);
      const configuration = {
        drawFileList: true, matching: 'lines', highlight: true, outputFormat,
      };
      const diff2htmlUis = new Diff2HtmlUI(targetElement, diffJsonList, configuration);
      diff2htmlUis.draw(); // 绘制页面
      diff2htmlUis.highlightCode(); // 高亮数据
      diff2htmlUis.fileListToggle(fileListToggle); // 是否折叠概要
    } else {
      // 使用html方法
      const diffHtmls = html(diffJsonList, {
        drawFileList: true, matching: 'lines', showFiles: true, outputFormat,
      });
      setDiffDatas(diffHtmls);
    }
  };

  return (
    isUseUi ? <div id={id || 'code-diff-ui'} /> : <div id='code-diff' dangerouslySetInnerHTML={{ __html: diffDatas }} />
  );
};

export default DiffComponent;