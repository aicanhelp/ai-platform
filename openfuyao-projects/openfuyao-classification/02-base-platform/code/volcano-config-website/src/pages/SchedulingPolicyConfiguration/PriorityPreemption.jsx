/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { CloseOutlined } from '@ant-design/icons';
import '@/styles/components/markdown.less';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeSanitize from 'rehype-sanitize';

export default function PriorityPreemption({ className = '', onClose, data }) {
  const handleClose = () => {
    onClose();
  };

  return (
    <div className='explanatoryPanel'>
      <div className='explanatoryPanel_header'>
        <div></div>
        <div><CloseOutlined onClick={handleClose} /></div>
      </div>
      <Markdown
        className={`markdown-body ${className}`}
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeSanitize]}
      >{data}</Markdown>
    </div>
  );
};