/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import React, { useState, useRef, useMemo, useCallback, useEffect } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { coldarkCold } from 'react-syntax-highlighter/dist/esm/styles/prism';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import mermaid from 'mermaid';

export default function SpecialCode({ inline, ...extra }) {
  if (inline) {
    return <RenderInline {...extra} />;
  };
  return <RenderBlock {...extra} />;
};

function RenderInline({ children = [] }) {
  const txt = children[0] || '';
  if (typeof txt === 'string' && /^\$\$(?<content>.*)\$\$/.test(txt)) {
    const match = /^\$\$(?<content>.*)\$\$/.exec(txt);
    const content = match?.groups?.content;
    if (content) {
      const html = katex.renderToString(content, { throwOnError: false });
      return (
        <code
          style={{ backgroundColor: 'rgb(227, 234, 242)', padding: '0.2rem', borderRadius: '0.2rem' }}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      );
    };
  };
  return <code>{txt}</code>;
};

function RenderBlock({ children = [], className, ...props }) {
  const markId = useRef(`mark${randomId()}`);
  const code = getCode(children);
  const [transformCode, setTransformCode] = useState('');

  const codeType = useMemo(() => {
    if (typeof code !== 'string' || typeof className !== 'string') {
      return false;
    };
    if (/^language-mermaid/.test(className.toLocaleLowerCase())) {
      return 'mermaid';
    } else if (/^language-katex/.test(className.toLocaleLowerCase())) {
      return 'katex';
    } else {
      return 'unknown'; // 或者其他默认类型，比如 'plain'，视具体需求而定
    };
  }, [code, className]);

  const renderMermaid = useCallback(async (params) => {
    const { svg } = await mermaid.render(markId.current, params);
    setTransformCode(svg);
  }, []);

  const renderKatex = useCallback(async (params) => {
    const strCode = katex.renderToString(params, { throwOnError: false });
    setTransformCode(strCode);
  }, []);

  useEffect(() => {
    if (codeType === 'mermaid') {
      renderMermaid(code);
      return; // 确保每个分支都有返回语句
    } else if (codeType === 'katex') {
      renderKatex(code);
      return;
    } else {
      return; // 确保每个分支都有返回语句
    }
  }, [code, codeType]);

  if (codeType === 'mermaid') {
    return (
      <div className='code-block' style={{ textAlign: 'center' }}>
        <code dangerouslySetInnerHTML={{ __html: transformCode }} />
      </div>
    );
  } else if (codeType === 'katex') {
    return (
      <code
        className='code-block'
        dangerouslySetInnerHTML={{ __html: transformCode }}
      />
    );
  } else {
    const match = /language-(?:\w+)/.exec(className || '');
    const language = match?.[0]?.split('-')[1]; // 提取匹配结果的第一个元素，并使用split函数来获取语言信息
    return (
      <SyntaxHighlighter
        language={language}
        showLineNumbers={false}
        style={coldarkCold}
        PreTag='div'
        className='syntax-hight-wrapper'
        {...props}
      >
        {children}
      </SyntaxHighlighter>
    );
  }
}

function randomId() {
  const array = new Uint8Array(16);
  window.crypto.getRandomValues(array);
  return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
}

function getCode(arr) {
  if (!Array.isArray(arr)) {
    return '';
  }

  return arr.map((_dt) => {
    const dt = _dt;
    if (typeof dt === 'string') {
      return dt;
    }
    if (dt.props && dt.props.children) {
      return getCode(dt.props.children);
    }
    return false;
  }).filter(Boolean).join('');
};


