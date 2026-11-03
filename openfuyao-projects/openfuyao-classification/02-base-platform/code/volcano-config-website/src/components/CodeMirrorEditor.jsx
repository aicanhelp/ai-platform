/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { basicSetup } from 'codemirror';
import { yaml } from '@codemirror/lang-yaml';
import { EditorView, keymap } from '@codemirror/view';
import { EditorState } from '@codemirror/state';
import {
  forwardRef,
  useCallback,
  useEffect,
  useRef,
  useImperativeHandle,
} from 'openinula';
import '@/styles/components/codeMirrorEditor.less';
import { indentWithTab } from '@codemirror/commands';

{
  /** yaml编辑器
  className 绑定类名
  yamlData 传递yaml数据
  changeYaml 编辑时回掉方法
  clearFn 重置回调
  isEdit 是否可编辑
*/
}
let viewRef = null;
const CodeMirrorEditor = forwardRef(
  (
    { className = '', yamlData = '', isEdit = true, changeYaml = () => { } },
    ref,
  ) => {
    useImperativeHandle(ref, () => ({ resetCodeEditor }));

    const editorRef = useRef(null);
    // 绘制 codeyaml
    const paintCanvas = useCallback(() => {
      if (!editorRef || !editorRef.current) {
        return;
      }
      // 初始状态
      const codeEditorStartState = EditorState.create({
        doc: yamlData,
        extensions: [
          keymap.of([indentWithTab]),
          basicSetup,
          onUpdate,
          yaml(),
          EditorView.editable.of(isEdit ? true : false),
        ],
      });
      const codeEditorView = new EditorView({
        state: codeEditorStartState,
        parent: editorRef.current,
      });
      viewRef = codeEditorView;

      () => {
        codeEditorView.destroy();
      };
    }, [editorRef]);

    // 更改
    const handleEdit = useCallback(() => {
      if (viewRef) {
        viewRef.setState(
          EditorState.create({
            doc: yamlData,
            extensions: [
              yaml(),
              basicSetup,
              onUpdate,
              keymap.of([indentWithTab]),
              EditorView.editable.of(isEdit ? true : false),
            ],
          })
        );
      }
    }, [isEdit]);

    const onUpdate = EditorView.updateListener.of((v) => {
      changeYaml(v.state.toJSON().doc); // 字符串string
    });

    // 重置
    const resetCodeEditor = (doc = '') => {
      if (viewRef) {
        viewRef.setState(
          EditorState.create({
            extensions: [basicSetup, yaml(), onUpdate, keymap.of([indentWithTab])],
            doc,
          })
        );
      }
    };

    useEffect(() => {
      handleEdit();
    }, [handleEdit]);

    useEffect(() => {
      paintCanvas();
    }, [paintCanvas]);

    return (
      <div
        className={`code_mirror_box ${className}_mirror`}
        ref={editorRef}
      ></div>
    );
  }
);

export default CodeMirrorEditor;
