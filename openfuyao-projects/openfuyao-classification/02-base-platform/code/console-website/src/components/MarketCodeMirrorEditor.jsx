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
import { EditorState, Compartment } from '@codemirror/state';
import {
  forwardRef,
  useCallback,
  useEffect,
  useLayoutEffect,
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
const MarketCodeMirrorEditor = forwardRef(
  (
    { className = '', yamlData = '', isEdit = true, changeYaml = () => { } },
    ref,
  ) => {
    useImperativeHandle(ref, () => ({ resetCodeEditor }));

    const marketEditorRef = useRef(null);
    const marketViewRef = useRef(null);
    const editableCompartmentRef = useRef(new Compartment());
    const changeYamlRef = useRef(changeYaml);
    changeYamlRef.current = changeYaml;
    const isEditRef = useRef(isEdit);
    isEditRef.current = isEdit;
    // 绘制 codeyaml
    const marketPaintCanvas = useCallback(() => {
      if (!marketEditorRef || !marketEditorRef.current) {
        return;
      }
      // 初始状态
      const marketCodeEditorStartState = EditorState.create({
        doc: yamlData,
        extensions: [
          basicSetup,
          keymap.of([indentWithTab]),
          yaml(),
          onUpdate,
          editableCompartmentRef.current.of(EditorView.editable.of(isEditRef.current ? true : false)),
        ],
      });
      const marketCodeEditorView = new EditorView({
        state: marketCodeEditorStartState,
        parent: marketEditorRef.current,
      });
      marketViewRef.current = marketCodeEditorView;
    }, [marketEditorRef]);

    // 更改
    const marketHandleEdit = useCallback(() => {
      if (marketViewRef.current) {
        marketViewRef.current.dispatch({
          effects: editableCompartmentRef.current.reconfigure(EditorView.editable.of(isEdit ? true : false)),
        });
      }
    }, [isEdit]);

    // 兼容外边数据更改
    useEffect(() => {
      if (!isEdit) {
        if (marketViewRef.current) {
          marketViewRef.current.dispatch({
            changes: { from: 0, to: marketViewRef.current.state.doc.length, insert: yamlData },
            effects: editableCompartmentRef.current.reconfigure(EditorView.editable.of(false)),
          });
        }
      }
    }, [isEdit, yamlData]);

    // 重置
    const resetCodeEditor = () => {
      if (marketViewRef.current) {
        marketViewRef.current.setState(
          EditorState.create({
            doc: '',
            extensions: [
              basicSetup,
              yaml(),
              onUpdate,
              keymap.of([indentWithTab]),
              editableCompartmentRef.current.of(EditorView.editable.of(isEditRef.current ? true : false)),
            ],
          }),
        );
      }
    };

    const onUpdate = EditorView.updateListener.of((v) => {
      changeYamlRef.current(v.state.toJSON().doc); // 字符串string
    });

    useLayoutEffect(() => {
      marketPaintCanvas();
      return () => {
        if (marketViewRef.current) {
          marketViewRef.current.destroy();
          marketViewRef.current = null;
        }
      };
    }, [marketPaintCanvas]);

    useEffect(() => {
      marketHandleEdit();
    }, [marketHandleEdit]);

    return (
      <div
        className={`code_mirror_box ${className}_mirror`}
        ref={marketEditorRef}
      ></div>
    );
  },
);

export default MarketCodeMirrorEditor;
