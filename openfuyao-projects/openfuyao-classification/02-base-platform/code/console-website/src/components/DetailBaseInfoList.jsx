/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 */
import { isValidElement } from 'openinula';

function isEmptyValue(value) {
  return value === null || value === undefined || value === '';
}

function renderDisplayValue(value) {
  if (isEmptyValue(value)) {
    return '--';
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  if (typeof value === 'string') {
    return value;
  }
  if (Array.isArray(value)) {
    return value.length ? value.join(', ') : '--';
  }
  if (isValidElement(value)) {
    return value;
  }
  // 普通对象无法直接作为 React child 渲染
  return '--';
}

function chunkPairs(list, size) {
  const rows = [];
  for (let i = 0; i < list.length; i += size) {
    rows.push(list.slice(i, i + size));
  }
  return rows;
}

/**
 * @typedef {Object} DetailBaseInfoItem
 * @property {string} key 标签文案（须为字符串）
 * @property {string | number | boolean | string[] | import('openinula').ReactNode} [value]
 * @property {string} [descriptionClassName] 追加在 base_description 上的 class
 * @property {object} [keyStyle] base_key 内联样式
 * @property {object} [valueStyle] base_value 内联样式
 */

/**
 * 详情页基本信息键值栅格
 * @param {DetailBaseInfoItem[]} [items=[]]
 * @param {1|2} [columns=2] 1 单列纵排；2 双列，顺序为先左后右再下一行
 */
export default function DetailBaseInfoList({ items = [], columns = 2 }) {
  const col = columns === 1 ? 1 : 2;
  const rows = chunkPairs(items, col);

  return (
    <div
      className="base_info_list"
    >
      {rows.map((row, rowIndex) => (
        <div className="flex_item_opt" key={`row-${rowIndex}`}>
          {row.map((item, cellIndex) => {
            const display = renderDisplayValue(item.value);
            const isPlainText = typeof display === 'string';
            const descClass = ['base_description', item.descriptionClassName].filter(Boolean).join(' ');
            return (
              <div
                className={descClass}
                key={`${rowIndex}-${cellIndex}-${item.key}`}
              >
                <p className="base_key" style={item.keyStyle}>{item.key}</p>
                {isPlainText ? (
                  <p className="base_value" style={item.valueStyle}>{display}</p>
                ) : (
                  <div className="base_value" style={item.valueStyle}>{display}</div>
                )}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
