import { useCallback, useEffect, useRef, useState } from 'openinula';

export const LOADING_DELAY = 300;

/**
 * 按钮延迟 loading：超过 delay（默认 300ms）才显示 spinner，
 * 快速完成的请求不会出现 loading 闪烁、文字被挤开的问题
 * @param {number} delay - 延迟显示 loading 的毫秒数
 * @returns {[boolean, Function, boolean]} [visibleLoading, setLoading, pending]
 *   - visibleLoading：延迟显示的 loading，用于 Button loading
 *   - setLoading：与 useState 相同的 setter
 *   - pending：立即置位的真实进行中状态，用于 disabled / 重入守卫
 */
export default function useButtonLoading(delay = LOADING_DELAY) {
  const [loading, setLoading] = useState(false);
  const [pending, setPending] = useState(false);
  const timer = useRef(null);

  useEffect(() => () => clearTimeout(timer.current), []);

  const setButtonLoading = useCallback(
    (value) => {
      clearTimeout(timer.current);
      setPending(value);
      if (value) {
        timer.current = setTimeout(() => setLoading(true), delay);
      } else {
        setLoading(false);
      }
    },
    [delay],
  );

  return [loading, setButtonLoading, pending];
}
