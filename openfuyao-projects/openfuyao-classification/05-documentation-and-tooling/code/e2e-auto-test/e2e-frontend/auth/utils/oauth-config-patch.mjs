/**
 * 直接通过 Kubernetes API（@kubernetes/client-node + kubeconfig）修改 OAuth 相关后端配置，
 * 等价于 auth/test-oauth-related.spec.ts 的 beforeAll/afterAll 通过前端 UI 做的事情，
 * 但不再依赖前端逐个改资源（也不依赖 kubectl 二进制）。
 *
 * 改动的资源（命名空间 openfuyao-system）：
 *   1. ConfigMap  oauth-server-config  -> data.config.json 里的 5 个字段
 *   2. Deployment user-management-operator -> 环境变量 LOCK_DURATION
 *   3. 重启 oauth-server（rollout restart），使 ConfigMap 生效
 *
 * 用法：
 *   node auth/utils/oauth-config-patch.mjs            # 默认 status：只查看当前值
 *   node auth/utils/oauth-config-patch.mjs apply      # 应用测试配置（短超时）
 *   node auth/utils/oauth-config-patch.mjs restore    # 恢复生产配置
 *
 * kubeconfig 解析（按优先级）：--kubeconfig=<path> > 环境变量 KUBECONFIG(可多段) > ~/.kube/config。
 * 不读取仓库内或 cwd 下的本地 kubeconfig。运行时会打印目标集群的 API server 地址以便核对。
 */

import { KubeConfig, CoreV1Api, AppsV1Api } from '@kubernetes/client-node';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// ----------------------------- 配置 -----------------------------
const NS = 'openfuyao-system';
const CONFIGMAP = 'oauth-server-config';
const CONFIG_KEY = 'config.json';
const DEPLOYMENT = 'user-management-operator';
const DEPLOY_CONTAINER = 'user-management-operator';
const OAUTH_DEPLOYMENT = 'oauth-server';

// ConfigMap config.json 字段：[key, 生产值(含引号), 测试值(含引号)]
// 与 test-oauth-related.spec.ts 的 OAUTH_CONFIG_REPLACEMENTS 完全一致
const CM_CHANGES = [
  ['SessionMaxAge', '"7200"', '"60"'],
  ['AuthCodeExp', '"5m"', '"30s"'],
  ['AccessTokenExp', '"5m"', '"30s"'],
  ['RefreshTokenExp', '"2h"', '"60s"'],
  ['LockDuration', '"20m"', '"1m"'],
];

// Deployment 环境变量：[name, 生产值, 测试值]（与 DEPLOYMENT_CONFIG_REPLACEMENTS 一致）
const DEPLOY_CHANGES = [
  ['LOCK_DURATION', '20', '1'],
];

const ROLLOUT_TIMEOUT_MS = 180_000;
// client-node v1.x 的 patch 默认 content-type 是 application/json-patch+json（数组），
// 且无法通过 header 覆盖，因此统一使用 RFC6902 JSON Patch（op/path/value 数组）。

// ----------------------------- 工具 -----------------------------
function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// 解析 kubeconfig 路径（按优先级）：--kubeconfig=<path> 显式参数 > 环境变量 KUBECONFIG(可多段) > ~/.kube/config。
// 显式参数直接采用（文件不存在时由 loadFromFile 抛出明确错误，避免静默落到错误集群）；
// 其余来源按存在性取第一个。不读取仓库内或 cwd 下的本地副本。
function resolveKubeconfigPath() {
  const arg = process.argv.find((a) => a.startsWith('--kubeconfig='));
  if (arg) return arg.slice('--kubeconfig='.length);
  const candidates = [
    ...(process.env.KUBECONFIG ? process.env.KUBECONFIG.split(':').filter(Boolean) : []),
    path.join(os.homedir(), '.kube', 'config'),
  ];
  return candidates.find((c) => c && fs.existsSync(c)) || '';
}

function buildKube() {
  const p = resolveKubeconfigPath();
  if (!p) {
    throw new Error(
      '找不到 kubeconfig。请通过 --kubeconfig=<path> 或环境变量 KUBECONFIG 指定，或确保存在 ~/.kube/config。',
    );
  }
  const kc = new KubeConfig();
  kc.loadFromFile(p);
  // 打印目标集群的 API server 地址，便于操作前确认连的是不是正确的集群
  const server = kc.getCurrentCluster()?.server || '<unknown>';
  const context = kc.currentContext || '<unknown>';
  return { kc, path: p, server, context };
}

// 将 config.json 中各字段“无条件置为”目标值（target: 'prod' | 'test'），返回新文本。
// 幂等：不依赖当前值，重复 apply/restore 也不会因值不匹配而失败；仅当字段不存在时才报错。
function setConfigJsonValues(text, target) {
  let next = text;
  for (const [key, prod, test] of CM_CHANGES) {
    const targetVal = target === 'prod' ? prod : test;
    const re = new RegExp(`"${escapeRe(key)}"\\s*:\\s*"[^"]*"`);
    if (!re.test(next)) {
      throw new Error(`ConfigMap 字段 ${key} 未找到，放弃修改（请先确认 status）`);
    }
    next = next.replace(re, `"${key}": ${targetVal}`);
  }
  return next;
}

async function readConfigJson(core) {
  const cm = await core.readNamespacedConfigMap({ name: CONFIGMAP, namespace: NS });
  const data = cm.data?.[CONFIG_KEY];
  if (data == null) throw new Error(`ConfigMap ${CONFIGMAP} 中找不到 data.${CONFIG_KEY}`);
  return data;
}

function extractCmValues(text) {
  const out = {};
  for (const [key] of CM_CHANGES) {
    const m = text.match(new RegExp(`"${escapeRe(key)}"\\s*:\\s*("[^"]*")`));
    out[key] = m ? m[1] : '<missing>';
  }
  return out;
}

async function readDeployEnv(apps, name) {
  const dep = await apps.readNamespacedDeployment({ name, namespace: NS });
  const c = dep.spec.template.spec.containers.find((x) => x.name === DEPLOY_CONTAINER) ||
    dep.spec.template.spec.containers[0];
  const env = {};
  for (const e of c.env || []) env[e.name] = e.value;
  return env;
}

async function patchConfigMap(core, newText) {
  // 整体替换 data.config.json 的值
  const patch = [{ op: 'replace', path: `/data/${CONFIG_KEY}`, value: newText }];
  await core.patchNamespacedConfigMap({ name: CONFIGMAP, namespace: NS, body: patch });
}

async function patchDeploymentEnv(apps, envName, value) {
  // 先读出当前 env，仅改目标变量后整体替换该容器的 env 数组（避免依赖下标）
  const dep = await apps.readNamespacedDeployment({ name: DEPLOYMENT, namespace: NS });
  const containers = dep.spec.template.spec.containers;
  const ci = containers.findIndex((c) => (c.env || []).some((e) => e.name === envName));
  if (ci < 0) throw new Error(`Deployment ${DEPLOYMENT} 中找不到环境变量 ${envName}`);
  const newEnv = containers[ci].env.map((e) => (e.name === envName ? { ...e, value } : e));
  const patch = [{ op: 'replace', path: `/spec/template/spec/containers/${ci}/env`, value: newEnv }];
  await apps.patchNamespacedDeployment({ name: DEPLOYMENT, namespace: NS, body: patch });
}

async function restartOAuthServer(apps, core) {
  // 删除 oauth-server 的 Pod，由 Deployment 自动重建（等价于测试里删 Pod 的行为）
  const dep = await apps.readNamespacedDeployment({ name: OAUTH_DEPLOYMENT, namespace: NS });
  const labels = dep.spec.selector.matchLabels || dep.spec.selector.matchExpressions;
  if (!labels || Array.isArray(labels)) {
    throw new Error(`无法从 ${OAUTH_DEPLOYMENT} selector 推导 labelSelector`);
  }
  const labelSelector = Object.entries(labels).map(([k, v]) => `${k}=${v}`).join(',');
  await core.deleteCollectionNamespacedPod({ namespace: NS, labelSelector });
}

async function waitForRollout(apps, name) {
  const start = Date.now();
  let last = '';
  while (Date.now() - start < ROLLOUT_TIMEOUT_MS) {
    const d = await apps.readNamespacedDeployment({ name, namespace: NS });
    const s = d.status || {};
    const rep = s.replicas || 0;
    const updated = s.updatedReplicas || 0;
    const ready = s.readyReplicas || 0;
    last = `replicas=${rep} updated=${updated} ready=${ready} unavailable=${s.unavailableReplicas || 0}`;
    if (rep > 0 && updated >= rep && ready >= rep && !s.unavailableReplicas) {
      console.log(`  ✓ ${name} rollout 完成 (${last})`);
      return true;
    }
    await new Promise((r) => setTimeout(r, 3000));
  }
  console.error(`  ✗ ${name} rollout 超时（最后状态：${last}）`);
  return false;
}

// ----------------------------- 命令 -----------------------------
async function status() {
  const { kc, path, server, context } = buildKube();
  const core = kc.makeApiClient(CoreV1Api);
  const apps = kc.makeApiClient(AppsV1Api);
  console.log(`kubeconfig: ${path}`);
  console.log(`cluster: ${context} (server: ${server})  <-- 请确认这是目标测试集群`);
  const text = await readConfigJson(core);
  console.log(`ConfigMap ${CONFIGMAP}/${CONFIG_KEY}:`);
  for (const [k, v] of Object.entries(extractCmValues(text))) console.log(`  ${k}: ${v}`);
  const env = await readDeployEnv(apps, DEPLOYMENT);
  console.log(`Deployment ${DEPLOYMENT} env:`);
  for (const [k] of DEPLOY_CHANGES) console.log(`  ${k}: ${env[k] ?? '<missing>'}`);
}

async function applyOrRestore(mode) {
  const to = mode === 'apply' ? 'test' : 'prod';
  const label = mode === 'apply' ? '应用测试配置' : '恢复生产配置';
  console.log(`==> ${label}（-> ${to}）`);

  const { kc, path, server, context } = buildKube();
  console.log(`  kubeconfig: ${path}`);
  console.log(`  cluster: ${context} (server: ${server})  <-- 请确认这是目标测试集群`);
  const core = kc.makeApiClient(CoreV1Api);
  const apps = kc.makeApiClient(AppsV1Api);

  // 1. ConfigMap（幂等置为目标值）
  const oldText = await readConfigJson(core);
  const newText = setConfigJsonValues(oldText, to);
  await patchConfigMap(core, newText);
  console.log('  ✓ ConfigMap oauth-server-config 已更新');
  for (const [k, v] of Object.entries(extractCmValues(newText))) console.log(`     ${k}: ${v}`);

  // 2. Deployment env（user-management-operator）—— env 变更会自动触发其 rollout
  for (const [name, prod, test] of DEPLOY_CHANGES) {
    const value = to === 'prod' ? prod : test;
    await patchDeploymentEnv(apps, name, value);
    console.log(`  ✓ Deployment ${DEPLOYMENT} env ${name}=${value}`);
  }

  // 3. 重启 oauth-server（删 Pod 让 Deployment 重建）使 ConfigMap 生效
  await restartOAuthServer(apps, core);
  console.log(`  ✓ 已删除 ${OAUTH_DEPLOYMENT} 旧 Pod，等待重建`);
  console.log('  等待 oauth-server rollout...');
  await waitForRollout(apps, OAUTH_DEPLOYMENT);

  console.log(`==> ${label} 完成`);
}

// ----------------------------- main -----------------------------
const cmd = process.argv.find((a) => !a.startsWith('-') && a !== process.argv[0] && a !== process.argv[1] && !a.endsWith('.mjs'));
try {
  if (cmd === 'apply') await applyOrRestore('apply');
  else if (cmd === 'restore') await applyOrRestore('restore');
  else await status();
} catch (err) {
  console.error('失败:', err.response?.body?.message || err.message);
  process.exit(1);
}
