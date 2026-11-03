import { APIRequestContext } from '@playwright/test';
import YAML from 'yaml';
import {
  FUYAO_BASE_URL,
  NODE_PATH,
  COLOCATION_CONFIG_CONFIGMAP,
  VOLCANO_CONFIG_CONFIGMAP,
} from '../constants';

export interface ColocationConfig {
  nodesConfig?: Array<{ name: string }>;
  [key: string]: unknown;
}

export interface VolcanoSchedulerOptions {
  usagePlugin?: {
    enable?: boolean;
    usageThreshold?: {
      enable?: boolean;
      cpu?: number;
      memory?: number;
    };
  };
  [key: string]: unknown;
}

export interface RubikOptions {
  eviction?: {
    enable?: boolean;
    cpuevict?: {
      threshold?: number;
    };
    memoryevict?: {
      threshold?: number;
    };
  };
  [key: string]: unknown;
}

const sleep = async (ms: number): Promise<void> => {
  await new Promise((resolve) => setTimeout(resolve, ms));
};

const readJsonResponse = async (
  response: Awaited<ReturnType<APIRequestContext['get']>>,
  label: string,
): Promise<Record<string, unknown>> => {
  const body = await response.text();
  if (!response.ok()) {
    throw new Error(
      `${label} failed: status=${response.status()} ${response.statusText()}; body=${body}`,
    );
  }
  try {
    return JSON.parse(body) as Record<string, unknown>;
  } catch (err) {
    throw new Error(
      `${label} returned non-JSON: status=${response.status()}; body=${body}; ` +
        `${err instanceof Error ? err.message : String(err)}`,
    );
  }
};

const getConfigMapDataWithRetry = async (
  apiContext: APIRequestContext,
  url: string,
  configName: string,
  requiredKeys: string[],
  timeoutMs = 30000,
  intervalMs = 3000,
): Promise<Record<string, string>> => {
  const deadline = Date.now() + timeoutMs;
  let lastStatus = -1;
  let lastError = '';
  let lastBody = '';

  while (Date.now() < deadline) {
    const response = await apiContext.get(url);
    lastStatus = response.status();
    const rawBody = await response.text();
    lastBody = rawBody;

    let parsed: Record<string, unknown> | null = null;
    try {
      parsed = JSON.parse(rawBody) as Record<string, unknown>;
    } catch (err) {
      lastError = `parse response json failed: ${err instanceof Error ? err.message : String(err)}`;
      await sleep(intervalMs);
      continue;
    }

    const data = parsed.data;
    if (!data || typeof data !== 'object') {
      lastError = `response has no configmap data field (status=${response.status()} ${response.statusText()})`;
      await sleep(intervalMs);
      continue;
    }

    const configData = data as Record<string, string>;
    const missingKeys = requiredKeys.filter((key) => typeof configData[key] !== 'string');
    if (missingKeys.length > 0) {
      lastError = `missing keys: ${missingKeys.join(', ')}`;
      await sleep(intervalMs);
      continue;
    }

    return configData;
  }

  const compactBody = lastBody.length > 1200 ? `${lastBody.slice(0, 1200)}...` : lastBody;
  throw new Error(
    `Failed to fetch ${configName} configmap data after ${timeoutMs}ms (status=${lastStatus}); last error: ${lastError}; last body: ${compactBody}`,
  );
};

export const fetchAllNode = async (apiContext: APIRequestContext): Promise<Record<string, unknown>> => {
  const response = await apiContext.get(FUYAO_BASE_URL + NODE_PATH);
  return readJsonResponse(response, 'GET nodes list');
};

export const fetchNode = async (apiContext: APIRequestContext, nodeName: string): Promise<Record<string, unknown>> => {
  const cacheBuster = `?_t=${Date.now()}`;
  const response = await apiContext.get(`${FUYAO_BASE_URL}${NODE_PATH}${nodeName}${cacheBuster}`);
  return readJsonResponse(response, `GET node ${nodeName}`);
};

export const fetchColocationConfig = async (
  apiContext: APIRequestContext
): Promise<[ColocationConfig, VolcanoSchedulerOptions, RubikOptions]> => {
  const data = await getConfigMapDataWithRetry(
    apiContext,
    FUYAO_BASE_URL + COLOCATION_CONFIG_CONFIGMAP,
    'colocation',
    ['colocation-options', 'volcano-scheduler-options', 'rubik-options'],
  );
  const colocationOptions = JSON.parse(data['colocation-options']) as ColocationConfig;
  const volcanoSchedulerOptions = JSON.parse(data['volcano-scheduler-options']) as VolcanoSchedulerOptions;
  const rubikOptions = JSON.parse(data['rubik-options']) as RubikOptions;
  return [colocationOptions, volcanoSchedulerOptions, rubikOptions];
};

export const fetchVolcanoConfig = async (apiContext: APIRequestContext): Promise<Record<string, unknown>> => {
  const data = await getConfigMapDataWithRetry(
    apiContext,
    FUYAO_BASE_URL + VOLCANO_CONFIG_CONFIGMAP,
    'volcano',
    ['volcano-scheduler.conf'],
  );
  const rawConfig = data['volcano-scheduler.conf'];

  try {
    return JSON.parse(rawConfig) as Record<string, unknown>;
  } catch {
    const parsed = YAML.parse(rawConfig) as Record<string, unknown> | null;
    return parsed ?? {};
  }
};
