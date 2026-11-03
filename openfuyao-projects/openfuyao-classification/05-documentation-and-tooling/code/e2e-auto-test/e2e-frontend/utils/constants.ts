import { env } from 'node:process';

export const FUYAO_BASE_URL = env.TEST_FUYAOURL ?? '';
export const FUYAO_USERNAME = env.TEST_USERNAME ?? '';
export const FUYAO_PASSWORD = env.TEST_PASSWORD ?? '';