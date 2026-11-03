/**
 * Playwright E2E 由 CI/本地 .env 注入；在类型层面视为已定义，避免每处非空断言。
 * 若缺少变量，运行期仍会失败。
 */
declare namespace NodeJS {
    interface ProcessEnv {
        TEST_GUIDEURL: string;
        TEST_USERNAME: string;
        TEST_PASSWORD: string;
        TEST_INITIAL_PASSWORD: string;
        TEST_NODE1_IP: string;
        TEST_NODE1_PASSWORD: string;
        TEST_NODE2_IP: string;
        TEST_NODE2_PASSWORD: string;
        TEST_NODE3_IP: string;
        TEST_NODE3_PASSWORD: string;
        TEST_NODE4_IP: string;
        TEST_NODE4_PASSWORD: string;
        TEST_NODE5_IP: string;
        TEST_NODE5_PASSWORD: string;
    }
}
