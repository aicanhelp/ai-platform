export const TEST_FUYAOURL = process.env.TEST_FUYAOURL;
export const TEST_USERNAME = process.env.TEST_USERNAME;
export const TEST_PASSWORD = process.env.TEST_PASSWORD;
export const TEST_INITIAL_PASSWORD = process.env.TEST_INITIAL_PASSWORD;

export const CLUSTER_ADMIN_USERNAME = "clusteradmin";
export const CLUSTER_ADMIN_INITIAL_PASSWORD = TEST_INITIAL_PASSWORD;
export const CLUSTER_ADMIN_PASSWORD = TEST_PASSWORD;

export const CLUSTER_EDITOR_USERNAME = "clustereditor";
export const CLUSTER_EDITOR_INITIAL_PASSWORD = TEST_INITIAL_PASSWORD;
export const CLUSTER_EDITOR_PASSWORD = TEST_PASSWORD;

export const CLUSTER_VIEWER_USERNAME = "clusterviewer";
export const CLUSTER_VIEWER_INITIAL_PASSWORD = TEST_INITIAL_PASSWORD;
export const CLUSTER_VIEWER_PASSWORD = TEST_PASSWORD;

// 用于邀请测试的临时用户
export const INVITE_TEST_USERNAME = "inviteuser";
export const INVITE_TEST_PASSWORD = TEST_PASSWORD;

// SSH 连接配置
export const TEST_SSH_ADDRESS = process.env.TEST_NODE2_IP;
export const TEST_SSH_USERNAME = "root";
export const TEST_SSH_PASSWORD = process.env.TEST_NODE2_PASSWORD;