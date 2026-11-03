export const urls = {
    loginUrl: '/oauth2/auth/login/fuyaoPasswordProvider',
    rbacManagementUrl: '/user_manage/user',
};

export const createClusterData = {
    imageRepoDomain: 'cr.openfuyao.cn',
    imageRepoIp: '119.3.216.9',
    offlineImageDomain: 'deploy.bocloud.k8s',
    chartDomain: 'https://helm.openfuyao.cn/',
    chartPort: 38080,
    fuyaoPort: 31616,
};

export const addonOptions = {
    noAddon: '无',
    openfuyaoCores: 'openFuyao Cores',
};

const PORT = '22';
const ROOTNAME = 'root';

/** 测试依赖环境变量；未设置时运行期可能异常 */
export interface NodeInfoEntry {
    name: string;
    ip: string;
    port: string;
    userName: string;
    password: string;
}

export const nodeInfo: {
    guideNodeInfo: NodeInfoEntry;
    node1Info: NodeInfoEntry;
    node2Info: NodeInfoEntry;
    node3Info: NodeInfoEntry;
    node4Info: NodeInfoEntry;
} = {
    guideNodeInfo: {
        name: 'guide-node',
        ip: process.env.TEST_NODE1_IP!,
        port: PORT,
        userName: ROOTNAME,
        password: process.env.TEST_NODE1_PASSWORD!,
    },
    node1Info: {
        name: 'node-test-1',
        ip: process.env.TEST_NODE2_IP!,
        port: PORT,
        userName: ROOTNAME,
        password: process.env.TEST_NODE2_PASSWORD!,
    },
    node2Info: {
        name: 'node-test-2',
        ip: process.env.TEST_NODE3_IP!,
        port: PORT,
        userName: ROOTNAME,
        password: process.env.TEST_NODE3_PASSWORD!,
    },
    node3Info: {
        name: 'node-test-3',
        ip: process.env.TEST_NODE4_IP!,
        port: PORT,
        userName: ROOTNAME,
        password: process.env.TEST_NODE4_PASSWORD!,
    },
    node4Info: {
        name: 'node-test-4',
        ip: process.env.TEST_NODE5_IP!,
        port: PORT,
        userName: ROOTNAME,
        password: process.env.TEST_NODE5_PASSWORD!,
    },
};

export const timeouts = {
    installTimeout: 30 * 60 * 1000,
    uninstallTimeout: 15 * 60 * 1000,
    totalTimeout: 35 * 60 * 1000,
    validateTimeout: 60 * 1000,
    upgradeVisibleTimeout: 5 * 60 * 1000,
};
