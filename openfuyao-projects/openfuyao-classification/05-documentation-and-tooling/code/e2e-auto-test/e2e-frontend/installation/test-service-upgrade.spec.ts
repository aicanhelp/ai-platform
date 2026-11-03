import type { Page } from "@playwright/test";
import { expect, test } from "@playwright/test";
import { addonOptions, urls, nodeInfo, timeouts } from "./constants/common";
import type { NodeInfoFields } from "./utils/clusterManage";
import { login, logout } from "./utils/rbac";
import {
    verifyClusterHealthy,
    verifyClusterInstalling,
    verifyClusterUpgrading,
    verifyClusterDeleted,
    verifyClusterHealthOrInstallFailed,
} from "./utils/verify";
import { deleteCluster, goToCreateClusterPage, setClusterInfo, setNodeInfo, setAdvancedConfig, createCluster, startClusterUpgrade } from "./utils/clusterManage";
import { withSshConnection } from "./utils/upgrade-pre";
// import { updateInstallerServiceRemotePaths } from "./utils/upgrade-pre";

const { node1Info, guideNodeInfo } = nodeInfo;
const { totalTimeout } = timeouts;
const TEST_CLUSTER_NAME ='bke-cluster-svc-cluster-upgrade';
const TEMP_REMOTE_PATCH_PATH = "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/ext-components/version-config/";
const TEMP_REMOTE_DEPLOY_PATH = "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/ext-components/version-config/";
const ORIGINAL_REMOTE_PATCH_PATH = "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/";
const ORIGINAL_REMOTE_DEPLOY_PATH = "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/latest/";


function requireEnv(name: string): string {
    const value = process.env[name];
    if (!value) {
        throw new Error(`${name} is required`);
    }
    return value;
}

function shellQuote(value: string | number): string {
    return `'${String(value).replace(/'/g, `'"'"'`)}'`;
}

function yamlSingleQuote(value: string | number): string {
    return `'${String(value).replace(/'/g, "''")}'`;
}

function escapeRegExp(value: string): string {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

async function verifyClusterInstallStartedOrSettled(page: Page, clusterName: string) {
    const statusPattern = new RegExp(`${escapeRegExp(clusterName)}.*(安装中|未知|健康|安装失败)`);
    await expect(page.getByRole('row', { name: statusPattern })).toBeVisible({ timeout: timeouts.validateTimeout });
}

async function createInitialClusterBySsh() {
    const guideNodeHost = requireEnv("TEST_NODE1_IP");
    const guideNodePassword = requireEnv("TEST_NODE1_PASSWORD");
    const serviceNodeIp = requireEnv("TEST_NODE2_IP");
    const serviceNodePassword = requireEnv("TEST_NODE2_PASSWORD");
    const serviceNode2Ip = requireEnv("TEST_NODE3_IP");
    const serviceNode2Password = requireEnv("TEST_NODE3_PASSWORD");
    const guideNodeUsername = "root";

    return withSshConnection(
        {
            host: guideNodeHost,
            username: guideNodeUsername,
            password: guideNodePassword,
            port: Number(guideNodeInfo.port),
            readyTimeout: 15000,
        },
        async (ssh) => {
            const commands = [
                {
                    command: "cp /bke/cluster/1master1node-nodes.yaml /opt/1master1node-nodes.yaml",
                    errorMessage: "Failed to copy node config",
                },
                {
                    command: `CLUSTER_NAME=${shellQuote(TEST_CLUSTER_NAME)} yq -i '.metadata.namespace = strenv(CLUSTER_NAME) | .metadata.labels."cluster.x-k8s.io/cluster-name" = strenv(CLUSTER_NAME)' /opt/1master1node-nodes.yaml`,
                    errorMessage: "Failed to update node metadata",
                },
                {
                    command: `NODE_IP=${shellQuote(serviceNodeIp)} NODE_PASSWORD=${shellQuote(serviceNodePassword)} yq -i 'with(select(documentIndex == 0); .spec.ip = strenv(NODE_IP) | .spec.password = strenv(NODE_PASSWORD) | .spec.username = "root")' /opt/1master1node-nodes.yaml`,
                    errorMessage: "Failed to update first node spec",
                },
                {
                    command: `NODE_IP=${shellQuote(serviceNode2Ip)} NODE_PASSWORD=${shellQuote(serviceNode2Password)} yq -i 'with(select(documentIndex == 1); .spec.ip = strenv(NODE_IP) | .spec.password = strenv(NODE_PASSWORD) | .spec.username = "root")' /opt/1master1node-nodes.yaml`,
                    errorMessage: "Failed to update second node spec",
                },
                {
                    command: "cp /bke/cluster/1master1node-cluster.yaml /opt/1master1node-cluster.yaml",
                    errorMessage: "Failed to copy cluster config",
                },
                {
                    command: `CLUSTER_NAME=${shellQuote(TEST_CLUSTER_NAME)} yq -i '.metadata.name = strenv(CLUSTER_NAME)' /opt/1master1node-cluster.yaml`,
                    errorMessage: "Failed to update cluster name",
                },
                {
                    command: `CLUSTER_NAMESPACE=${shellQuote(TEST_CLUSTER_NAME)} yq -i '.metadata.namespace = strenv(CLUSTER_NAMESPACE)' /opt/1master1node-cluster.yaml`,
                    errorMessage: "Failed to update cluster namespace",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("etcdVersion")).etcdVersion) = "v3.6.7-of.0"' /opt/1master1node-cluster.yaml`,
                    errorMessage: "Failed to update etcdVersion",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("kubernetesVersion")).kubernetesVersion) = "v1.33.1-of.2"' /opt/1master1node-cluster.yaml`,
                    errorMessage: "Failed to update kubernetesVersion",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "coredns").version) = "v1.12.2-of.1"' /opt/1master1node-cluster.yaml`,
                    errorMessage: "Failed to update coredns version",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "kubeproxy").version) = "v1.33.1-of.2"' /opt/1master1node-cluster.yaml`,
                    errorMessage: "Failed to update kubeproxy version",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "calico").version) = "v3.31.3"' /opt/1master1node-cluster.yaml`,
                    errorMessage: "Failed to update calico version",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("containerdVersion")).containerdVersion) = "v2.1.1"' /opt/1master1node-cluster.yaml`,
                    errorMessage: "Failed to update containerdVersion",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!seq")) |= map(select((.name // "") != "cluster-api" and (.name // "") != "openfuyao-system-controller"))' /opt/1master1node-cluster.yaml`,
                    errorMessage: "Failed to remove addons from cluster config",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("openFuyaoVersion")).openFuyaoVersion) = "v26.05"' /opt/1master1node-cluster.yaml`,
                    errorMessage: "Failed to update openFuyaoVersion",
                },
                {
                    command: "timeout 10m bke cluster create -f /opt/1master1node-cluster.yaml -n /opt/1master1node-nodes.yaml",
                    errorMessage: "Failed to create cluster by bke",
                },
            ];

            for (const { command, errorMessage } of commands) {
                const result = await ssh.execCommand(command);
                if (result.code !== 0) {
                    throw new Error(`${errorMessage}: ${result.stderr || result.stdout}`);
                }
            }
        },
    );
}

async function createInitialClusterBySsh2(){
    const guideNodeHost = requireEnv("TEST_NODE1_IP");
    const guideNodePassword = requireEnv("TEST_NODE1_PASSWORD");
    const vipHost = requireEnv("WORKLOAD_CLUSTER_VIP_HOST");
    const vipPort = requireEnv("WORKLOAD_CLUSTER_VIP_PORT");
    const master1Ip = requireEnv("TEST_NODE2_IP");
    const master1Password = requireEnv("TEST_NODE2_PASSWORD");
    const master2Ip = requireEnv("TEST_NODE3_IP");
    const master2Password = requireEnv("TEST_NODE3_PASSWORD");
    const master3Ip = requireEnv("TEST_NODE4_IP");
    const master3Password = requireEnv("TEST_NODE4_PASSWORD");
    const workerIp = requireEnv("TEST_NODE5_IP");
    const workerPassword = requireEnv("TEST_NODE5_PASSWORD");
    const guideNodeUsername = "root";

    return withSshConnection(
        {
            host: guideNodeHost,
            username: guideNodeUsername,
            password: guideNodePassword,
            port: Number(guideNodeInfo.port),
            readyTimeout: 15000,
        },
        async (ssh) => {
            const appendWorkerNodeCommand = `cat >> /opt/3master-nodes.yaml <<'EOF'
---
apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  creationTimestamp: null
  labels:
    cluster.x-k8s.io/cluster-name: ${TEST_CLUSTER_NAME}
  name: bke-cluster-n1
  namespace: ${TEST_CLUSTER_NAME}
spec:
  hostname: n1
  ip: ${workerIp}
  password: ${yamlSingleQuote(workerPassword)}
  port: "22"
  role:
    - node
  username: root
status: {}
EOF`;

            const commands = [
                {
                    command: `VIP_HOST=${shellQuote(vipHost)} VIP_PORT=${shellQuote(vipPort)} yq -i '.spec.controlPlaneEndpoint.host = strenv(VIP_HOST) | .spec.controlPlaneEndpoint.port = env(VIP_PORT)' /opt/1master1node-cluster.yaml`,
                    errorMessage: "Failed to update controlPlaneEndpoint",
                },
                {
                    command: "cp /bke/cluster/3master-nodes.yaml /opt/3master-nodes.yaml",
                    errorMessage: "Failed to copy 3master node config",
                },
                {
                    command: `CLUSTER_NAME=${shellQuote(TEST_CLUSTER_NAME)} yq -i '.metadata.namespace = strenv(CLUSTER_NAME) | .metadata.labels."cluster.x-k8s.io/cluster-name" = strenv(CLUSTER_NAME)' /opt/3master-nodes.yaml`,
                    errorMessage: "Failed to update 3master node metadata",
                },
                {
                    command: `NODE_IP=${shellQuote(master1Ip)} NODE_PASSWORD=${shellQuote(master1Password)} yq -i 'with(select(documentIndex == 0); .spec.ip = strenv(NODE_IP) | .spec.password = strenv(NODE_PASSWORD) | .spec.username = "root")' /opt/3master-nodes.yaml`,
                    errorMessage: "Failed to update first master node spec",
                },
                {
                    command: `NODE_IP=${shellQuote(master2Ip)} NODE_PASSWORD=${shellQuote(master2Password)} yq -i 'with(select(documentIndex == 1); .spec.ip = strenv(NODE_IP) | .spec.password = strenv(NODE_PASSWORD) | .spec.username = "root")' /opt/3master-nodes.yaml`,
                    errorMessage: "Failed to update second master node spec",
                },
                {
                    command: `NODE_IP=${shellQuote(master3Ip)} NODE_PASSWORD=${shellQuote(master3Password)} yq -i 'with(select(documentIndex == 2); .spec.ip = strenv(NODE_IP) | .spec.password = strenv(NODE_PASSWORD) | .spec.username = "root")' /opt/3master-nodes.yaml`,
                    errorMessage: "Failed to update third master node spec",
                },
                {
                    command: appendWorkerNodeCommand,
                    errorMessage: "Failed to append worker node config",
                },
                {
                    command: "timeout 10m bke cluster create -f /opt/1master1node-cluster.yaml -n /opt/3master-nodes.yaml",
                    errorMessage: "Failed to create 3master1worker cluster by bke",
                },
            ];

            for (const { command, errorMessage } of commands) {
                const result = await ssh.execCommand(command);
                if (result.code !== 0) {
                    throw new Error(`${errorMessage}: ${result.stderr || result.stdout}`);
                }
            }
        },
    );
}

// 填写集群信息，可指定集群名称、fuyao版本等
async function fillClusterInfo(
    page: Page,
    clusterName: string,
    masterInfo: NodeInfoFields,
    fuyaoVersion?: string,
    addon?: string,
    _useBootstrap = false,
) {
    await goToCreateClusterPage(page);
    await setClusterInfo(page, clusterName, fuyaoVersion);
    await setNodeInfo(page, 'master', 0, masterInfo);
    await setAdvancedConfig(page, addon ?? addonOptions.noAddon);
};

// 将集群升级至指定版本
async function upgradeCluster(page: Page, clusterName: string, upgradeVersion: string) {
    await startClusterUpgrade(page, clusterName, upgradeVersion);
    await verifyClusterUpgrading(page, clusterName);
    await verifyClusterHealthy(page, clusterName);
};

// test.beforeAll(async () => {
//     const result = await updateInstallerServiceRemotePaths({
//         host: guideNodeInfo.ip,
//         username: guideNodeInfo.userName,
//         password: guideNodeInfo.password,
//         remotePatchPath: TEMP_REMOTE_PATCH_PATH,
//         remoteDeployPath: TEMP_REMOTE_DEPLOY_PATH,
//     });
//
//     console.log("beforeAll verifyOutput:\n", result.verifyOutput);
// });
//
// test.afterAll(async () => {
//     const result = await updateInstallerServiceRemotePaths({
//         host: guideNodeInfo.ip,
//         username: guideNodeInfo.userName,
//         password: guideNodeInfo.password,
//         remotePatchPath: ORIGINAL_REMOTE_PATCH_PATH,
//         remoteDeployPath: ORIGINAL_REMOTE_DEPLOY_PATH,
//     });
//     console.log("afterAll verifyOutput:\n", result.verifyOutput);
// });

test.beforeEach(async ({ page }) => {
    // 登录 admin 账号
    await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl)
    await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
});

test.afterEach(async ({ page }) => {
    await page.goto(process.env.TEST_GUIDEURL)
    try {
        await page.getByRole('cell', { name: TEST_CLUSTER_NAME }).waitFor({ state: 'visible' });
        await verifyClusterHealthOrInstallFailed(page, TEST_CLUSTER_NAME);
        await deleteCluster(page, TEST_CLUSTER_NAME);
        await verifyClusterDeleted(page, TEST_CLUSTER_NAME);
    } finally {
        await logout(page);
    }
});

test.describe("引导节点升级业务集群", { tag: ["@post-init","@custom-repo"] }, async () => {
    test.describe.configure({ mode: 'serial' });
    test("选择v25.12-rc.3模拟基线+补丁安装业务集群", { tag: ['@skip-temporarily'] }, async ({ page }) => {
        test.setTimeout(totalTimeout);
        await test.step("创建初始集群", async () => {
            const fuyaoVersion = "v25.12-rc.3";
            const clusterName = TEST_CLUSTER_NAME;
            // 设置 集群 信息
            await fillClusterInfo(page, clusterName, node1Info, fuyaoVersion)
            // 创建集群
            await createCluster(page);
            // 验证
            await verifyClusterInstalling(page, clusterName);
            await verifyClusterHealthy(page, clusterName);
        });
    });

    // 引导节点创建业务集群，1master1worker业务集群升级
    test("1master1worker业务集群升级", async ({ page }) => {
        test.setTimeout(totalTimeout);
        /*
        await test.step("创建初始集群", async () => {
            const fuyaoVersion = "v26.03";
            // 设置 集群 信息
            await fillClusterInfo(page, clusterName, node1Info, fuyaoVersion)
            // 创建集群
            await page.getByRole('button', { name: '一键校验' }).click();
            await createCluster(page);
            // 验证
            await verifyClusterInstalling(page, clusterName);
            await verifyClusterHealthy(page, clusterName);
            await expect(page.getByRole('cell', { name: 'v26.03' }).first()).toBeVisible();
        });
        */

        await test.step("create initial cluster by ssh", async () => {
            await createInitialClusterBySsh();
            await verifyClusterInstallStartedOrSettled(page, TEST_CLUSTER_NAME);
            await verifyClusterHealthy(page, TEST_CLUSTER_NAME);
            await expect(page.getByRole('cell', { name: 'v26.05' }).first()).toBeVisible();
        });

        await test.step("集群升级", async () => {
            const upgradeVersion = "v26.06";
            await upgradeCluster(page, TEST_CLUSTER_NAME, upgradeVersion);
            await expect(page.getByRole('cell', { name: 'v26.06' }).first()).toBeVisible();
            await expect(page.getByRole('cell', { name: '1.34.3-of.1' }).first()).toBeVisible();
        });
    });

    test("3master1worker业务集群升级", async ({ page }) => {
        test.setTimeout(totalTimeout);


        await test.step("create initial cluster by ssh", async () => {
            await createInitialClusterBySsh2();
            await verifyClusterInstallStartedOrSettled(page, TEST_CLUSTER_NAME);
            await verifyClusterHealthy(page, TEST_CLUSTER_NAME);
            await expect(page.getByRole('cell', { name: 'v26.05' }).first()).toBeVisible();
        });

        await test.step("集群升级", async () => {
            const upgradeVersion = "v26.06";
            await upgradeCluster(page, TEST_CLUSTER_NAME, upgradeVersion);
            await expect(page.getByRole('cell', { name: 'v26.06' }).first()).toBeVisible();
            await expect(page.getByRole('cell', { name: '1.34.3-of.1' }).first()).toBeVisible();
        });
    });

});
