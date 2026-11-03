import { expect, test } from "@playwright/test";
import { addonOptions, urls, nodeInfo, timeouts } from "./constants/common";
import { login, logout } from "./utils/rbac";
import {
    verifyClusterHealthy,
    verifyClusterInstalling,
    verifyClusterDeleted,
    checkFieldsAndValidateErrorMessages,
    verifyClusterHealthOrInstallFailed,
} from "./utils/verify";
import {
    deleteCluster,
    setClusterInfo,
    setNodeInfo,
    addNode,
    setAdvancedConfig,
    createCluster,
    validateCluster,
    goToCreateClusterPage,
    setBalanceIp,
} from "./utils/clusterManage";
import { newContext } from "@/utils/common";
const { node1Info, node2Info, node3Info, node4Info, guideNodeInfo } = nodeInfo;
const { totalTimeout, validateTimeout } = timeouts;
const TEST_CLUSTER_NAME = 'test-cluster';
const DUMMY_CLUSTER_NAME = 'dummy-cluster';
import { NodeSSH } from "node-ssh";
const kubeconfig = process.env.TEST_KUBECONFIG || "/etc/rancher/k3s/k3s.yaml";

async function connectNode1SSH() {
    const ssh = new NodeSSH();
    await ssh.connect({
        host: node1Info.ip,
        username: node1Info.userName,
        password: node1Info.password,
        port: 22,
        readyTimeout: 15000,
    });
    return ssh;
}

async function queryCrictlImagesOnNode1() {
    const ssh = await connectNode1SSH();
    try {
        const res = await ssh.execCommand('crictl images 2>&1');
        if (res.code !== 0) {
            throw new Error(`crictl images failed: ${res.stderr || res.stdout}`);
        }
        const lines = res.stdout.split('\n').map(s => s.trim()).filter(Boolean);
        if (lines.length <= 1) return [];
        const images = [];
        for (let i = 1; i < lines.length; i++) {
            const cols = lines[i].split(/\s+/);
            if (cols.length >= 2) {
                images.push(`${cols[0]}:${cols[1]}`);
            }
        }
        return images;
    } finally {
        ssh.dispose();
    }
}

// 1) 镜像列表中应同时出现 cr.openfuyao.cn 和 hub.oepkgs.net
// 2) 关键镜像：pause 来自 hub.oepkgs.net，coredns 来自 cr.openfuyao.cn
async function verifyMultiImageRepo() {
    const images = await queryCrictlImagesOnNode1();
    if (!images.length) return false;

    const joined = images.join('\n');
    const hasCr = joined.includes('cr.openfuyao.cn');
    const hasHub = joined.includes('hub.oepkgs.net');

    let pauseOK = false;
    let corednsOK = false;
    for (const line of images) {
        if (line.includes('pause') && line.includes('hub.oepkgs.net')) {
            pauseOK = true;
        }
        if (line.includes('coredns') && line.includes('cr.openfuyao.cn')) {
            corednsOK = true;
        }
        if (pauseOK && corednsOK) break;
    }
    return hasCr && hasHub && pauseOK && corednsOK;
}

function shellQuote(value: string | number) {
    return `'${String(value).replace(/'/g, `'"'"'`)}'`;
}

function buildKubectlCmd(cfg: string) {
    return `kubectl --kubeconfig=${shellQuote(cfg)}`;
}

async function connectGuideSSH() {
    const ssh = new NodeSSH();
    await ssh.connect({
        host: guideNodeInfo.ip,
        username: guideNodeInfo.userName,
        password: guideNodeInfo.password,
        port: 22,
        readyTimeout: 15000,
    });
    return ssh;
}

let originalBkeConfig = '';

async function saveBkeConfigOnGuide() {
    const ssh = await connectGuideSSH();
    try {
        const kubectl = buildKubectlCmd(kubeconfig);
        const res = await ssh.execCommand(`${kubectl} get cm -n cluster-system bke-config -oyaml`);
        if (res.code !== 0) throw new Error(`get bke-config failed: ${res.stderr || res.stdout}`);
        originalBkeConfig = res.stdout;
    } finally {
        ssh.dispose();
    }
}

async function restoreBkeConfigOnGuide() {
    if (!originalBkeConfig) return;
    const ssh = await connectGuideSSH();
    try {
        const kubectl = buildKubectlCmd(kubeconfig);
        const delRes = await ssh.execCommand(`${kubectl} delete cm -n cluster-system bke-config --ignore-not-found=true`);
        if (delRes.code !== 0) throw new Error(`delete bke-config failed: ${delRes.stderr || delRes.stdout}`);
        const createRes = await ssh.execCommand(`echo ${shellQuote(originalBkeConfig)} | ${kubectl} create -f -`);
        if (createRes.code !== 0) throw new Error(`create bke-config failed: ${createRes.stderr || createRes.stdout}`);
    } finally {
        ssh.dispose();
    }
}

async function setBkeAgentHealthPortOnGuide(newPort: string) {
    const ssh = await connectGuideSSH();
    try {
        const kubectl = buildKubectlCmd(kubeconfig);
        // 直接用 sed 将 data 中的 "agentHealthPort: \"58080\"" 改为新端口（兼容字符串/数值）
        const getCmd = `${kubectl} get cm -n cluster-system bke-config -oyaml`;
        const getRes = await ssh.execCommand(getCmd);
        if (getRes.code !== 0) throw new Error(`get bke-config failed: ${getRes.stderr || getRes.stdout}`);
        const patched = getRes.stdout
            .replace(/agentHealthPort:\s*"?58080"?/g, `agentHealthPort: "${newPort}"`);
        const applyRes = await ssh.execCommand(`echo ${shellQuote(patched)} | ${kubectl} apply -f -`);
        if (applyRes.code !== 0) throw new Error(`apply bke-config failed: ${applyRes.stderr || applyRes.stdout}`);
    } finally {
        ssh.dispose();
    }
}

async function checkBkeagentHealthOnNode1(port: string) {
    const ssh = await connectNode1SSH();
    try {
        const res = await ssh.execCommand(`curl -s http://127.0.0.1:${port}/healthz`);
        if (res.code !== 0) {
            return false;
        }
        return (res.stdout || '').trim() === 'ok';
    } finally {
        ssh.dispose();
    }
}

test.describe('集群创建', { tag: ['@post-init', '@online'] }, () => {

    test.describe('创建集群操作', async () => {
        test.setTimeout(totalTimeout);

        test.beforeEach(async ({ page }) => {
            await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl)
            await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
            await goToCreateClusterPage(page);
        });

        test.afterEach(async ({ page }) => {
            try {
                await page.getByRole('cell', { name: TEST_CLUSTER_NAME }).waitFor({ state: 'visible' });
                await verifyClusterHealthOrInstallFailed(page, TEST_CLUSTER_NAME);
                await deleteCluster(page, TEST_CLUSTER_NAME);
                await verifyClusterDeleted(page, TEST_CLUSTER_NAME);
            } finally {
                await logout(page);
            }
        });

        test("test 1Master cluster with addon", {tag: ['@skip-temporarily']}, async ({ page }) => {
            await setClusterInfo(page, TEST_CLUSTER_NAME);
            await setNodeInfo(page, 'master', 0, node1Info);
            await setAdvancedConfig(page, addonOptions.openfuyaoCores);
            await validateCluster(page);
            await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: '创 建' })).toBeEnabled();
            await createCluster(page);
            await verifyClusterInstalling(page, TEST_CLUSTER_NAME);
            await verifyClusterHealthy(page, TEST_CLUSTER_NAME);
        });

        /*
         用例名称：安装业务集群（默认端口）
         用例步骤：1）下发集群安装，预期结果1和2
         预期结果：1）集群成功拉起，集群使用的镜像来源cr.openfuyao.cn和hub.oepkgs.net
                 2）bkeagent监听58080端口
        */
        test("test 1Master cluster without addon", async ({ page }) => {
            await setClusterInfo(page, TEST_CLUSTER_NAME);
            await setNodeInfo(page, 'master', 0, node1Info);
            await setAdvancedConfig(page, addonOptions.noAddon);
            await validateCluster(page);
            await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: '创 建' })).toBeEnabled();
            await createCluster(page);
            await verifyClusterInstalling(page, TEST_CLUSTER_NAME);
            await verifyClusterHealthy(page, TEST_CLUSTER_NAME);
            const ok = await verifyMultiImageRepo();
            expect(ok).toBeTruthy();
            const agentHealthy = await checkBkeagentHealthOnNode1("58080");
            expect(agentHealthy).toBeTruthy();
        });

        /*
         用例名称：1）安装业务集群（自定义端口）
         用例步骤：1）修改引导集群的cluster-system/bke-config的agentHealthPort为58090
                  2）下发集群安装，预期结果1
         预期结果：1）集群成功拉起，bkeagent监听58090端口
        */
        test("test bkeagent custom health port 58090", { tag: ['@skip-temporarily'] }, async ({ page }) => {
            // 1) 修改 bke-config 中的 agentHealthPort 为 58090
            await saveBkeConfigOnGuide();
            try {
                await setBkeAgentHealthPortOnGuide("58090");
                // 2) 拉起业务集群
                await setClusterInfo(page, TEST_CLUSTER_NAME);
                await setNodeInfo(page, 'master', 0, node1Info);
                await setAdvancedConfig(page, addonOptions.noAddon);
                await validateCluster(page);
                await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
                await expect(page.getByRole('button', { name: '创 建' })).toBeEnabled();
                await createCluster(page);
                await verifyClusterInstalling(page, TEST_CLUSTER_NAME);
                await verifyClusterHealthy(page, TEST_CLUSTER_NAME);
                // 3) 检测 58090 健康监听
                const ok = await checkBkeagentHealthOnNode1("58090");
                expect(ok).toBeTruthy();
            } finally {
                // 4) 恢复 bke-config
                await restoreBkeConfigOnGuide();
            }
        });

        test('test create 1Master1Node cluster', async ({ page }) => {
            // 设置 1MASTER1NODE 信息
            await setClusterInfo(page, TEST_CLUSTER_NAME);
            await setNodeInfo(page, 'master', 0, node1Info);
            await addNode(page, 'node');
            await setNodeInfo(page, 'node', 0, node2Info);
            await setAdvancedConfig(page, addonOptions.noAddon);
            await validateCluster(page);
            await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: '创 建' })).toBeEnabled();
            await createCluster(page);
            await verifyClusterInstalling(page, TEST_CLUSTER_NAME);
            await verifyClusterHealthy(page, TEST_CLUSTER_NAME);
        });

        test('test create 3Master cluster', async ({ page }) => {
            await setClusterInfo(page, TEST_CLUSTER_NAME);
            await setNodeInfo(page, 'master', 0, node1Info);
            await addNode(page, 'master');
            await setNodeInfo(page, 'master', 1, node2Info);
            await addNode(page, 'master');
            await setNodeInfo(page, 'master', 2, node3Info);
            await setAdvancedConfig(page, addonOptions.noAddon);
            await validateCluster(page);
            await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: '创 建' })).toBeEnabled();
            await createCluster(page);
            await verifyClusterInstalling(page, TEST_CLUSTER_NAME);
            await verifyClusterHealthy(page, TEST_CLUSTER_NAME);
        });
    });

    test.describe('集群创建校验', () => {

        test.beforeAll(async ({ browser }) => {
            // 额外创建集群，用于创建集群时校验中与已有集群名称、节点IP、节点名称重复等情况
            const context = await newContext(browser);
            const page = await context.newPage();
            await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl)
            await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);

            await test.step('创建 Dummy 集群"', async () => {
                // 使用 node4，和正式测试逻辑中的 node1、node2、node3 错开
                await goToCreateClusterPage(page);
                await setClusterInfo(page, DUMMY_CLUSTER_NAME);
                await setNodeInfo(page, 'master', 0, node4Info);
                await setAdvancedConfig(page, addonOptions.noAddon);

                await validateCluster(page);
                await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
                await expect(page.getByRole('button', { name: '创 建' })).toBeEnabled();
                await createCluster(page);
                await verifyClusterInstalling(page, DUMMY_CLUSTER_NAME);
                await verifyClusterHealthy(page, DUMMY_CLUSTER_NAME);
            });

            await logout(page)
            await context.close()
        });

        test.afterAll(async ({ browser }) => {
            const context = await newContext(browser);
            const page = await context.newPage();

            await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl)
            await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
            try {
                await page.getByRole('cell', { name: DUMMY_CLUSTER_NAME }).waitFor({ state: 'visible' });
                await verifyClusterHealthOrInstallFailed(page, DUMMY_CLUSTER_NAME);
                await deleteCluster(page, DUMMY_CLUSTER_NAME);
                await verifyClusterDeleted(page, DUMMY_CLUSTER_NAME);
            } finally {
                await logout(page);
                await context.close();
            }
        });

        test.beforeEach(async ({ page }) => {
            await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl)
            await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
            await goToCreateClusterPage(page);
        });

        test('新增集群与已有集群名称相同时', async ({ page }) => {
            const clusterName = DUMMY_CLUSTER_NAME

            await test.step('填写集群信息，集群名称与管理节点其他集群名称重名', async () => {
                await setClusterInfo(page, clusterName);
                await setNodeInfo(page, 'master', 0, node1Info);
                await setAdvancedConfig(page, addonOptions.noAddon);
            });

            await test.step('点击校验，验证集群名称已存在', async () => {
                await validateCluster(page);
                await expect(page.getByText('集群名称已存在')).toBeVisible();
            });
        });

        test('新增集群单个虚假IP', async ({ page }) => {
            const clusterName = TEST_CLUSTER_NAME

            await test.step('填写集群信息，使用虚假IP', async () => {
                const invalidNodeInfo = { ...node1Info, ip: '192.0.2.1' };
                await setClusterInfo(page, clusterName);
                await setNodeInfo(page, 'master', 0, invalidNodeInfo);
                await setAdvancedConfig(page, addonOptions.noAddon);
            });

            await test.step('点击校验，验证虚假IP时校验失败', async () => {
                await validateCluster(page);
                await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
                await expect(page.getByRole("button", { name: '创 建' })).toBeDisabled();
            });
        });

        test('新增集群多个虚假IP', async ({ page }) => {
            const clusterName = TEST_CLUSTER_NAME

            await test.step('填写集群信息，使用虚假IP', async () => {
                const invalidNode1Info = { ...node1Info, ip: '192.0.2.1' };
                const invalidNode2Info = { ...node2Info, ip: '198.51.100.2' };
                await setClusterInfo(page, clusterName);
                await setNodeInfo(page, 'master', 0, invalidNode1Info);
                await addNode(page, 'node');
                await setNodeInfo(page, 'node', 0, invalidNode2Info);
                await setAdvancedConfig(page, addonOptions.noAddon);
            });

            await test.step('点击校验，验证虚假IP时校验失败', async () => {
                await validateCluster(page);
                await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
                await expect(page.getByRole("button", { name: '创 建' })).toBeDisabled();
            });
        });

        test('新增集群节点SSH登录失败（端口错误）', async ({ page }) => {
            const clusterName = TEST_CLUSTER_NAME

            await test.step('填写集群信息，使用虚假端口', async () => {
                const invalidNode1Info = { ...node1Info, port: '33' };

                await setClusterInfo(page, clusterName);
                await setNodeInfo(page, 'master', 0, invalidNode1Info);
                await setAdvancedConfig(page, addonOptions.noAddon);
            });

            await test.step('点击校验，验证虚假端口时校验失败', async () => {
                await validateCluster(page);
                await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
                await expect(page.getByRole("button", { name: '创 建' })).toBeDisabled();
            });
        });

        test('新增集群节点SSH登录失败（账号错误）', async ({ page }) => {
            const clusterName = TEST_CLUSTER_NAME

            await test.step('填写集群信息，使用虚假账号', async () => {
                const invalidNode1Info = { ...node1Info, userName: 'admin1' };

                await setClusterInfo(page, clusterName);
                await setNodeInfo(page, 'master', 0, invalidNode1Info);
                await setAdvancedConfig(page, addonOptions.noAddon);
            });

            await test.step('点击校验，验证虚假账号时校验失败', async () => {
                await validateCluster(page);
                await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
                await expect(page.getByRole("button", { name: '创 建' })).toBeDisabled();
            });
        });

        test('新增集群节点SSH登录失败（密码错误）', async ({ page }) => {
            const clusterName = TEST_CLUSTER_NAME

            await test.step('填写集群信息，使用虚假密码', async () => {
                const invalidNode1Info = { ...node1Info, password: 'xxxxxx' };

                await setClusterInfo(page, clusterName);
                await setNodeInfo(page, 'master', 0, invalidNode1Info);
                await setAdvancedConfig(page, addonOptions.noAddon);
            });

            await test.step('点击校验，验证虚假密码时校验失败', async () => {
                await validateCluster(page);
                await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
                await expect(page.getByRole("button", { name: '创 建' })).toBeDisabled();
            });
        });

        test('新增集群管理集群下其他节点已使用该IP（单个）', async ({ page }) => {
            const clusterName = TEST_CLUSTER_NAME

            await test.step('填写集群信息，使用其他节点已使用的IP', async () => {
                const invalidNodeInfo = { ...node1Info, ip: node4Info.ip };    // node4已被临时集群使用

                await setClusterInfo(page, clusterName);
                await setNodeInfo(page, 'master', 0, invalidNodeInfo);
                await setAdvancedConfig(page, addonOptions.noAddon);
            });

            await test.step('点击校验，验证管理集群下其他节点已使用该IP时校验失败', async () => {
                await validateCluster(page);
                await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
                await expect(page.getByRole("button", { name: '创 建' })).toBeDisabled();
            });
        });

        test('新增集群管理集群下其他节点已使用该IP（多个）', async ({ page }) => {
            const clusterName = TEST_CLUSTER_NAME

            await test.step('填写集群信息，使用其他节点已使用的IP', async () => {
                const invalidNode1Info = { ...node1Info, ip: node4Info.ip };    // node4已被临时集群使用
                const invalidNode2Info = { ...node2Info, ip: node4Info.ip };

                await setClusterInfo(page, clusterName);
                await setNodeInfo(page, 'master', 0, invalidNode1Info);
                await addNode(page, 'node');
                await setNodeInfo(page, 'node', 0, invalidNode2Info);
                await setAdvancedConfig(page, addonOptions.noAddon);
            });

            await test.step('点击校验，验证管理集群`下其他节点已使用该IPs时校验失败', async () => {
                await validateCluster(page);
                const msg = `节点IP "${node4Info.ip}" 重复`
                await expect(page.getByText(msg)).toBeVisible({ timeout: validateTimeout });
                await expect(page.getByRole("button", { name: '创 建' })).toBeDisabled();
            });
        }); 

        test('新增集群新增节点之间存在相同IP时', async ({ page }) => {
            const clusterName = TEST_CLUSTER_NAME

            await test.step('填写集群信息，使用相同IP', async () => {
                const invalidNode1Info = node1Info;
                const invalidNode2Info = { ...node2Info, ip: node1Info.ip };

                await setClusterInfo(page, clusterName);
                await setNodeInfo(page, 'master', 0, invalidNode1Info);
                await addNode(page, 'node');
                await setNodeInfo(page, 'node', 0, invalidNode2Info);
                await setAdvancedConfig(page, addonOptions.noAddon);
            });

            await test.step('点击校验，验证新增节点之间存在相同IP时校验失败', async () => {
                await validateCluster(page);
                const msg = `节点IP "${node1Info.ip}" 重复`
                await expect(page.getByText(msg)).toBeVisible({ timeout: validateTimeout });
                await expect(page.getByRole("button", { name: '创 建' })).toBeDisabled();
            });
        });

        test('新增HA集群HA IP校验', async ({ page }) => {
            const clusterName = TEST_CLUSTER_NAME

            await test.step('填写集群信息，使用其他节点未使用的IP', async () => {
                await setClusterInfo(page, clusterName);
                await setNodeInfo(page, 'master', 0, node1Info);
                await addNode(page, 'master');
                await setNodeInfo(page, 'master', 1, node2Info);
                await addNode(page, 'master');
                await setNodeInfo(page, 'master', 2, node3Info);
                await setAdvancedConfig(page, addonOptions.noAddon);
            });

            await test.step('点击校验，验证负载均衡IP为空时校验通过', async () => {
                await page.getByRole('button', { name: '一键校验' }).click();
                await expect(page.getByText('一键校验成功')).toBeVisible();
                await expect(page.getByRole("button", { name: '创 建' })).toBeEnabled();
            });

            await test.step('点击校验，验证负载均衡IP ping不通时校验通过', async () => {
                await setBalanceIp(page, '192.0.2.1');
                await validateCluster(page);
                await expect(page.getByText('一键校验成功')).toBeVisible();
                await expect(page.getByRole("button", { name: '创 建' })).toBeEnabled();
            });

            await test.step('点击校验，验证负载均衡IP ping通时校验失败', async () => {
                await setBalanceIp(page, node1Info.ip);
                await validateCluster(page);
                await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
                await expect(page.getByRole("button", { name: '创 建' })).toBeDisabled();
            });
        });

        test('新增集群新增节点间名称重复', async ({ page }) => {
            const clusterName = TEST_CLUSTER_NAME

            await test.step('填写集群信息，新增节点间名称重复', async () => {
                const invalidNode1Info = node1Info;
                const invalidNode2Info = { ...node2Info, name: node1Info.name };

                await setClusterInfo(page, clusterName);
                await setNodeInfo(page, 'master', 0, invalidNode1Info);
                await addNode(page, 'node');
                await setNodeInfo(page, 'node', 0, invalidNode2Info);
                await setAdvancedConfig(page, addonOptions.noAddon);
            });

            await test.step('点击校验，验证新增节点间名称重复时校验失败', async () => {
                await validateCluster(page);
                const msg = `节点名称 "${node1Info.name}" 重复` 
                await expect(page.getByText(msg)).toBeVisible({ timeout: validateTimeout });
                await expect(page.getByRole("button", { name: '创 建' })).toBeDisabled();
            });
        });

        test('新增集群节点信息填写不完整时', async ({ page }) => {
            const clusterName = TEST_CLUSTER_NAME

            await test.step('填写集群信息，节点信息填写不完整', async () => {
                await setClusterInfo(page, clusterName);
                await setNodeInfo(page, 'master', 0, {});
                await addNode(page, 'node');
                await setNodeInfo(page, 'node', 0, {});
                await setAdvancedConfig(page, addonOptions.noAddon);
            });

            await test.step('点击校验，显示哪些信息未填写', async () => {
                await page.getByRole('button', { name: '一键校验' }).click();
                const masterRows = page.locator('form div').filter({ hasText: 'Master信息节点名称IP地址端口号用户名密码操作添加新节点' })
                const nodeRows = page.locator('form div').filter({ hasText: 'Node信息节点名称IP地址端口号用户名密码操作添加新节点' })
                await checkFieldsAndValidateErrorMessages(masterRows);
                await checkFieldsAndValidateErrorMessages(nodeRows);
            });
        });
    });

});
