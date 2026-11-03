import { readFile } from 'fs/promises';
import { expect, test } from "@playwright/test";
import { NodeSSH } from "node-ssh";
import { createClusterData, urls, nodeInfo, timeouts, addonOptions } from "./constants/common";
import { login, firstLoginChangePassword, logout } from "./utils/rbac";
import {
    verifyClusterHealthy,
    verifyClusterInstalling,
    verifyClusterDeleted,
    verifyClusterHealthOrInstallFailed,
} from "./utils/verify";
import {
    installComponent,
    checkResourceNum
} from "./utils/market";
import {
    deleteCluster,
    setClusterInfo,
    addNode,
    startClusterUpgrade,
    setNodeInfo,
    goToCreateClusterPage,
    setAdvancedConfig,
    setAdvancedOfflineConfig
} from "./utils/clusterManage.js";

const { node1Info, node2Info, guideNodeInfo } = nodeInfo;
const { totalTimeout, validateTimeout } = timeouts;
const deploymentActiveTime = 20 * 1000;
const TEST_CLUSTER_NAME = 'test-cluster';
// 引导节点信息：从环境变量中获取 IP
const bootstrapIp = guideNodeInfo.ip;
const kubeconfig = process.env.TEST_KUBECONFIG || "/etc/rancher/k3s/k3s.yaml";

let originalBkeConfig = '';

function shellQuote(value: string | number) {
    return `'${String(value).replace(/'/g, `'"'"'`)}'`;
}

function buildKubectlCmd(kubeconfig: string) {
    return `kubectl --kubeconfig=${shellQuote(kubeconfig)}`;
}

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

// 1) 镜像列表中应包含 deploy.bocloud.k8s
// 2) 关键镜像：pause 与 coredns 来自 deploy.bocloud.k8s
async function verifyOfflineLocalRepoImages() {
    const images = await queryCrictlImagesOnNode1(); 
    if (!images.length) return false;
    const joined = images.join('\n');
    const hasDeploy = joined.includes('deploy.bocloud.k8s');

    let pauseOK = false;
    let corednsOK = false;
    for (const line of images) {
        if (line.includes('pause') && line.includes('deploy.bocloud.k8s')) {
            pauseOK = true;
        }
        if (line.includes('coredns') && line.includes('deploy.bocloud.k8s')) {
            corednsOK = true;
        }
        if (pauseOK && corednsOK) break;
    }
    return hasDeploy && pauseOK && corednsOK;
}

async function connectSSH() {
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

async function saveBkeConfig() {
    const ssh = await connectSSH();
    try {
        const kubectl = buildKubectlCmd(kubeconfig);
        const result = await ssh.execCommand(`${kubectl} get cm -n cluster-system bke-config -oyaml`);
        if (result.code !== 0) {
            throw new Error(`Failed to get bke-config: ${result.stderr || result.stdout}`);
        }
        originalBkeConfig = result.stdout;
        console.log('bke-config saved successfully');
    } finally {
        ssh.dispose();
    }
}

async function restoreBkeConfig() {
    if (!originalBkeConfig) {
        console.log('No original bke-config to restore');
        return;
    }
    const ssh = await connectSSH();
    try {
        const kubectl = buildKubectlCmd(kubeconfig);
        
        const deleteResult = await ssh.execCommand(`${kubectl} delete cm -n cluster-system bke-config --ignore-not-found=true`);
        if (deleteResult.code !== 0) {
            throw new Error(`Failed to delete bke-config: ${deleteResult.stderr || deleteResult.stdout}`);
        }
        
        const createResult = await ssh.execCommand(`echo '${originalBkeConfig.replace(/'/g, "'\\''")}' | ${kubectl} create -f -`);
        if (createResult.code !== 0) {
            throw new Error(`Failed to create bke-config: ${createResult.stderr || createResult.stdout}`);
        }
        console.log('bke-config restored successfully');
    } finally {
        ssh.dispose();
    }
}

// 读取 YAML 文件为字符串
async function loadYamlAsString(filePath: string) {
    try {
        const fileContents = await readFile(filePath, 'utf8');
        return fileContents;
    } catch (error) {
        console.error(`Error reading YAML file: ${error}`);
        throw error;
    }
}

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

test.describe('离线引导节点', { tag: ['@post-init', '@offline'] }, () => {
    test.describe.configure({ mode: 'serial' });

    test.beforeAll(async () => {
        await saveBkeConfig();
    });

    test.afterAll(async () => {
        await restoreBkeConfig();
    });

    test('上传版本yaml并创建集群并升级', {tag: ['@offline']}, async ({ page }) => {
        test.setTimeout(totalTimeout);
        const clusterName = TEST_CLUSTER_NAME;
        const version = "v25.10"
        const upgradeVersion = "v25.10.999"
        await test.step("上传安装版本yaml", async () => {
            const versionYaml = await loadYamlAsString('./installation/versionYaml/v25_10.yaml');
            await page.getByRole('button', { name: '补丁上传' }).click();
            await page.getByRole('textbox').locator('div').fill(versionYaml);
            await page.getByRole('button', { name: /确\s*定/ }).click();
            await expect(page.getByText('上传成功')).toBeVisible();
        });
        await test.step("创建初始集群", async () => {
            await goToCreateClusterPage(page);
            await setClusterInfo(page, clusterName, version);
            await setNodeInfo(page, 'master', 0, node1Info);
            // 设置高级配置并校验
            await test.step('设置高级配置', async () => {
                await setAdvancedConfig(page, addonOptions.noAddon);
                await page.getByRole('button', { name: '一键校验' }).click();
                await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
            });
            // 创建集群
            await page.getByRole('button', { name: /创\s*建/ }).click();
            await verifyClusterInstalling(page, clusterName);
            await verifyClusterHealthOrInstallFailed(page, clusterName);
        });
        await test.step("上传升级版本yaml", async () => {
            const versionYaml = await loadYamlAsString('./installation/versionYaml/v25_10_999.yaml');
            await page.getByRole('button', { name: '补丁上传' }).click();
            await page.getByRole('textbox').locator('div').fill(versionYaml);
            await page.getByRole('button', { name: /确\s*定/ }).click();
            await expect(page.getByText('上传成功')).toBeVisible();
        });
        await test.step("集群升级", async () => {
            await startClusterUpgrade(page, clusterName, upgradeVersion);
            // 等待升级完成，集群恢复健康状态
            await verifyClusterHealthOrInstallFailed(page, clusterName);
        });
    });

    test('创建集群并检查应用市场', {tag: ['@offline']}, async ({ page }) => {
        test.setTimeout(totalTimeout);
        const clusterName = TEST_CLUSTER_NAME;
        const fuyaoUrl = "https://" + node2Info.ip + ":" + createClusterData.fuyaoPort;
        await test.step('创建业务集群', async () => {
            // 设置 1MASTER1NODE 信息
            await test.step('配置集群信息', async () => {
                await goToCreateClusterPage(page);
                await setClusterInfo(page, clusterName, "latest")
                await setNodeInfo(page, 'master', 0, node1Info);
                await addNode(page, 'node');
                await setNodeInfo(page, 'node', 0, node2Info);
            });
            // 设置高级配置并校验
            await test.step('设置高级配置', async () => {
                await page.getByRole('combobox', { name: '镜像仓库地址' }).click();
                await page.getByTitle(createClusterData.offlineImageDomain).click();
                await page.getByRole('combobox', { name: '镜像仓库IP地址' }).click();
                await page.getByTitle(bootstrapIp).click();
                await page.getByRole('combobox', { name: 'Chart仓库地址' }).click();
                const chartAdd = "http://" + bootstrapIp + ":" + createClusterData.chartPort;
                await page.getByTitle(chartAdd).click();
                await page.getByRole('button', { name: '一键校验' }).click();
                await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
            });
            // 创建集群
            await page.getByRole('button', { name: /创\s*建/ }).click();
            await verifyClusterInstalling(page, clusterName);
            await verifyClusterHealthy(page, clusterName);
        });
        await test.step('登录业务集群并进入应用市场', async () => {
            // 初次登录业务集群修改密码
            await page.goto(fuyaoUrl + urls.loginUrl);
            await test.step("验证使用正确的用户名和密码登陆成功", async () => {
                await page.getByRole('textbox', { name: '*用户名' }).fill(process.env.TEST_USERNAME);
                await page.getByRole('textbox', { name: '*密码' }).fill(process.env.TEST_INITIAL_PASSWORD);
                await page.getByRole('button', { name: '登录' }).click();
            });
            await test.step("验证首次登录必须修改密码", async () => {
                await expect(page.getByRole('heading', { name: '初次修改密码' })).toBeVisible();
            });
            await test.step("初次登录修改密码成功", async () => {
                await firstLoginChangePassword(page);
                await expect(page.getByText(process.env.TEST_USERNAME)).toBeVisible();
            });
            await page.getByText('应用市场').click();
        });
        const componentName = "volcano-config-service";
        const appName = "test-volcano";
        await test.step('部署应用', async () => {
            await installComponent(page, componentName, appName);
            await expect(page.getByText('部署成功，即将跳转扩展组件管理页面')).toBeVisible();
            await page.getByRole('textbox', { name: '搜索Extend名称' }).fill(appName);
            await page.getByRole('button', { name: 'search' }).click();
            await expect(page.getByRole('cell', { name: 'check-circle 部署成功' })).toBeVisible();
            await page.waitForTimeout(deploymentActiveTime);    // 等待deployment进入active状态
        });
        // 针对volcano-config-service，检查deployment和deamonSet处于active状态的个数
        await test.step('检查资源是否成功部署', async () => {
            const namespace = "volcano-system";
            const deploymentNum = 5;
            const deamonSetNum = 3;
            await checkResourceNum(page, namespace, "Deployment", deploymentNum);
            await page.getByRole('link', { name: "DaemonSet" }).click();
            await expect(page.getByRole('row', { name: 'Active' }).getByRole('paragraph')).toHaveCount(deamonSetNum);
        });
        await test.step('卸载应用', async () => {
            await page.getByRole('link', { name: '扩展组件管理' }).click();
            await page.getByRole('textbox', { name: '搜索Extend名称' }).fill(appName);
            await page.getByRole('img', { name: 'more' }).locator('svg').click();
            await page.getByRole('button', { name: '卸载' }).click();
            await expect(page.getByText('卸载扩展组件', { exact: true })).toBeVisible();
            await page.getByRole('checkbox', { name: '已了解卸载操作无法恢复' }).check();
            await page.getByRole('button', { name: '卸 载' }).click();
            await expect(page.getByText('扩展组件' + appName + '卸载成功')).toBeVisible();
        });
    });

    /*
     用例名称：安装业务集群
     用例步骤：1）下发集群安装，预期结果1
     预期结果：1）集群成功拉起，集群使用的镜像均来源deploy.bocloud.k8s
    */
    test("test 1Master cluster without addon", async ({ page }) => {
        const clusterName = TEST_CLUSTER_NAME;
        await test.step('创建业务集群', async () => {
            // 设置 1MASTER 信息
            await test.step('配置集群信息', async () => {
                await goToCreateClusterPage(page);
                await setClusterInfo(page, clusterName, "latest");
                await setNodeInfo(page, 'master', 0, node1Info);
            });
            // 设置高级配置并校验
            await test.step('设置高级配置', async () => {
                await setAdvancedOfflineConfig(page, addonOptions.noAddon);
                await page.getByRole('button', { name: '一键校验' }).click();
                await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
            });
            // 创建集群
            await page.getByRole('button', { name: /创\s*建/ }).click();
            await verifyClusterInstalling(page, clusterName);
            await verifyClusterHealthy(page, clusterName);
        });
        const ok = await verifyOfflineLocalRepoImages();
        expect(ok).toBeTruthy();
    });
});
