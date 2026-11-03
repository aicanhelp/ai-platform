import { expect, test } from "@playwright/test";
import {
    installComponent,
    checkComponent,
    changePullPolicyLineWithWait,
    openYamlQuickDeploy,
    inputYamlContent,
    deleteResources,
    uninstallManagedApps,
} from "./utils/application";
import {
    totalTimeout,
    serviceClusterUrl,
    componentNameExt,
    buildDeployPlans,
    extPlan,
    type DeployPlan,
} from "./utils/constants";

function buildQuickDeployYaml(deploymentName: string, serviceName: string) {
    return `---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${deploymentName}
  namespace: default
  labels:
    app: busybox
spec:
  replicas: 2
  selector:
    matchLabels:
      app: busybox
  template:
    metadata:
      labels:
        app: busybox
    spec:
      containers:
      - name: busybox
        image: busybox:latest
        ports:
        - containerPort: 80
---
apiVersion: v1
kind: Service
metadata:
  name: ${serviceName}
  namespace: default
spec:
  selector:
    app: busybox
  ports:
  - port: 80
    targetPort: 80
  type: ClusterIP
`;
}

test.describe("应用管理-创建", () => {
    test.describe.configure({ timeout: totalTimeout });
    let createdApps: DeployPlan[] = [];

    test.afterEach(async ({ page }) => {
        await uninstallManagedApps(page, createdApps);
        createdApps = [];
    });

    test("【应用创建-应用市场HELM-003】应用市场-应用部署-参数配置-yaml修改-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        const [plan] = buildDeployPlans([extPlan(componentNameExt, "ext")]);
        await installComponent(page, plan.componentName, plan.appName, plan.version, plan.source, {
            beforeConfirm: async (p) => {
                const { oldLine, newLine } = await changePullPolicyLineWithWait(p);
                await p.getByRole("tab", { name: "变化" }).click();
                await expect(p.getByRole("cell", { name: "- " + oldLine }).first()).toBeVisible();
                await expect(p.getByRole("cell", { name: "+ " + newLine }).first()).toBeVisible();
            },
        });
        createdApps.push(plan);
        await checkComponent(page, plan.appName, { entry: plan.entry });
    });

    test("【应用创建-应用市场HELM-006】应用市场-应用部署-部署成功后跳转到应用管理-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        const [plan] = buildDeployPlans([extPlan(componentNameExt, "ext")]);
        await installComponent(page, plan.componentName, plan.appName, plan.version, plan.source);
        createdApps.push(plan);
        await checkComponent(page, plan.appName, { entry: plan.entry });
    });
});

test.describe("顶栏 YAML 一键部署", () => {
    test.describe.configure({ mode: "serial", timeout: totalTimeout });

    let yamlDeploymentName: string;
    let yamlServiceName: string;

    test.beforeEach(async ({ page }) => {
        await page.goto(serviceClusterUrl + "/container_platform/applicationManageHelm");

        const suffix = Date.now().toString().slice(-6);
        yamlDeploymentName = `busybox-deployment-${suffix}`;
        yamlServiceName = `busybox-service-${suffix}`;
        const yamlContent = buildQuickDeployYaml(yamlDeploymentName, yamlServiceName);

        await openYamlQuickDeploy(page);
        await inputYamlContent(page, yamlContent);
    });

    test("【应用创建-顶栏YAML-002】顶部右侧-yaml一键部署-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await page.waitForTimeout(1_000);
        await page.getByRole("button", { name: "部 署" }).click();
        await expect(page.getByText('Deployment创建成功')).toBeVisible({ timeout: 60_000 });
        await expect(page.getByText('Service创建成功')).toBeVisible({ timeout: 60_000 });
        await deleteResources(page, yamlDeploymentName, yamlServiceName);
    });

    test("【应用创建-顶栏YAML-003】顶部右侧-yaml一键部署-导出查找复制-取消重置-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        const downloadPromise = page.waitForEvent("download", { timeout: 60_000 });
        await page.getByText("导出").click();
        const download = await downloadPromise;
        expect(download.suggestedFilename().toLowerCase()).toContain(".yaml");
        await page.getByRole("img", { name: "copy" }).locator("svg").click();
        await expect(page.getByText("复制成功")).toBeVisible();
        await page.getByRole("button", { name: "重 置" }).click();
        await expect(page.getByText(yamlDeploymentName)).toHaveCount(0);
        await expect(page.getByText(yamlServiceName)).toHaveCount(0);
        await page.getByRole("button", { name: "取 消" }).click();
        await expect(page).toHaveURL(/\/container_platform\/appMarket\/marketCategory/);
        await expect(page.getByRole("img", { name: "appstore-add" })).toBeVisible();
    });
});
