import { randomBytes } from "node:crypto";

export const totalTimeout = 10 * 60 * 1000;
export const upgradeTime = 2 * 60 * 1000;
export const revertTime = 1 * 60 * 1000;

export const serviceClusterUrl = process.env.TEST_FUYAOURL ?? "";
export const TEST_SSH_ADDRESS = process.env.TEST_NODE2_IP ?? "";
export const TEST_SSH_PASSWORD = process.env.TEST_NODE2_PASSWORD ?? "";

/** application 测试使用的命名空间（a 开头，UI 下拉排序靠前） */
export const applicationTestNamespace = "a-e2e-test";

/** 本地测试 Helm 包名称 */
export const componentNameApp = "e2e-helm-release";
export const componentNameExt = "e2e-monitor-extension";
export const componentNameExt2 = "e2e-logging-extension";

/** 应用市场本地仓库名称 */
export const localPackageSource = "local";

/** 应用、扩展组件容器名称（主容器与 chart 同名；sidecar 无日志） */
export const chartSidecarContainer = `${componentNameApp}-sidecar`;
export const extSidecarContainer = `${componentNameExt}-sidecar`;

/** 本地测试包统一基线 / 备升级版本 */
export const packageVersion = "1.0.0";
export const packageVersionNext = "1.1.0";

export type EntryType = "extendManage" | "applicationManageHelm";

export type DeployPlan = {
    componentName: string;
    appName: string;
    entry: EntryType;
    version?: string;
    source?: string;
    appNamePrefix?: string;
};

export type DeployPlanTemplate = {
    componentName: string;
    entry: EntryType;
    version?: string;
    source?: string;
    appNamePrefix: string;
};

export const APP_PREFIX = "test-app";
export const EXT_PREFIX = "test-extension";

/**
 * 计算 chart 对应的 K8s 资源名称。
 *
 * e2e-helm-release 使用标准 helm fullname（releaseName 包含 chartName 时直接用
 * releaseName，否则拼接 `${releaseName}-${chartName}`）。
 * 扩展组件 chart 中 Deployment 名称是硬编码的 chartName，直接返回 chartName。
 */
export function chartResourceName(releaseName: string, chartName: string): string {
    if (chartName !== componentNameApp) {
        // extension charts hardcode deployment name to chart name
        return chartName;
    }
    return releaseName.includes(chartName) ? releaseName : `${releaseName}-${chartName}`;
}

/** Helm app deploy plan template */
export function appPlan(
    componentName = componentNameApp,
    prefix = APP_PREFIX,
): DeployPlanTemplate {
    return {
        componentName,
        entry: "applicationManageHelm",
        appNamePrefix: prefix,
        version: packageVersion,
        source: localPackageSource,
    };
}

/** Extension component deploy plan template */
export function extPlan(
    componentName = componentNameExt,
    prefix = EXT_PREFIX,
): DeployPlanTemplate {
    return {
        componentName,
        entry: "extendManage",
        appNamePrefix: prefix,
        version: packageVersion,
        source: localPackageSource,
    };
}

/**
 * 根据模板列表生成 DeployPlan，内置随机 hash 保证每次运行 appName 唯一。
 * 注意：hash 必须在调用方生成，不要从外部传入。
 */
export function buildDeployPlans(templates: DeployPlanTemplate[]): DeployPlan[] {
    const hash = randomBytes(4).toString("hex");
    const multi = templates.length > 1;
    return templates.map((template, index) => ({
        componentName: template.componentName,
        entry: template.entry,
        version: template.version,
        source: template.source ?? localPackageSource,
        appNamePrefix: template.appNamePrefix,
        appName: multi
            ? `${template.appNamePrefix}-${hash}-${index}`
            : `${template.appNamePrefix}-${hash}`,
    }));
}
