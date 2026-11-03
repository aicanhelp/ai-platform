import { NodeSSH } from "node-ssh";

type SshClient = InstanceType<typeof NodeSSH>;

export const BKEAGENT_SOURCE_BASE_URL =
    "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/cluster-api-provider-bke/releases/download";
export const BKEAGENT_LOCAL_DIR = "/bke/mount/source_registry/files";

function shellQuote(value: string | number): string {
    return `'${String(value).replace(/'/g, `'"'"'`)}'`;
}

function buildKubectlCmd(kubeconfig: string, kubectlCmd = "kubectl"): string {
    return `${kubectlCmd} --kubeconfig=${shellQuote(kubeconfig)}`;
}

function assertSshCredentials(
    host: string | undefined,
    username: string | undefined,
    password: string | undefined,
): void {
    if (!host || !username || !password) {
        throw new Error("SSH host/username/password are required");
    }
}

export async function withSshConnection<T>(
    {
        host,
        username,
        password,
        port = 22,
        readyTimeout = 15000,
    }: {
        host: string;
        username: string;
        password: string;
        port?: number;
        readyTimeout?: number;
    },
    handler: (ssh: SshClient) => Promise<T>,
): Promise<T> {
    assertSshCredentials(host, username, password);
    const ssh = new NodeSSH();
    try {
        await ssh.connect({ host, username, password, port, readyTimeout });
        return await handler(ssh);
    } finally {
        ssh.dispose();
    }
}

async function execOrThrow(ssh: SshClient, command: string, errorMessage: string) {
    const result = await ssh.execCommand(command);
    if (result.code !== 0) {
        throw new Error(`${errorMessage}: ${result.stderr || result.stdout}`);
    }
    return result;
}

export async function prepareBkeagentBinaryOnGuideNode({
    host,
    username,
    password,
    version,
    port = 22,
    readyTimeout = 15000,
}: {
    host: string;
    username: string;
    password: string;
    version: string;
    port?: number;
    readyTimeout?: number;
}) {
    if (!version) {
        throw new Error("version is required");
    }

    return withSshConnection({ host, username, password, port, readyTimeout }, async (ssh) => {
        const archCmd = `ARCH=$(uname -m)
case $ARCH in
x86_64) ARCH="amd64";;
aarch64) ARCH="arm64";;
esac
printf '%s' "$ARCH"`;
        const archResult = await execOrThrow(ssh, archCmd, "Failed to detect guide node architecture");
        const arch = String(archResult.stdout || "").trim();
        if (!arch) {
            throw new Error("guide node architecture is empty");
        }

        const sourceFileName = `bkeagent_linux_${arch}`;
        const targetFileName = `bkeagent-${version}-linux-${arch}`;
        const sourceUrl = `${BKEAGENT_SOURCE_BASE_URL}/${version}/${sourceFileName}`;
        const targetPath = `${BKEAGENT_LOCAL_DIR}/${targetFileName}`;

        const syncCmd = `set -euo pipefail
mkdir -p ${shellQuote(BKEAGENT_LOCAL_DIR)}
wget -q -O ${shellQuote(targetPath)} ${shellQuote(sourceUrl)}
chmod 0644 ${shellQuote(targetPath)}
ls -l ${shellQuote(targetPath)}`;
        await execOrThrow(ssh, syncCmd, "Failed to sync bkeagent binary to guide node");

        return {
            sourceUrl,
            targetPath,
            arch,
            version,
        };
    });
}

async function findInstallerServicePod(ssh: SshClient, namespace: string, kubeconfig: string) {
    const kubectl = buildKubectlCmd(kubeconfig);
    const findCmd = [
        `${kubectl} get pod -n ${shellQuote(namespace)} -o name`,
        "grep '^pod/installer-service'",
        "head -n 1",
        "sed 's#^pod/##'",
    ].join(" | ");

    const findResult = await execOrThrow(ssh, findCmd, "Failed to find installer-service pod");

    const podName = String(findResult.stdout || "").trim();
    if (!podName) {
        throw new Error("Cannot find installer-service pod");
    }

    return { namespace, podName };
}

async function editInstallerServiceDeployment(
    ssh: SshClient,
    namespace: string,
    remotePatchPath: string,
    remoteDeployPath: string,
    kubeconfig: string,
) {
    const kubectl = buildKubectlCmd(kubeconfig);
    const editCmd = `${kubectl} set env deployment/installer-service -n ${shellQuote(namespace)} REMOTE_PATCH_PATH=${shellQuote(remotePatchPath)} REMOTE_DEPLOY_PATH=${shellQuote(remoteDeployPath)}`;

    await execOrThrow(ssh, editCmd, "Failed to update Deployment env");

    const rolloutCmd = `${kubectl} rollout status deployment/installer-service -n ${shellQuote(namespace)} --timeout=180s`;
    await execOrThrow(ssh, rolloutCmd, "Deployment rollout not ready");
}

async function verifyInstallerServiceDeployment(
    ssh: SshClient,
    namespace: string,
    remotePatchPath: string,
    remoteDeployPath: string,
    kubeconfig: string,
) {
    const kubectl = buildKubectlCmd(kubeconfig);
    const verifyCmd = `${kubectl} get deploy installer-service -n ${shellQuote(namespace)} -o yaml | awk '/name: REMOTE_PATCH_PATH/{getline; print}/name: REMOTE_DEPLOY_PATH/{getline; print}'`;
    const verifyResult = await execOrThrow(ssh, verifyCmd, "Failed to verify Deployment env");

    const output = String(verifyResult.stdout || "");
    const hasPatchPath = output.includes(remotePatchPath);
    const hasDeployPath = output.includes(remoteDeployPath);
    if (!hasPatchPath || !hasDeployPath) {
        throw new Error(`Env verification failed. Output: ${output}`);
    }

    return output;
}

export async function updateInstallerServiceRemotePaths({
    host,
    username,
    password,
    remotePatchPath,
    remoteDeployPath,
    namespace = "openfuyao-system",
    kubeconfig = "/etc/rancher/k3s/k3s.yaml",
    port = 22,
    readyTimeout = 15000,
}: {
    host: string;
    username: string;
    password: string;
    remotePatchPath: string;
    remoteDeployPath: string;
    namespace?: string;
    kubeconfig?: string;
    port?: number;
    readyTimeout?: number;
}) {
    if (!remotePatchPath || !remoteDeployPath) {
        throw new Error("remotePatchPath and remoteDeployPath are required");
    }

    return withSshConnection({ host, username, password, port, readyTimeout }, async (ssh) => {
        await editInstallerServiceDeployment(ssh, namespace, remotePatchPath, remoteDeployPath, kubeconfig);
        const { podName } = await findInstallerServicePod(ssh, namespace, kubeconfig);
        const verifyOutput = await verifyInstallerServiceDeployment(
            ssh,
            namespace,
            remotePatchPath,
            remoteDeployPath,
            kubeconfig,
        );

        return {
            namespace,
            podName,
            remotePatchPath,
            remoteDeployPath,
            verifyOutput,
        };
    });
}

async function editBkeControllerManagerDeployment(
    ssh: SshClient,
    namespace: string,
    manifestsImage: string,
    kubeconfig: string,
) {
    const kubectl = buildKubectlCmd(kubeconfig);
    const patchPayload = JSON.stringify({
        spec: {
            template: {
                spec: {
                    initContainers: [
                        {
                            name: "manifests",
                            image: manifestsImage,
                        },
                    ],
                },
            },
        },
    });
    const patchCmd = `${kubectl} patch deployment bke-controller-manager -n ${shellQuote(namespace)} --type='strategic' -p=${shellQuote(patchPayload)}`;

    await execOrThrow(ssh, patchCmd, "Failed to patch bke-controller-manager image");

    const rolloutCmd = `${kubectl} rollout status deployment/bke-controller-manager -n ${shellQuote(namespace)} --timeout=180s`;
    await execOrThrow(ssh, rolloutCmd, "bke-controller-manager rollout not ready");
}

async function verifyBkeControllerManagerDeployment(
    ssh: SshClient,
    namespace: string,
    manifestsImage: string,
    kubeconfig: string,
) {
    const kubectl = buildKubectlCmd(kubeconfig);
    const verifyCmd = `${kubectl} get deploy bke-controller-manager -n ${shellQuote(namespace)} -o jsonpath='{.spec.template.spec.initContainers[?(@.name=="manifests")].image}'`;
    const verifyResult = await execOrThrow(ssh, verifyCmd, "Failed to verify bke-controller-manager image");

    const output = String(verifyResult.stdout || "").trim().replace(/^'+|'+$/g, "");
    if (output !== manifestsImage) {
        throw new Error(`bke-controller-manager image verification failed. expected=${manifestsImage}, actual=${output}`);
    }

    return output;
}

export async function updateBkeControllerManagerManifests({
    host,
    username,
    password,
    manifestsImage,
    namespace = "cluster-system",
    kubeconfig = "/etc/rancher/k3s/k3s.yaml",
    port = 22,
    readyTimeout = 15000,
}: {
    host: string;
    username: string;
    password: string;
    manifestsImage: string;
    namespace?: string;
    kubeconfig?: string;
    port?: number;
    readyTimeout?: number;
}) {
    if (!manifestsImage) {
        throw new Error("manifestsImage is required");
    }

    return withSshConnection({ host, username, password, port, readyTimeout }, async (ssh) => {
        await editBkeControllerManagerDeployment(ssh, namespace, manifestsImage, kubeconfig);
        const verifyOutput = await verifyBkeControllerManagerDeployment(ssh, namespace, manifestsImage, kubeconfig);

        return {
            namespace,
            manifestsImage,
            verifyOutput,
        };
    });
}

async function waitKubectlAndDaemonSetReady(
    ssh: SshClient,
    namespace: string,
    daemonSetName: string,
    kubeconfig: string,
    kubectlCmd: string,
    maxAttempts = 12,
    intervalSeconds = 10,
    rolloutTimeoutSeconds = 300,
) {
    const kubectl = buildKubectlCmd(kubeconfig, kubectlCmd);
    let lastError = "";

    for (let i = 1; i <= maxAttempts; i++) {
        const checkDsCmd = `${kubectl} -n ${shellQuote(namespace)} get ds ${shellQuote(daemonSetName)}`;
        const checkDsResult = await ssh.execCommand(checkDsCmd);

        if (checkDsResult.code === 0) {
            const rolloutCmd = `${kubectl} -n ${shellQuote(namespace)} rollout status ds/${shellQuote(daemonSetName)} --timeout=${rolloutTimeoutSeconds}s`;
            const rolloutResult = await ssh.execCommand(rolloutCmd);
            if (rolloutResult.code === 0) {
                return;
            }
            lastError = rolloutResult.stderr || rolloutResult.stdout || `rollout status failed at attempt ${i}`;
        } else {
            lastError = checkDsResult.stderr || checkDsResult.stdout || `get ds failed at attempt ${i}`;
        }

        if (i < maxAttempts) {
            await new Promise((resolve) => setTimeout(resolve, intervalSeconds * 1000));
        }
    }

    throw new Error(`kubectl/daemonset readiness check failed after ${maxAttempts} attempts: ${lastError}`);
}

async function setDaemonSetImage(
    ssh: SshClient,
    namespace: string,
    daemonSetName: string,
    image: string,
    kubeconfig: string,
    kubectlCmd: string,
    waitRollout = true,
) {
    const kubectl = buildKubectlCmd(kubeconfig, kubectlCmd);
    const setImageCmd = `${kubectl} set image daemonset/${shellQuote(daemonSetName)} -n ${shellQuote(namespace)} \*=${shellQuote(image)}`;
    await execOrThrow(ssh, setImageCmd, "Failed to set daemonset image");

    if (!waitRollout) {
        return;
    }

    const rolloutCmd = `${kubectl} rollout status daemonset/${shellQuote(daemonSetName)} -n ${shellQuote(namespace)} --timeout=180s`;
    await execOrThrow(ssh, rolloutCmd, "DaemonSet rollout not ready");
}

async function verifyDaemonSetImage(
    ssh: SshClient,
    namespace: string,
    daemonSetName: string,
    expectedImage: string,
    kubeconfig: string,
    kubectlCmd: string,
) {
    const kubectl = buildKubectlCmd(kubeconfig, kubectlCmd);
    const verifyCmd = `${kubectl} get daemonset ${shellQuote(daemonSetName)} -n ${shellQuote(namespace)} -o jsonpath='{.spec.template.spec.containers[*].image}'`;
    const verifyResult = await execOrThrow(ssh, verifyCmd, "Failed to verify daemonset image");

    const output = String(verifyResult.stdout || "").trim().replace(/^'+|'+$/g, "");
    if (!output.includes(expectedImage)) {
        throw new Error(`DaemonSet image verification failed. expected includes=${expectedImage}, actual=${output}`);
    }

    return output;
}

export async function updateBkeagentDeployerImage({
    host,
    username,
    password,
    image,
    namespace = "cluster-system",
    daemonSetName = "bkeagent-deployer",
    kubeconfig = "/etc/kubernetes/admin.conf",
    kubectlCmd = "/usr/bin/kubectl",
    port = 22,
    readyTimeout = 15000,
    waitBeforeSetImage = true,
    waitBeforeSetImageTimeoutSeconds = 180,
    waitRollout = true,
}: {
    host: string;
    username: string;
    password: string;
    image: string;
    namespace?: string;
    daemonSetName?: string;
    kubeconfig?: string;
    kubectlCmd?: string;
    port?: number;
    readyTimeout?: number;
    waitBeforeSetImage?: boolean;
    waitBeforeSetImageTimeoutSeconds?: number;
    waitRollout?: boolean;
}) {
    if (!image) {
        throw new Error("image is required");
    }

    return withSshConnection({ host, username, password, port, readyTimeout }, async (ssh) => {
        const diagnoseCmd = [
            "hostname",
            "whoami",
            "echo $PATH",
            `echo kubectlCmd=${shellQuote(kubectlCmd)}`,
            "command -v kubectl || true",
            "type kubectl || true",
            "ls -l /usr/bin/kubectl /usr/local/bin/kubectl /bin/kubectl 2>/dev/null || true",
            `${buildKubectlCmd(kubeconfig, kubectlCmd)} config current-context || true`,
            `${buildKubectlCmd(kubeconfig, kubectlCmd)} get ns cluster-system -o name || true`,
            `${buildKubectlCmd(kubeconfig, kubectlCmd)} get ds -n cluster-system bkeagent-deployer -o name || true`,
        ].join("; ");
        const diagnoseResult = await ssh.execCommand(diagnoseCmd);
        const diagnoseStdout = String(diagnoseResult.stdout || "").trim();
        const diagnoseStderr = String(diagnoseResult.stderr || "").trim();

        try {
            if (waitBeforeSetImage) {
                await waitKubectlAndDaemonSetReady(
                    ssh,
                    namespace,
                    daemonSetName,
                    kubeconfig,
                    kubectlCmd,
                    12,
                    10,
                    waitBeforeSetImageTimeoutSeconds,
                );
            }

            await setDaemonSetImage(ssh, namespace, daemonSetName, image, kubeconfig, kubectlCmd, waitRollout);
            const verifyOutput = await verifyDaemonSetImage(
                ssh,
                namespace,
                daemonSetName,
                image,
                kubeconfig,
                kubectlCmd,
            );

            return {
                namespace,
                daemonSetName,
                image,
                verifyOutput,
                diagnoseStdout,
                diagnoseStderr,
            };
        } catch (error: unknown) {
            const msg = error instanceof Error ? error.message : String(error);
            throw new Error(`${msg}\n[kubectl diagnose stdout]\n${diagnoseStdout}\n[kubectl diagnose stderr]\n${diagnoseStderr}`);
        }
    });
}

async function setDeploymentImage(
    ssh: SshClient,
    namespace: string,
    deploymentName: string,
    containerName: string,
    image: string,
    kubeconfig: string,
    waitRollout = true,
) {
    const kubectl = buildKubectlCmd(kubeconfig);
    const setImageCmd = `${kubectl} set image deployment/${shellQuote(deploymentName)} -n ${shellQuote(namespace)} ${shellQuote(containerName)}=${shellQuote(image)}`;
    await execOrThrow(ssh, setImageCmd, "Failed to set deployment image");

    if (!waitRollout) {
        return;
    }

    const rolloutCmd = `${kubectl} rollout status deployment/${shellQuote(deploymentName)} -n ${shellQuote(namespace)} --timeout=180s`;
    await execOrThrow(ssh, rolloutCmd, "Deployment rollout not ready");
}

async function getDeploymentImage(
    ssh: SshClient,
    namespace: string,
    deploymentName: string,
    containerName: string,
    kubeconfig: string,
) {
    const kubectl = buildKubectlCmd(kubeconfig);
    const getCmd = `${kubectl} get deployment ${shellQuote(deploymentName)} -n ${shellQuote(namespace)} -o jsonpath='{.spec.template.spec.containers[?(@.name=="${containerName}")].image}'`;
    const getResult = await execOrThrow(ssh, getCmd, "Failed to get deployment image");

    const output = String(getResult.stdout || "").trim().replace(/^'+|'+$/g, "");
    if (!output) {
        throw new Error(`Cannot find deployment container image: ${namespace}/${deploymentName}:${containerName}`);
    }

    return output;
}

async function restartDeploymentPods(ssh: SshClient, namespace: string, deploymentName: string, kubeconfig: string) {
    const kubectl = buildKubectlCmd(kubeconfig);
    const restartCmd = `${kubectl} rollout restart deployment/${shellQuote(deploymentName)} -n ${shellQuote(namespace)}`;
    await execOrThrow(ssh, restartCmd, "Failed to restart deployment pods");
}

async function getDeploymentPodSelector(
    ssh: SshClient,
    namespace: string,
    deploymentName: string,
    kubeconfig: string,
) {
    const kubectl = buildKubectlCmd(kubeconfig);
    const selectorCmd = `${kubectl} get deployment ${shellQuote(deploymentName)} -n ${shellQuote(namespace)} -o go-template='{{range $k,$v := .spec.selector.matchLabels}}{{printf "%s=%s," $k $v}}{{end}}'`;
    const selectorResult = await execOrThrow(ssh, selectorCmd, "Failed to get deployment selector");

    const selector = String(selectorResult.stdout || "").trim().replace(/,+$/, "");
    if (!selector) {
        throw new Error(`Deployment selector is empty: ${namespace}/${deploymentName}`);
    }

    return selector;
}

async function waitDeploymentImagePullError(
    ssh: SshClient,
    namespace: string,
    deploymentName: string,
    kubeconfig: string,
    maxAttempts = 24,
    intervalSeconds = 10,
) {
    const kubectl = buildKubectlCmd(kubeconfig);
    const selector = await getDeploymentPodSelector(ssh, namespace, deploymentName, kubeconfig);

    for (let i = 1; i <= maxAttempts; i++) {
        const reasonCmd = `${kubectl} get pods -n ${shellQuote(namespace)} -l ${shellQuote(selector)} -o jsonpath='{range .items[*]}{range .status.containerStatuses[*]}{.state.waiting.reason}{"\n"}{end}{end}'`;
        const reasonResult = await ssh.execCommand(reasonCmd);
        const reasonOutput = `${reasonResult.stdout || ""}\n${reasonResult.stderr || ""}`;
        if (reasonResult.code === 0 && /(ImagePullBackOff|ErrImagePull|InvalidImageName)/.test(reasonOutput)) {
            return reasonOutput;
        }

        const eventsCmd = `${kubectl} get events -n ${shellQuote(namespace)} --field-selector involvedObject.kind=Pod,type=Warning --sort-by=.lastTimestamp | tail -n 30`;
        const eventsResult = await ssh.execCommand(eventsCmd);
        const eventsOutput = `${eventsResult.stdout || ""}\n${eventsResult.stderr || ""}`;
        if (/(ImagePullBackOff|ErrImagePull|InvalidImageName|Failed to pull image|Back-off pulling image)/.test(eventsOutput)) {
            return `${reasonOutput}\n${eventsOutput}`;
        }

        const describeCmd = `${kubectl} describe deploy ${shellQuote(deploymentName)} -n ${shellQuote(namespace)}`;
        const describeResult = await ssh.execCommand(describeCmd);
        const describeOutput = `${describeResult.stdout || ""}\n${describeResult.stderr || ""}`;
        if (/(ImagePullBackOff|ErrImagePull|InvalidImageName|Failed to pull image)/.test(describeOutput)) {
            return `${reasonOutput}\n${describeOutput}`;
        }

        if (i < maxAttempts) {
            await new Promise((resolve) => setTimeout(resolve, intervalSeconds * 1000));
        }
    }

    throw new Error(`Timed out waiting deployment image pull error: ${namespace}/${deploymentName}, selector=${selector}`);
}

export async function getDeploymentContainerImage({
    host,
    username,
    password,
    namespace,
    deploymentName,
    containerName,
    kubeconfig = "/etc/kubernetes/admin.conf",
    port = 22,
    readyTimeout = 15000,
}: {
    host: string;
    username: string;
    password: string;
    namespace: string;
    deploymentName: string;
    containerName: string;
    kubeconfig?: string;
    port?: number;
    readyTimeout?: number;
}) {
    if (!namespace || !deploymentName || !containerName) {
        throw new Error("namespace/deploymentName/containerName are required");
    }

    return withSshConnection({ host, username, password, port, readyTimeout }, async (ssh) => {
        return await getDeploymentImage(ssh, namespace, deploymentName, containerName, kubeconfig);
    });
}

export async function updateDeploymentContainerImage({
    host,
    username,
    password,
    namespace,
    deploymentName,
    containerName,
    image,
    kubeconfig = "/etc/kubernetes/admin.conf",
    port = 22,
    readyTimeout = 15000,
    waitRollout = true,
}: {
    host: string;
    username: string;
    password: string;
    namespace: string;
    deploymentName: string;
    containerName: string;
    image: string;
    kubeconfig?: string;
    port?: number;
    readyTimeout?: number;
    waitRollout?: boolean;
}) {
    if (!namespace || !deploymentName || !containerName || !image) {
        throw new Error("namespace/deploymentName/containerName/image are required");
    }

    return withSshConnection({ host, username, password, port, readyTimeout }, async (ssh) => {
        await setDeploymentImage(ssh, namespace, deploymentName, containerName, image, kubeconfig, waitRollout);
        const verifyOutput = await getDeploymentImage(ssh, namespace, deploymentName, containerName, kubeconfig);
        if (verifyOutput !== image) {
            throw new Error(`Deployment image verification failed. expected=${image}, actual=${verifyOutput}`);
        }

        return {
            namespace,
            deploymentName,
            containerName,
            image,
            verifyOutput,
        };
    });
}

export async function triggerDeploymentImagePullError({
    host,
    username,
    password,
    namespace,
    deploymentName,
    kubeconfig = "/etc/kubernetes/admin.conf",
    port = 22,
    readyTimeout = 15000,
    maxAttempts = 24,
    intervalSeconds = 10,
}: {
    host: string;
    username: string;
    password: string;
    namespace: string;
    deploymentName: string;
    kubeconfig?: string;
    port?: number;
    readyTimeout?: number;
    maxAttempts?: number;
    intervalSeconds?: number;
}) {
    if (!namespace || !deploymentName) {
        throw new Error("namespace/deploymentName are required");
    }

    return withSshConnection({ host, username, password, port, readyTimeout }, async (ssh) => {
        await restartDeploymentPods(ssh, namespace, deploymentName, kubeconfig);
        const evidence = await waitDeploymentImagePullError(
            ssh,
            namespace,
            deploymentName,
            kubeconfig,
            maxAttempts,
            intervalSeconds,
        );

        return {
            namespace,
            deploymentName,
            evidence,
        };
    });
}
