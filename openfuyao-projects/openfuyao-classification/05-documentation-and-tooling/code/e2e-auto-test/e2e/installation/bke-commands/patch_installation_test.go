package bke_commands

import (
	"encoding/base64"
	"fmt"
	"os"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	installation "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"
)

func getEnvOrDefault(key, defaultValue string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return defaultValue
	}
	return value
}

// 补丁包构建与部署相关用例（OCI 策略离线包）
var _ = BKECommandDescribe("补丁包构建和部署", Label("installation", "patch", "post-init", "online"), Ordered, func() {
	var (
		sshExecutor *executor.SSHExecutor
		guideConfig *config.GuideNodeConfig

		patchYAMLPath string
		patchTarPath  string
		extractedPath string
		TarName       string
		patchPath     string
		largeYAMLPath string
		largeTarName  string
		largeTarPath  string
	)

	BeforeAll(func() {
		guideConfig = config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty(), "GUIDE_NODE_HOST 环境变量必须设置")
		Expect(guideConfig.Password).NotTo(BeEmpty(), "GUIDE_NODE_PASSWORD 环境变量必须设置")

		var err error
		sshExecutor, err = executor.NewSSHExecutor(
			guideConfig.Host,
			guideConfig.Port,
			guideConfig.Username,
			guideConfig.Password,
		)
		Expect(err).NotTo(HaveOccurred(), "应该成功连接到引导节点")

		// 确定 testpatch.yaml 文件路径（统一在 /root 目录下）
		patchYAMLPath = "/root/testpatch.yaml"
		patchTarPath = "/root/testpatch.tar.gz"
		extractedPath = "/root/testpatch"
		patchPath = "./testpatch"
		TarName = "testpatch.tar.gz"
		largeYAMLPath = "/root/testpatch-large.yaml"
		largeTarName = "testpatch-large.tar.gz"
		largeTarPath = "/root/testpatch-large.tar.gz"
	})

	AfterAll(func() {
		// 清理临时文件
		if sshExecutor != nil {
			if patchTarPath != "" {
				sshExecutor.Exec(fmt.Sprintf("rm -f %s", patchTarPath))
			}
			if extractedPath != "" {
				sshExecutor.Exec(fmt.Sprintf("rm -rf %s", extractedPath))
			}
			if patchYAMLPath != "" {
				sshExecutor.Exec(fmt.Sprintf("rm -f %s", patchYAMLPath))
			}
			if largeTarPath != "" {
				sshExecutor.Exec(fmt.Sprintf("rm -f %s", largeTarPath))
			}
			if largeTarName != "" {
				sshExecutor.Exec(fmt.Sprintf("rm -f /root/%s", largeTarName))
			}
			if largeYAMLPath != "" {
				sshExecutor.Exec(fmt.Sprintf("rm -f %s", largeYAMLPath))
			}
			_ = sshExecutor.Close()
		}
	})

	// Author: ZYH-ang
	It("应该成功构建OCI策略补丁包", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		By("上传 testpatch.yaml 文件到引导节点")
		// 获取 testpatch.yaml 文件内容
		fileContent := installation.GetTestPatchYAMLContent()

		// 使用 base64 编码上传文件，避免特殊字符问题
		encoded := base64.StdEncoding.EncodeToString(fileContent)
		uploadCmd := fmt.Sprintf("cat <<'EOF' | base64 -d > %s\n%s\nEOF", patchYAMLPath, encoded)
		uploadResult, err := sshExecutor.Exec(uploadCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功上传 testpatch.yaml 文件")
		Expect(uploadResult.ExitCode).To(Equal(0), fmt.Sprintf("上传文件应该成功，stderr: %s", uploadResult.Stderr))

		// 验证文件已上传
		checkFileCmd := fmt.Sprintf("cd /root && test -f %s && echo exists || echo notfound", patchYAMLPath)
		fileResult, err := sshExecutor.Exec(checkFileCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功检查上传的文件")
		Expect(fileResult.Stdout).To(ContainSubstring("exists"), "testpatch.yaml 文件应该存在")
		GinkgoWriter.Printf("testpatch.yaml 文件已上传到引导节点: %s\n", patchYAMLPath)

		By("执行 bke build patch 命令")
		buildCmd := fmt.Sprintf("cd /root && bke build patch -f %s -t %s --strategy=oci", patchYAMLPath, TarName)
		result, err := sshExecutor.Exec(buildCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功执行 bke build patch 命令")
		Expect(result.ExitCode).To(Equal(0), fmt.Sprintf("bke build patch 应该成功退出，stderr: %s", result.Stderr))

		// 验证输出包含 "Packaging complete"
		By("验证输出包含 'Packaging complete'")
		Eventually(func() bool {
			return strings.Contains(result.Stdout, "Packaging complete")
		}).WithTimeout(180*time.Second). // 180 秒超时
							WithPolling(2*time.Second). // 每 2 秒检查一次
							Should(BeTrue(), "输出应该包含 'Packaging complete'")
		GinkgoWriter.Printf("构建输出:\n%s\n", result.Stdout)

		By("验证补丁包文件存在")
		checkFileCmd = fmt.Sprintf("cd /root && test -f %s && echo exists || echo notfound", patchTarPath)
		fileResult, err = sshExecutor.Exec(checkFileCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功检查补丁包文件")
		Expect(fileResult.Stdout).To(ContainSubstring("exists"), "补丁包文件应该存在")
		GinkgoWriter.Printf("补丁包文件已创建: %s\n", patchTarPath)
	})

	// Author: ZYH-ang
	It("应该成功解压并部署OCI补丁包", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		By("验证补丁包文件存在")
		checkFileCmd := fmt.Sprintf("cd /root && test -f %s && echo exists || echo notfound", patchTarPath)
		fileResult, err := sshExecutor.Exec(checkFileCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功检查补丁包文件")
		Expect(fileResult.Stdout).To(ContainSubstring("exists"), "补丁包文件应该存在")

		By("解压补丁包")
		extractCmd := fmt.Sprintf("cd /root && tar -xzvf %s", TarName)
		extractResult, err := sshExecutor.Exec(extractCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功解压补丁包")
		Expect(extractResult.ExitCode).To(Equal(0), fmt.Sprintf("解压应该成功，stderr: %s", extractResult.Stderr))

		By("验证解压目录存在")
		checkExtractedCmd := fmt.Sprintf("cd /root && test -d %s && echo exists || echo notfound", extractedPath)
		extractedResult, err := sshExecutor.Exec(checkExtractedCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功检查解压目录")
		Expect(extractedResult.Stdout).To(ContainSubstring("exists"), "解压目录应该存在")

		By("验证 OCI Layout 目录结构（【制备离线包-003】）")
		layoutBase := extractedPath + "/volumes/oci-layout"
		checkVolOCI := fmt.Sprintf("test -d %s && echo exists || echo notfound", layoutBase)
		volOCIResult, err := sshExecutor.Exec(checkVolOCI)
		Expect(err).NotTo(HaveOccurred(), "应该成功检查 volumes/oci-layout 目录")
		Expect(volOCIResult.ExitCode).To(Equal(0), fmt.Sprintf("检查 volumes/oci-layout，stderr: %s", volOCIResult.Stderr))
		Expect(volOCIResult.Stdout).To(ContainSubstring("exists"), "应存在 volumes/oci-layout 目录")

		checkOCI := fmt.Sprintf("test -f %s/oci-layout && echo exists || echo notfound", layoutBase)
		ociFileResult, err := sshExecutor.Exec(checkOCI)
		Expect(err).NotTo(HaveOccurred(), "应该成功检查 oci-layout 文件")
		Expect(ociFileResult.ExitCode).To(Equal(0), fmt.Sprintf("检查 oci-layout 文件，stderr: %s", ociFileResult.Stderr))
		Expect(ociFileResult.Stdout).To(ContainSubstring("exists"), "应存在 volumes/oci-layout/oci-layout（OCI Image Layout 版本声明）")

		checkIndex := fmt.Sprintf("test -f %s/index.json && echo exists || echo notfound", layoutBase)
		indexResult, err := sshExecutor.Exec(checkIndex)
		Expect(err).NotTo(HaveOccurred(), "应该成功检查 index.json")
		Expect(indexResult.ExitCode).To(Equal(0), fmt.Sprintf("检查 index.json，stderr: %s", indexResult.Stderr))
		Expect(indexResult.Stdout).To(ContainSubstring("exists"), "应存在 volumes/oci-layout/index.json")

		checkBlobsDir := fmt.Sprintf("test -d %s/blobs/sha256 && echo exists || echo notfound", layoutBase)
		blobsDirResult, err := sshExecutor.Exec(checkBlobsDir)
		Expect(err).NotTo(HaveOccurred(), "应该成功检查 blobs/sha256 目录")
		Expect(blobsDirResult.ExitCode).To(Equal(0), fmt.Sprintf("检查 blobs/sha256，stderr: %s", blobsDirResult.Stderr))
		Expect(blobsDirResult.Stdout).To(ContainSubstring("exists"), "应存在 volumes/oci-layout/blobs/sha256 目录")

		blobCountCmd := fmt.Sprintf("find %s/blobs/sha256 -type f 2>/dev/null | wc -l", layoutBase)
		blobCountResult, err := sshExecutor.Exec(blobCountCmd)
		Expect(err).NotTo(HaveOccurred(), "应该统计 blobs/sha256 下文件数量")
		Expect(blobCountResult.ExitCode).To(Equal(0), fmt.Sprintf("find wc 应成功，stderr: %s", blobCountResult.Stderr))
		blobCountStr := strings.TrimSpace(blobCountResult.Stdout)
		Expect(blobCountStr).NotTo(BeEmpty(), "blobs/sha256 文件数量输出不应为空")
		Expect(blobCountStr).NotTo(Equal("0"), "blobs/sha256 下应存在镜像层或 manifest 等 blob 文件")
		GinkgoWriter.Printf("OCI Layout 检查通过: %s（blob 文件数: %s）\n", layoutBase, blobCountStr)

		By("验证多架构镜像同步（【制备离线包-002】）")
		layoutDir := layoutBase

		By("找到 index.json 第一个 manifest digest，并解析对应blob")
		manifestBlobCmd := fmt.Sprintf(
			"cd %s && MANIFEST=$(cat index.json | jq -r '.manifests[0].digest' | sed 's/sha256://') && echo Manifest digest: $MANIFEST && cat blobs/sha256/$MANIFEST | jq '.'",
			layoutDir,
		)
		manifestBlobResult, err := sshExecutor.Exec(manifestBlobCmd)
		Expect(err).NotTo(HaveOccurred())
		Expect(manifestBlobResult.ExitCode).To(Equal(0), fmt.Sprintf("解析 manifest blob 应成功，stderr: %s", manifestBlobResult.Stderr))
		Expect(manifestBlobResult.Stdout).To(ContainSubstring("schemaVersion"), "manifest blob 应包含 schemaVersion 字段（jq 解析成功的兜底断言）")

		By("从 manifest blob 校验多架构 manifests（manifest 数量 >= 2）")
		blobManifestLenCmd := fmt.Sprintf(
			"cd %s && MANIFEST=$(cat index.json | jq -r '.manifests[0].digest' | sed 's/sha256://') && cat blobs/sha256/$MANIFEST | jq '.manifests | length'",
			layoutDir,
		)
		manifestLenResult, err := sshExecutor.Exec(blobManifestLenCmd)
		Expect(err).NotTo(HaveOccurred())
		Expect(manifestLenResult.ExitCode).To(Equal(0), fmt.Sprintf("检查 manifest blob 的 manifests 数量应成功，stderr: %s", manifestLenResult.Stderr))
		manifestLen := strings.TrimSpace(manifestLenResult.Stdout)
		Expect(manifestLen).NotTo(BeEmpty(), "manifests 数量不应为空")
		Expect(manifestLen).ToNot(Equal("0"), "manifests 数量应大于 0")
		Expect(manifestLen).ToNot(Equal("1"), "manifests 数量应大于等于 2（多架构）")

		By("从 manifest blob 校验 platforms 至少包含 amd64 与 arm64")
		archCmd := fmt.Sprintf(
			"cd %s && MANIFEST=$(cat index.json | jq -r '.manifests[0].digest' | sed 's/sha256://') && cat blobs/sha256/$MANIFEST | jq -r '.manifests[].platform.architecture' | sort -u",
			layoutDir,
		)
		archResult, err := sshExecutor.Exec(archCmd)
		Expect(err).NotTo(HaveOccurred())
		Expect(archResult.ExitCode).To(Equal(0), fmt.Sprintf("解析平台架构应成功，stderr: %s", archResult.Stderr))
		Expect(archResult.Stdout).To(ContainSubstring("amd64"), fmt.Sprintf("应包含 amd64 平台，实际输出: %s", strings.TrimSpace(archResult.Stdout)))
		Expect(archResult.Stdout).To(ContainSubstring("arm64"), fmt.Sprintf("应包含 arm64 平台，实际输出: %s", strings.TrimSpace(archResult.Stdout)))

		GinkgoWriter.Printf("OCI layout manifest blob manifests 数量: %s, platforms:\n%s\n", manifestLen, archResult.Stdout)

		By("执行 bke registry patch 部署命令")
		patchCmd := fmt.Sprintf("cd /root && bke registry patch --source %s --target 0.0.0.0:40443", patchPath)
		patchResult, err := sshExecutor.Exec(patchCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功执行 bke registry patch 命令")
		Expect(patchResult.ExitCode).To(Equal(0), fmt.Sprintf("bke registry patch 应该成功退出，stderr: %s", patchResult.Stderr))
		GinkgoWriter.Printf("部署输出:\n%s\n", patchResult.Stdout)
	})

	// Author: ZYH-ang
	It("大规模镜像同步（50+ 镜像）应成功构建并生成补丁包（【制备离线包-008】）", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		fileContent := installation.GetTestPatchYAMLContentLargeScale()
		encoded := base64.StdEncoding.EncodeToString(fileContent)
		uploadCmd := fmt.Sprintf("cat <<'EOF' | base64 -d > %s\n%s\nEOF", largeYAMLPath, encoded)
		uploadResult, err := sshExecutor.Exec(uploadCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功上传大规模镜像配置文件")
		Expect(uploadResult.ExitCode).To(Equal(0), fmt.Sprintf("上传文件应该成功，stderr: %s", uploadResult.Stderr))

		By("执行 bke build patch 命令")
		buildCmd := fmt.Sprintf("cd /root && bke build patch -f %s -t %s --strategy=oci", largeYAMLPath, largeTarName)
		result, err := sshExecutor.Exec(buildCmd)
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("bke build patch 应该成功，stderr: %s", result.Stderr))
		Expect(result.ExitCode).To(Equal(0), fmt.Sprintf("bke build patch 应该成功退出，stderr: %s", result.Stderr))
		Expect(result.Stdout).To(ContainSubstring("Packaging complete"), "输出应该包含 'Packaging complete'")

		By("验证补丁包文件存在")
		checkTarCmd := fmt.Sprintf("cd /root && test -f %s && echo exists || echo notfound", largeTarPath)
		tarResult, err := sshExecutor.Exec(checkTarCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功检查补丁包文件")
		Expect(tarResult.ExitCode).To(Equal(0))
		Expect(strings.TrimSpace(tarResult.Stdout)).To(Equal("exists"), "应生成补丁文件(.tar.gz)")
		GinkgoWriter.Printf("大规模补丁包文件已创建: %s\n", largeTarPath)
	})

	// Author: ZYH-ang
	It("应该支持网络中断后自动恢复制备离线包", Label("network-disconnect", "skip-temporarily"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		var (
			buildCmd      string
			interruptTime time.Duration = 30 * time.Second  // 30秒后中断网络
			recoveryTime  time.Duration = 30 * time.Second  // 30秒后恢复网络
			totalTimeout  time.Duration = 300 * time.Second // 总超时时间
		)

		By("准备 patch 配置文件（复用 getTestPatchYAMLContent）")
		fileContent := installation.GetTestPatchYAMLContent()
		encoded := base64.StdEncoding.EncodeToString(fileContent)
		uploadCmd := fmt.Sprintf("cat <<'EOF' | base64 -d > %s\n%s\nEOF", patchYAMLPath, encoded)
		uploadResult, err := sshExecutor.Exec(uploadCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功上传 patch 配置文件")
		Expect(uploadResult.ExitCode).To(Equal(0), fmt.Sprintf("上传文件应该成功，stderr: %s", uploadResult.Stderr))

		checkYAMLCmd := fmt.Sprintf("cd /root && test -f %s && echo exists || echo notfound", patchYAMLPath)
		yamlResult, err := sshExecutor.Exec(checkYAMLCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功检查 patch 配置文件")
		Expect(strings.TrimSpace(yamlResult.Stdout)).To(Equal("exists"), "patch 配置文件应该存在")

		// 清理历史产物，避免误判
		_, _ = sshExecutor.Exec(fmt.Sprintf("rm -f %s", patchTarPath))

		By("准备构建命令")
		buildCmd = fmt.Sprintf("cd /root && bke build patch -f %s -t %s --strategy=oci", patchYAMLPath, TarName)

		By("启动构建进程")
		resultChan := make(chan executor.ExecResult)
		errChan := make(chan error)

		go func() {
			result, err := sshExecutor.Exec(buildCmd)
			if err != nil {
				errChan <- err
				return
			}
			resultChan <- *result
		}()

		By(fmt.Sprintf("等待 %v 后模拟网络中断", interruptTime))
		time.Sleep(interruptTime)

		By("模拟网络中断 - 注释 nameserver 1.1.1.1")
		interruptCmd := "sed -i 's/^nameserver 1.1.1.1/# nameserver 1.1.1.1/' /etc/resolv.conf"
		_, err = sshExecutor.Exec(interruptCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功注释 nameserver")

		By(fmt.Sprintf("等待 %v 让网络中断生效", recoveryTime))
		time.Sleep(recoveryTime)

		By("恢复网络 - 取消注释 nameserver 1.1.1.1")
		recoveryCmd := "sed -i 's/^# nameserver 1.1.1.1/nameserver 1.1.1.1/' /etc/resolv.conf"
		_, err = sshExecutor.Exec(recoveryCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功恢复 nameserver")

		By("等待构建完成")
		select {
		case result := <-resultChan:
			Expect(result.ExitCode).To(Equal(0), fmt.Sprintf("构建应该成功完成，stderr: %s", result.Stderr))
			Expect(result.Stdout).To(ContainSubstring("Packaging complete"), "输出应该包含 'Packaging complete'")

			By("验证离线包文件存在")
			checkFileCmd := fmt.Sprintf("cd /root && test -f %s && echo exists || echo notfound", patchTarPath)
			fileResult, err := sshExecutor.Exec(checkFileCmd)
			Expect(err).NotTo(HaveOccurred(), "应该成功检查离线包文件")
			Expect(fileResult.Stdout).To(ContainSubstring("exists"), "离线包文件应该存在")

		case err := <-errChan:
			Fail(fmt.Sprintf("构建过程中发生错误: %v", err))
		case <-time.After(totalTimeout):
			Fail("构建超时，可能网络中断影响过大")
		}
	})

	// Author: ZYH-ang
	It("应该验证镜像 digest 一致", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		By("获取源镜像 digest")
		sourceDigestCmd := "skopeo inspect docker://cr.openfuyao.cn/openfuyao/user-management-operator:1.2.3 | jq -r .Digest"
		sourceResult, err := sshExecutor.Exec(sourceDigestCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功获取源镜像 digest")
		Expect(sourceResult.ExitCode).To(Equal(0), fmt.Sprintf("获取源镜像 digest 应该成功，stderr: %s", sourceResult.Stderr))
		sourceDigest := strings.TrimSpace(sourceResult.Stdout)
		Expect(sourceDigest).NotTo(BeEmpty(), "源镜像 digest 不应该为空")
		GinkgoWriter.Printf("源镜像 digest: %s\n", sourceDigest)

		By("获取 OCI 布局镜像 digest")
		ociDigestCmd := fmt.Sprintf("skopeo inspect oci:%s/volumes/oci-layout:kubernetes/user-management-operator:1.2.3 | jq -r .Digest", extractedPath)
		ociResult, err := sshExecutor.Exec(ociDigestCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功获取 OCI 布局镜像 digest")
		Expect(ociResult.ExitCode).To(Equal(0), fmt.Sprintf("获取 OCI 布局镜像 digest 应该成功，stderr: %s", ociResult.Stderr))
		ociDigest := strings.TrimSpace(ociResult.Stdout)
		Expect(ociDigest).NotTo(BeEmpty(), "OCI 布局镜像 digest 不应该为空")
		GinkgoWriter.Printf("OCI 布局镜像 digest: %s\n", ociDigest)

		By("获取目标 registry 镜像 digest")
		targetDigestCmd := "skopeo inspect docker://127.0.0.1:40443/kubernetes/user-management-operator:1.2.3 --tls-verify=false | jq -r .Digest"
		targetResult, err := sshExecutor.Exec(targetDigestCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功获取目标 registry 镜像 digest")
		Expect(targetResult.ExitCode).To(Equal(0), fmt.Sprintf("获取目标 registry 镜像 digest 应该成功，stderr: %s", targetResult.Stderr))
		targetDigest := strings.TrimSpace(targetResult.Stdout)
		Expect(targetDigest).NotTo(BeEmpty(), "目标 registry 镜像 digest 不应该为空")
		GinkgoWriter.Printf("目标 registry 镜像 digest: %s\n", targetDigest)

		By("验证三个 digest 相同")
		Expect(sourceDigest).To(Equal(ociDigest), "源镜像 digest 应该与 OCI 布局镜像 digest 相同")
		Expect(ociDigest).To(Equal(targetDigest), "OCI 布局镜像 digest 应该与目标 registry 镜像 digest 相同")
		Expect(sourceDigest).To(Equal(targetDigest), "源镜像 digest 应该与目标 registry 镜像 digest 相同")
		GinkgoWriter.Printf("所有镜像 digest 验证通过: %s\n", sourceDigest)
	})

	// Author: ZYH-ang
	It("原有构建方式验证", Label("skip-temporarily", "docker", "bke"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		By("准备构建配置文件")
		fileContent := installation.GetTestPatchYAMLContent()
		encoded := base64.StdEncoding.EncodeToString(fileContent)
		uploadCmd := fmt.Sprintf("cat <<'EOF' | base64 -d > %s\n%s\nEOF", patchYAMLPath, encoded)
		uploadResult, err := sshExecutor.Exec(uploadCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功上传构建配置文件")
		Expect(uploadResult.ExitCode).To(Equal(0), fmt.Sprintf("上传文件应该成功，stderr: %s", uploadResult.Stderr))

		By("执行原有 registry 方式的 bke build 构建指令")
		buildCmd := fmt.Sprintf("cd /root && bke build patch -f %s -t %s", patchYAMLPath, TarName)
		buildResult, err := sshExecutor.Exec(buildCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功执行构建指令")
		Expect(buildResult.ExitCode).To(Equal(0), fmt.Sprintf("构建应该成功，stderr: %s", buildResult.Stderr))
		Expect(buildResult.Stdout).To(ContainSubstring("Packaging complete"), "构建日志应该包含完成提示")

		By("检查构建产物是否存在")
		checkTarCmd := fmt.Sprintf("cd /root && test -f %s && echo exists || echo notfound", patchTarPath)
		tarResult, err := sshExecutor.Exec(checkTarCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功检查构建产物")
		Expect(strings.TrimSpace(tarResult.Stdout)).To(Equal("exists"), "应生成补丁包文件")
	})

	// Author: ZYH-ang
	It("增量离线镜像包流程制作", Label("skip-temporarily", "docker", "bke"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		By("准备 patchV1.yaml 文件")
		fileContent := installation.GetTestPatchYAMLContentLargeScale()
		encoded := base64.StdEncoding.EncodeToString(fileContent)
		patchV1YAMLPath := "/root/patchV1.yaml"
		patchV1TarName := "patchV1.tar.gz"
		patchV1TarPath := "/root/" + patchV1TarName
		uploadCmd := fmt.Sprintf("cat <<'EOF' | base64 -d > %s\n%s\nEOF", patchV1YAMLPath, encoded)
		uploadResult, err := sshExecutor.Exec(uploadCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功上传 patchV1.yaml")
		Expect(uploadResult.ExitCode).To(Equal(0), fmt.Sprintf("上传文件应该成功，stderr: %s", uploadResult.Stderr))

		By("执行增量制备命令")
		buildCmd := fmt.Sprintf("cd /root && bke build patch -f %s -t %s", patchV1YAMLPath, patchV1TarName)
		buildResult, err := sshExecutor.Exec(buildCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功执行增量制备")
		Expect(buildResult.ExitCode).To(Equal(0), fmt.Sprintf("增量制备应该成功，stderr: %s", buildResult.Stderr))
		Expect(buildResult.Stdout).To(ContainSubstring("Packaging complete"), "输出应该包含完成提示")

		By("同步本地镜像并检查补丁包产物")
		checkTarCmd := fmt.Sprintf("cd /root && test -f %s && echo exists || echo notfound", patchV1TarPath)
		tarResult, err := sshExecutor.Exec(checkTarCmd)
		Expect(err).NotTo(HaveOccurred(), "应该成功检查 patchV1 补丁包")
		Expect(strings.TrimSpace(tarResult.Stdout)).To(Equal("exists"), "应生成 patchV1.tar.gz")
	})
})
