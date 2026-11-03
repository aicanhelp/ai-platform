package local

import (
	"encoding/json"
	"fmt"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"helm.sh/helm/v3/pkg/chart"

	"application-management-service/pkg/constant"
	helmop "application-management-service/pkg/helm"
	helmmodel "application-management-service/pkg/models/helm"
	"application-management-service/pkg/server/param"
	"application-management-service/test/functional/local/chartgetter"
)

// --- Shared Helper Functions ---
func minimalChartForEnvtest(name, version string) *chart.Chart {
	return &chart.Chart{
		Metadata: &chart.Metadata{Name: name, Version: version, APIVersion: "v2"},
		Templates: []*chart.File{
			{
				Name: "templates/cm.yaml",
				Data: []byte("apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: test-cm-{{ .Release.Name }}\ndata:\n  key: value\n"),
			},
		},
	}
}

func deployParam(ns, release, repo, chartName, ver, values string) helmmodel.DeployParam {
	return helmmodel.DeployParam{
		Namespace: ns,
		Release:   release,
		RepoName:  repo,
		ChartName: chartName,
		Version:   ver,
		Values:    values,
	}
}

func installReadonlyRelease(op helmop.Operation, cg *chartgetter.FakeChartGetter, ns string) string {
	releaseName := fmt.Sprintf("readonly-rel-%d", time.Now().UnixNano())
	chartToInstall := minimalChartForEnvtest("mychart", "1.0.0")
	cg.WithChart("repo1", "mychart", "1.0.0", chartToInstall)

	params := deployParam(ns, releaseName, "repo1", "mychart", "1.0.0", "replicaCount: 1")
	resp, status := op.Install(params)
	Expect(status).To(Equal(202))
	Expect(resp.Msg).To(Equal("helm chart installing"))

	return releaseName
}

var _ = Describe("Operation Envtest 集成测试", func() {

	Context("查询类", func() {
		var (
			op helmop.Operation
			cg *chartgetter.FakeChartGetter
			ns string
		)

		BeforeEach(func() {
			ns = "default"
			cg = chartgetter.New()

			var err error
			op, err = newOpForTest(cg)
			Expect(err).NotTo(HaveOccurred())
		})

		Context("【应用运维-扩展详情-001】详情基础信息", func() {
			var releaseName string

			BeforeEach(func() {
				releaseName = installReadonlyRelease(op, cg, ns)
			})

			It("当 release 存在时，应返回 release 详情", func() {
				resp, status := op.GetReleaseInfo(ns, releaseName)
				Expect(status).To(Equal(200))
				Expect(resp.Code).To(Equal(int32(constant.Success)))

				b, _ := json.Marshal(resp.Data)
				Expect(string(b)).To(ContainSubstring(releaseName))
				Expect(string(b)).To(ContainSubstring(ns))
				Expect(string(b)).To(ContainSubstring("deployed"))
			})
		})

		Context("【应用运维-扩展详情-006】YAML/Values 展示", func() {
			var releaseName string

			BeforeEach(func() {
				releaseName = installReadonlyRelease(op, cg, ns)
			})

			It("当 release 存在时，应返回 manifest 字符串", func() {
				resp, status := op.GetManifest(ns, releaseName)
				Expect(status).To(Equal(200))
				Expect(resp.Code).To(Equal(int32(constant.Success)))

				manifest, ok := resp.Data.(string)
				Expect(ok).To(BeTrue())
				Expect(manifest).To(ContainSubstring("apiVersion: v1"))
			})

			It("当 release 存在时，应返回 values 对象", func() {
				resp, status := op.GetValues(ns, releaseName)
				Expect(status).To(Equal(200))
				Expect(resp.Code).To(Equal(int32(constant.Success)))

				vals, ok := resp.Data.(map[string]any)
				Expect(ok).To(BeTrue())
				Expect(vals).To(HaveKeyWithValue("replicaCount", float64(1)))
			})
		})

		Context("【应用运维-扩展组件管理概览-001】扩展组件列表", func() {
			var (
				extRelease string
				appRelease string
			)

			BeforeEach(func() {
				// 安装扩展组件：chart keywords 包含 FuyaoExtensionKeyword
				extRelease = fmt.Sprintf("ext-rel-%d", time.Now().UnixNano())
				extChart := minimalChartForEnvtest("extchart", "1.0.0")
				extChart.Metadata.Keywords = []string{constant.FuyaoExtensionKeyword}
				cg.WithChart("repo1", "extchart", "1.0.0", extChart)
				op.Install(deployParam(ns, extRelease, "repo1", "extchart", "1.0.0", ""))

				// 安装普通应用：不包含 extension/component 关键字
				appRelease = fmt.Sprintf("app-rel-%d", time.Now().UnixNano())
				appChart := minimalChartForEnvtest("appchart", "1.0.0")
				cg.WithChart("repo1", "appchart", "1.0.0", appChart)
				op.Install(deployParam(ns, appRelease, "repo1", "appchart", "1.0.0", ""))
			})

			It("当 extension=true 时，应只返回扩展组件列表（data.items）", func() {
				resp, status := op.GetReleases(param.ReleaseParameter{Namespace: ns, Extension: "true"})
				Expect(status).To(Equal(200))
				Expect(resp.Code).To(Equal(int32(constant.Success)))

				listResp, ok := resp.Data.(*helmmodel.ListResponse)
				Expect(ok).To(BeTrue(), "resp.Data should be *models.helm.ListResponse")
				Expect(listResp.Items).NotTo(BeEmpty())

				b, _ := json.Marshal(listResp.Items)
				itemsStr := string(b)
				Expect(itemsStr).To(ContainSubstring(extRelease))
				Expect(itemsStr).NotTo(ContainSubstring(appRelease))
			})
		})
	})

	Context("生命周期 (Install, Upgrade, Rollback, Uninstall)", func() {
		var (
			op helmop.Operation
			cg *chartgetter.FakeChartGetter
			ns string
		)

		BeforeEach(func() {
			ns = "default"
			cg = chartgetter.New()
			var err error
			op, err = newOpForTest(cg)
			Expect(err).NotTo(HaveOccurred())
		})

		Context("【应用创建-应用市场HELM-001】部署应用", func() {
			It("当调用 Install 时，应成功并返回 202 Accepted", func() {
				releaseName := "lifecycle-install"
				chart := minimalChartForEnvtest("mychart", "1.0.0")
				cg.WithChart("repo1", "mychart", "1.0.0", chart)
				params := deployParam(ns, releaseName, "repo1", "mychart", "1.0.0", "")

				resp, status := op.Install(params)

				Expect(status).To(Equal(202))
				Expect(resp.Msg).To(Equal("helm chart installing"))
			})

			It("当 Chart 获取失败时，Install 应返回 500 错误", func() {
				releaseName := "install-chart-fail"
				cg.WithError(fmt.Errorf("mock get chart error"))
				params := deployParam(ns, releaseName, "repo1", "fail-chart", "1.0.0", "")

				resp, status := op.Install(params)

				Expect(status).To(Equal(500))
				Expect(resp.Code).To(Equal(int32(constant.ServerError)))
				// Current behavior: chart getter failures are mapped to a generic upstream busy message
				Expect(resp.Msg).To(Equal("remote server busy"))
				// Ensure the fake getter path was executed
				Expect(cg.Last.Repo).To(Equal("repo1"))
				Expect(cg.Last.Chart).To(Equal("fail-chart"))
				Expect(cg.Last.Version).To(Equal("1.0.0"))
			})
		})

		Context("【应用运维-扩展升级回退-002】升级扩展组件", func() {
			It("当 release 已存在时调用 Upgrade，应成功并返回 202 Accepted", func() {
				releaseName := "lifecycle-upgrade"
				chartV1 := minimalChartForEnvtest("mychart", "1.0.0")
				cg.WithChart("repo1", "mychart", "1.0.0", chartV1)
				op.Install(deployParam(ns, releaseName, "repo1", "mychart", "1.0.0", ""))

				chartV2 := minimalChartForEnvtest("mychart", "1.1.0")
				cg.WithChart("repo1", "mychart", "1.1.0", chartV2)
				params := deployParam(ns, releaseName, "repo1", "mychart", "1.1.0", "replicaCount: 2")

				resp, status := op.Upgrade(params)

				Expect(status).To(Equal(202))
				Expect(resp.Msg).To(Equal("helm chart upgrading"))
			})
		})

		Context("【应用运维-扩展升级回退-013】回退扩展组件", func() {
			It("当存在历史版本时调用 Rollback，应成功并返回 200 OK", func() {
				releaseName := "lifecycle-rollback"
				cg.WithChart("repo1", "mychart", "1.0.0", minimalChartForEnvtest("mychart", "1.0.0"))
				cg.WithChart("repo1", "mychart", "1.1.0", minimalChartForEnvtest("mychart", "1.1.0"))
				op.Install(deployParam(ns, releaseName, "repo1", "mychart", "1.0.0", ""))
				op.Upgrade(deployParam(ns, releaseName, "repo1", "mychart", "1.1.0", ""))

				resp, status := op.RollbackRelease(ns, releaseName, "mychart", 1, 10)

				Expect(status).To(Equal(200))
				Expect(resp.Code).To(Equal(int32(constant.Success)))
			})
		})

		Context("【应用运维-扩展卸载-002】卸载扩展组件", func() {
			It("当 release 存在时调用 Uninstall，应成功并返回 200 OK", func() {
				releaseName := "lifecycle-uninstall"
				cg.WithChart("repo1", "mychart", "1.0.0", minimalChartForEnvtest("mychart", "1.0.0"))
				op.Install(deployParam(ns, releaseName, "repo1", "mychart", "1.0.0", ""))

				resp, status := op.Uninstall(ns, releaseName)

				Expect(status).To(Equal(200))
				Expect(resp.Code).To(Equal(int32(constant.Success)))
			})
		})
	})
})
