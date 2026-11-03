package console

import (
	"encoding/json"
	"io"
	"net/http"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Node 资源接口测试", Label("with-workload-cluster"), func() {
	Context("Node 查询", func() {
		It("查询成功", func() {
			resp, err := makeRequest("GET", "/api/v1/nodes", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

			var nodeList struct {
				Items []map[string]interface{} `json:"items"`
			}
			err = json.Unmarshal(body, &nodeList)
			Expect(err).NotTo(HaveOccurred(), string(body))
			Expect(len(nodeList.Items)).To(BeNumerically(">", 0))
		})
	})
})
