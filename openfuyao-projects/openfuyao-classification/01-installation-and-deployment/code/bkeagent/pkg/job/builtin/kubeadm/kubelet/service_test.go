/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package kubelet

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	confv1beta1 "gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"
)

// TestServiceGenerator_GenerateService_NilConfig 测试异常场景：传入 nil 配置，验证返回错误
func TestServiceGenerator_GenerateService_NilConfig(t *testing.T) {
	// 1. 准备临时目录
	tempDir := t.TempDir()
	generator := NewServiceData(tempDir)

	// 2. 传入 nil 配置，执行生成
	err := generator.GenerateService(nil, nil, nil)
	if err == nil {
		t.Fatal("GenerateService 传入 nil 配置时未返回错误，不符合预期")
	}

	// 3. 验证错误信息
	expectedErrMsg := "service config is nil"
	if !strings.Contains(err.Error(), expectedErrMsg) {
		t.Errorf("错误信息不符合预期：期望包含 %q，实际为 %q", expectedErrMsg, err.Error())
	}

	// 4. 验证文件未创建
	servicePath := filepath.Join(tempDir, KubeletFileName)
	_, err = os.Stat(servicePath)
	if !os.IsNotExist(err) {
		t.Errorf("传入 nil 配置时，不应创建文件 %s，但文件已存在", servicePath)
	}
}

// TestServiceGenerator_GenerateService_PartialConfig 测试边界场景：只传入部分字段，验证未配置字段不渲染
func TestServiceGenerator_GenerateService_PartialConfig(t *testing.T) {
	// 1. 准备临时目录
	tempDir := t.TempDir()
	generator := NewServiceData(tempDir)

	// 2. 构造部分配置（仅必填字段 + 部分可选字段）
	testConfig := &confv1beta1.KubeletService{
		ExecStart: "/usr/bin/kubelet --config=/etc/kubernetes/kubelet.conf", // 必填
		Restart:   "on-failure",                                             // 可选
		User:      "root",                                                   // 可选
		// 其他字段（如 ExecStartPre、Environment）未配置
	}

	// 3. 执行生成
	err := generator.GenerateService(testConfig, nil, nil)
	if err != nil {
		t.Fatalf("GenerateService 失败：%v", err)
	}

	// 4. 验证内容：已配置字段存在，未配置字段不存在
	servicePath := filepath.Join(tempDir, KubeletFileName)
	content, err := os.ReadFile(servicePath)
	if err != nil {
		t.Fatalf("读取文件失败：%v", err)
	}
	contentStr := string(content)

	// 已配置字段应存在
	expectedPresent := []string{
		"ExecStart=/usr/bin/kubelet --config=/etc/kubernetes/kubelet.conf",
		"Restart=on-failure",
		"User=root",
		"WantedBy=multi-user.target",
	}
	for _, field := range expectedPresent {
		if !strings.Contains(contentStr, field) {
			t.Errorf("已配置字段缺失：%q", field)
		}
	}

	// 未配置字段应不存在
	expectedAbsent := []string{
		"ExecStartPre=",
		"Environment=",
		"EnvironmentFile=",
		//"CustomExtra",
		"KillMode=",
	}
	for _, field := range expectedAbsent {
		if strings.Contains(contentStr, field) {
			t.Errorf("未配置字段不应存在：%q", field)
		}
	}
}

// TestServiceGenerator_generateServiceContent_TemplateRender 测试模板渲染逻辑（单独验证 content 生成）
func TestServiceGenerator_generateServiceContent_TemplateRender(t *testing.T) {
	// 1. 构造测试配置
	testConfig := &confv1beta1.KubeletService{
		ExecStart:   "/usr/bin/kubelet --node-ip=192.168.100.48",
		Environment: []string{"TEST_KEY=test_value"},
		CustomExtra: map[string]string{"TestField": "testValue"},
	}

	// 2. 创建 generator（临时目录无关，仅测试 generateServiceContent）
	generator := NewServiceData("")

	// 3. 调用生成 content 方法
	content, err := generator.generateServiceContent(testConfig)
	if err != nil {
		t.Fatalf("generateServiceContent 失败：%v", err)
	}

	// 4. 验证渲染结果
	expectedLines := []string{
		"ExecStart=/usr/bin/kubelet --node-ip=192.168.100.48",
		`Environment="TEST_KEY=test_value"`,
		"TestField=testValue",
		"[Install]",
		"WantedBy=multi-user.target",
	}
	for _, line := range expectedLines {
		if !strings.Contains(content, line) {
			t.Errorf("渲染内容缺失：%q，完整内容：\n%s", line, content)
		}
	}
}
