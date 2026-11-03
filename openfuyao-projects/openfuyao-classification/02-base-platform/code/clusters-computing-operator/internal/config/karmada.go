package config

import (
	"bytes"
	"fmt"
	karmadav1alpha1 "github.com/karmada-io/karmada/operator/pkg/apis/operator/v1alpha1"
	"html/template"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	clusterv1alpha1 "openfuyao.com/clusters-computing-operator/api/v1alpha1"
	"os"
)

type TemplateData struct {
	Namespace string
	Karmada   *clusterv1alpha1.KarmadaDeployment
}

func LoadKarmadaConfig(templatePath string, tc *clusterv1alpha1.TenantCluster) (*karmadav1alpha1.Karmada, error) {
	templateContent, err := os.ReadFile(templatePath)
	if err != nil {
		return nil, fmt.Errorf("read config file error:%w", err)
	}

	// Prepare template data
	data := TemplateData{
		Namespace: tc.Namespace,
		Karmada:   &tc.Spec.ControlPlane.Karmada,
	}

	// Parse the template
	tmpl, err := template.New("karmada").Parse(string(templateContent))
	if err != nil {
		return nil, fmt.Errorf("failed to parse template: %w", err)
	}

	// Execute the template
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return nil, fmt.Errorf("failed to execute template: %w", err)
	}

	//  Create a new scheme and add the Karmada API
	s := runtime.NewScheme()
	karmadav1alpha1.AddToScheme(s)

	// Create a decoder
	decoder := serializer.NewCodecFactory(s).UniversalDeserializer()

	// Decode the YAML file
	obj, _, err := decoder.Decode(buf.Bytes(), nil, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decode: %w", err)
	}

	karmada, ok := obj.(*karmadav1alpha1.Karmada)
	if !ok {
		return nil, fmt.Errorf("decoded object is not a Karmada")
	}
	return karmada, nil
}
