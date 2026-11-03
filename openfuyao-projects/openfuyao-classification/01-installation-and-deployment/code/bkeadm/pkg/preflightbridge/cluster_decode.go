/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package preflightbridge

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	kyaml "k8s.io/apimachinery/pkg/util/yaml"
	kjson "sigs.k8s.io/json"
	yaml2 "sigs.k8s.io/yaml"
)

type inputFieldError struct {
	path string
	err  error
}

func (e *inputFieldError) Error() string { return e.err.Error() }
func (e *inputFieldError) Unwrap() error { return e.err }

func inputErrorField(fallback string, err error) string {
	var fieldErr *inputFieldError
	if errors.As(err, &fieldErr) && fieldErr != nil && fieldErr.path != "" {
		return fieldErr.path
	}
	return fallback
}

func decodeYAMLStrict(raw []byte, target any, root string) error {
	var document any
	if err := yaml2.UnmarshalStrict(raw, &document); err != nil {
		return err
	}
	jsonRaw, err := kyaml.ToJSON(raw)
	if err != nil {
		return err
	}
	strictErrors, err := kjson.UnmarshalStrict(jsonRaw, target)
	if err != nil {
		return addInputField(root, decodeErrorField(err), err)
	}
	if len(strictErrors) != 0 {
		return addInputField(root, decodeErrorField(strictErrors[0]), strictErrors[0])
	}
	return nil
}

func addInputField(root, field string, err error) error {
	if field == "" {
		return err
	}
	return &inputFieldError{path: joinInputField(root, field), err: err}
}

func decodeErrorField(err error) string {
	var fieldErr kjson.FieldError
	if errors.As(err, &fieldErr) {
		return fieldErr.FieldPath()
	}
	const marker = "Go struct field "
	message := err.Error()
	start := strings.Index(message, marker)
	if start < 0 {
		return ""
	}
	field := message[start+len(marker):]
	if end := strings.Index(field, " of type "); end >= 0 {
		field = field[:end]
	}
	if dot := strings.IndexByte(field, '.'); dot >= 0 {
		return field[dot+1:]
	}
	return ""
}

func joinInputField(root, field string) string {
	if root == "nodes" && strings.HasPrefix(field, "items[") {
		return root + strings.TrimPrefix(field, "items")
	}
	if strings.HasPrefix(field, "[") {
		return root + field
	}
	return root + "." + field
}

func decodeBKENodeDocuments(raw []byte) ([]configv1beta1.BKENode, error) {
	documents, err := splitYAMLDocuments(raw)
	if err != nil {
		return nil, err
	}
	if len(documents) == 0 {
		return nil, errors.New("no BKENode resources found in the file")
	}
	if len(documents) == 1 {
		return decodeBKENodeDocument(documents[0])
	}
	nodes := make([]configv1beta1.BKENode, 0, len(documents))
	for index, document := range documents {
		var node configv1beta1.BKENode
		root := fmt.Sprintf("nodes[%d]", index)
		if err := decodeYAMLStrict(document, &node, root); err != nil {
			return nil, err
		}
		nodes = append(nodes, node)
	}
	return nodes, nil
}

func splitYAMLDocuments(raw []byte) ([][]byte, error) {
	reader := kyaml.NewYAMLReader(bufio.NewReader(bytes.NewReader(raw)))
	var documents [][]byte
	for {
		document, err := reader.Read()
		if err == io.EOF {
			return documents, nil
		}
		if err != nil {
			return nil, fmt.Errorf("read node YAML document: %w", err)
		}
		if len(bytes.TrimSpace(document)) != 0 {
			documents = append(documents, document)
		}
	}
}

func decodeBKENodeDocument(raw []byte) ([]configv1beta1.BKENode, error) {
	kind, err := nodeDocumentKind(raw)
	if err != nil {
		return nil, err
	}
	switch kind {
	case "list":
		var list bkeNodeListDocument
		if err := decodeYAMLStrict(raw, &list, "nodes"); err != nil {
			return nil, err
		}
		if len(list.Items) == 0 {
			return nil, errors.New("nodes.items must contain at least one BKENode")
		}
		return list.Items, nil
	case "slice":
		var nodes []configv1beta1.BKENode
		if err := decodeYAMLStrict(raw, &nodes, "nodes"); err != nil {
			return nil, err
		}
		if len(nodes) == 0 {
			return nil, errors.New("nodes must contain at least one BKENode")
		}
		return nodes, nil
	default:
		var node configv1beta1.BKENode
		if err := decodeYAMLStrict(raw, &node, "nodes[0]"); err != nil {
			return nil, err
		}
		return []configv1beta1.BKENode{node}, nil
	}
}

func nodeDocumentKind(raw []byte) (string, error) {
	jsonRaw, err := kyaml.ToJSON(raw)
	if err != nil {
		return "", err
	}
	var document any
	if err := json.Unmarshal(jsonRaw, &document); err != nil {
		return "", err
	}
	switch typed := document.(type) {
	case []any:
		return "slice", nil
	case map[string]any:
		if _, ok := typed["items"]; ok {
			return "list", nil
		}
		return "node", nil
	default:
		return "", errors.New("nodes YAML must contain an object or list")
	}
}
