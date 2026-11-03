package utils

import (
	"bytes"
	"fmt"
	"io"
	"strings"

	"gopkg.in/yaml.v3"
)

func normalizeFixtureBytes(data []byte) []byte {
	s := string(data)
	// Legacy KServe PD prefill templates duplicate container ports; collapse for strict YAML parsers.
	const dupPorts = "          ports:\n            - name: http\n              containerPort: 8000\n              protocol: TCP\n"
	if strings.Count(s, "          ports:\n") >= 2 && strings.Contains(s, dupPorts) {
		s = strings.Replace(s, dupPorts, "", 1)
	}
	return []byte(s)
}

func parseYAMLDocuments(data []byte) ([]map[string]interface{}, error) {
	data = normalizeFixtureBytes(data)
	dec := yaml.NewDecoder(bytes.NewReader(data))
	var docs []map[string]interface{}
	for {
		var doc map[string]interface{}
		if err := dec.Decode(&doc); err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		if doc == nil {
			continue
		}
		docs = append(docs, doc)
	}
	return docs, nil
}

func marshalYAMLDocuments(docs []map[string]interface{}) ([]byte, error) {
	var buf bytes.Buffer
	for i, doc := range docs {
		if i > 0 {
			buf.WriteString("---\n")
		}
		enc := yaml.NewEncoder(&buf)
		enc.SetIndent(2)
		if err := enc.Encode(doc); err != nil {
			return nil, err
		}
		if err := enc.Close(); err != nil {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}

func docKind(doc map[string]interface{}) string {
	k, _ := doc["kind"].(string)
	return k
}

func findDocIndex(docs []map[string]interface{}, kind string) int {
	for i, doc := range docs {
		if docKind(doc) == kind {
			return i
		}
	}
	return -1
}

func mustMap(v interface{}) map[string]interface{} {
	if v == nil {
		return map[string]interface{}{}
	}
	m, ok := v.(map[string]interface{})
	if !ok {
		return map[string]interface{}{}
	}
	return m
}

func deepMerge(dst, src map[string]interface{}) {
	for k, v := range src {
		if vMap, ok := v.(map[string]interface{}); ok {
			existing := mustMap(dst[k])
			deepMerge(existing, vMap)
			dst[k] = existing
			continue
		}
		dst[k] = v
	}
}

func mergeYAMLPatch(doc map[string]interface{}, patchYAML string) error {
	var patch map[string]interface{}
	if err := yaml.Unmarshal([]byte(patchYAML), &patch); err != nil {
		return err
	}
	deepMerge(doc, patch)
	return nil
}

func deleteMapPath(doc map[string]interface{}, path ...string) {
	if len(path) == 0 {
		return
	}
	cur := doc
	for i := 0; i < len(path)-1; i++ {
		next, ok := cur[path[i]].(map[string]interface{})
		if !ok {
			return
		}
		cur = next
	}
	delete(cur, path[len(path)-1])
}

func insvcDoc(docs []map[string]interface{}) (map[string]interface{}, error) {
	idx := findDocIndex(docs, "InferNexService")
	if idx < 0 {
		return nil, fmt.Errorf("InferNexService document not found")
	}
	return docs[idx], nil
}

func insvcConfigDoc(docs []map[string]interface{}) (map[string]interface{}, error) {
	idx := findDocIndex(docs, "InferNexServiceConfig")
	if idx < 0 {
		return nil, fmt.Errorf("InferNexServiceConfig document not found")
	}
	return docs[idx], nil
}

func llmisvcDoc(docs []map[string]interface{}) (map[string]interface{}, error) {
	idx := findDocIndex(docs, "LLMInferenceService")
	if idx < 0 {
		return nil, fmt.Errorf("LLMInferenceService document not found")
	}
	return docs[idx], nil
}

func llmisvcConfigDoc(docs []map[string]interface{}) (map[string]interface{}, error) {
	idx := findDocIndex(docs, "LLMInferenceServiceConfig")
	if idx < 0 {
		return nil, fmt.Errorf("LLMInferenceServiceConfig document not found")
	}
	return docs[idx], nil
}

func dropDocsExceptKinds(docs []map[string]interface{}, kinds ...string) []map[string]interface{} {
	allowed := map[string]struct{}{}
	for _, k := range kinds {
		allowed[k] = struct{}{}
	}
	out := make([]map[string]interface{}, 0, len(kinds))
	for _, doc := range docs {
		if _, ok := allowed[docKind(doc)]; ok {
			out = append(out, doc)
		}
	}
	return out
}

func prependDocs(docs, prefix []map[string]interface{}) []map[string]interface{} {
	out := make([]map[string]interface{}, 0, len(prefix)+len(docs))
	out = append(out, prefix...)
	out = append(out, docs...)
	return out
}

func yamlDocumentsEqual(a, b []byte) (bool, string, error) {
	docsA, err := parseYAMLDocuments(a)
	if err != nil {
		return false, "", fmt.Errorf("parse built: %w", err)
	}
	docsB, err := parseYAMLDocuments(b)
	if err != nil {
		return false, "", fmt.Errorf("parse legacy: %w", err)
	}
	if len(docsA) != len(docsB) {
		return false, fmt.Sprintf("document count mismatch: built=%d legacy=%d", len(docsA), len(docsB)), nil
	}
	for i := range docsA {
		eq, diff := mapsEqual(docsA[i], docsB[i])
		if !eq {
			return false, fmt.Sprintf("doc %d kind=%s: %s", i, docKind(docsA[i]), diff), nil
		}
	}
	return true, "", nil
}

func mapsEqual(a, b map[string]interface{}) (bool, string) {
	ab, err := yaml.Marshal(a)
	if err != nil {
		return false, err.Error()
	}
	bb, err := yaml.Marshal(b)
	if err != nil {
		return false, err.Error()
	}
	if string(ab) == string(bb) {
		return true, ""
	}
	return false, strings.TrimSpace(firstDiffLine(string(ab), string(bb)))
}

func firstDiffLine(a, b string) string {
	la := strings.Split(a, "\n")
	lb := strings.Split(b, "\n")
	n := len(la)
	if len(lb) < n {
		n = len(lb)
	}
	for i := 0; i < n; i++ {
		if la[i] != lb[i] {
			return fmt.Sprintf("line %d built=%q legacy=%q", i+1, la[i], lb[i])
		}
	}
	if len(la) != len(lb) {
		return fmt.Sprintf("line count built=%d legacy=%d", len(la), len(lb))
	}
	return "unknown diff"
}
