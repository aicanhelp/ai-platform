/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package utils

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestSafeJoinAllowsRootDestination(t *testing.T) {
	got, err := safeJoin("/", "etc/")
	if err != nil {
		t.Fatalf("safeJoin(/, etc/) unexpected error: %v", err)
	}
	if got != "/etc" {
		t.Fatalf("safeJoin(/, etc/) = %q, want /etc", got)
	}

	got, err = safeJoin("/", "etc/containerd/config.toml")
	if err != nil {
		t.Fatalf("safeJoin nested path unexpected error: %v", err)
	}
	if got != "/etc/containerd/config.toml" {
		t.Fatalf("got %q", got)
	}
}

func TestSafeJoinRejectsTraversal(t *testing.T) {
	base := t.TempDir()

	cases := []string{
		"../outside",
		"..",
		filepath.Join("ok", "..", "..", "outside"),
	}
	if runtime.GOOS != "windows" {
		cases = append(cases, "/etc/passwd")
	}

	for _, name := range cases {
		if _, err := safeJoin(base, name); err == nil {
			t.Fatalf("safeJoin(%q, %q) should reject traversal", base, name)
		}
	}
}

func TestUnTarToRootStylePaths(t *testing.T) {
	dir := t.TempDir()
	archive := filepath.Join(dir, "pkg.tar")
	dst := filepath.Join(dir, "root")
	if err := os.MkdirAll(dst, 0o755); err != nil {
		t.Fatal(err)
	}

	f, err := os.Create(archive)
	if err != nil {
		t.Fatal(err)
	}
	tw := tar.NewWriter(f)
	for _, entry := range []struct {
		name string
		dir  bool
		body string
	}{
		{name: "etc/", dir: true},
		{name: "etc/containerd/", dir: true},
		{name: "etc/containerd/config.toml", body: "version = 2\n"},
		{name: "usr/local/bin/containerd", body: "#!/bin/sh\n"},
	} {
		hdr := &tar.Header{Name: entry.name, Mode: 0o644}
		if entry.dir {
			hdr.Typeflag = tar.TypeDir
			hdr.Mode = 0o755
		} else {
			hdr.Typeflag = tar.TypeReg
			hdr.Size = int64(len(entry.body))
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatal(err)
		}
		if !entry.dir {
			if _, err := tw.Write([]byte(entry.body)); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	if err := UnTar(archive, dst); err != nil {
		t.Fatalf("UnTar failed: %v", err)
	}
	cfg := filepath.Join(dst, "etc", "containerd", "config.toml")
	data, err := os.ReadFile(cfg)
	if err != nil {
		t.Fatalf("read extracted config: %v", err)
	}
	if string(data) != "version = 2\n" {
		t.Fatalf("unexpected content: %q", data)
	}
}

func TestTarRoundTripRelative(t *testing.T) {
	src := t.TempDir()
	sub := filepath.Join(src, "subdir")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "hello.txt"), []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sub, "nested.txt"), []byte("nested"), 0o644); err != nil {
		t.Fatal(err)
	}

	work := t.TempDir()
	archive := filepath.Join(work, "bundle.tar.gz")
	dst := filepath.Join(work, "out")
	if err := os.MkdirAll(dst, 0o755); err != nil {
		t.Fatal(err)
	}

	if err := Tar(src, archive, false); err != nil {
		t.Fatalf("Tar failed: %v", err)
	}

	count, err := UnTarWithCount(archive, dst)
	if err != nil {
		t.Fatalf("UnTarWithCount failed: %v", err)
	}
	if count == 0 {
		t.Fatal("expected extracted entries")
	}

	hello, err := os.ReadFile(filepath.Join(dst, "hello.txt"))
	if err != nil {
		t.Fatalf("read hello.txt: %v", err)
	}
	if string(hello) != "hello" {
		t.Fatalf("hello.txt = %q", hello)
	}

	nested, err := os.ReadFile(filepath.Join(sub, "nested.txt"))
	if err != nil {
		t.Fatalf("source nested.txt should still exist: %v", err)
	}
	_ = nested

	extractedNested, err := os.ReadFile(filepath.Join(dst, "subdir", "nested.txt"))
	if err != nil {
		t.Fatalf("read nested.txt: %v", err)
	}
	if string(extractedNested) != "nested" {
		t.Fatalf("nested.txt = %q", extractedNested)
	}
}

func TestTarAbsolutePaths(t *testing.T) {
	src := t.TempDir()
	filePath := filepath.Join(src, "abs.txt")
	if err := os.WriteFile(filePath, []byte("abs"), 0o644); err != nil {
		t.Fatal(err)
	}

	work := t.TempDir()
	archive := filepath.Join(work, "abs.tar.gz")

	if err := Tar(src, archive, true); err != nil {
		t.Fatalf("Tar absolute failed: %v", err)
	}

	info, err := os.Stat(archive)
	if err != nil {
		t.Fatalf("stat archive: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("expected non-empty absolute tar archive")
	}
}

func writeTarGzArchive(t *testing.T, entries []struct {
	name string
	body string
	dir  bool
}) []byte {
	t.Helper()
	var raw bytes.Buffer
	tw := tar.NewWriter(&raw)
	for _, entry := range entries {
		hdr := &tar.Header{Name: entry.name, Mode: 0o644}
		if entry.dir {
			hdr.Typeflag = tar.TypeDir
			hdr.Mode = 0o755
		} else {
			hdr.Typeflag = tar.TypeReg
			hdr.Size = int64(len(entry.body))
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatal(err)
		}
		if !entry.dir {
			if _, err := tw.Write([]byte(entry.body)); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}

	var gz bytes.Buffer
	gw := gzip.NewWriter(&gz)
	if _, err := gw.Write(raw.Bytes()); err != nil {
		t.Fatal(err)
	}
	if err := gw.Close(); err != nil {
		t.Fatal(err)
	}
	return gz.Bytes()
}

func TestUnTarGzipArchive(t *testing.T) {
	data := writeTarGzArchive(t, []struct {
		name string
		body string
		dir  bool
	}{
		{name: "data.txt", body: "gz-content"},
	})

	work := t.TempDir()
	archive := filepath.Join(work, "layer.tar.gz")
	dst := filepath.Join(work, "out")
	if err := os.WriteFile(archive, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dst, 0o755); err != nil {
		t.Fatal(err)
	}

	if err := UnTar(archive, dst); err != nil {
		t.Fatalf("UnTar gzip archive failed: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(dst, "data.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "gz-content" {
		t.Fatalf("data.txt = %q", got)
	}
}
