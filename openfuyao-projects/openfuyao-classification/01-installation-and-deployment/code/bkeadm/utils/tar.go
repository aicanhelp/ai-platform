/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
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
	"bufio"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

// Tar 将 src 目录打包成 tar.gz 压缩包，dst 为压缩包的保存路径
func Tar(src, dst string, absolute bool) error {
	fw, err := createTarFile(dst)
	if err != nil {
		return err
	}
	defer fw.Close()

	gw, tw, err := createCompressedWriter(fw)
	if err != nil {
		return err
	}
	defer gw.Close()
	defer tw.Close()

	return packDirectory(src, absolute, tw)
}

func createTarFile(dst string) (*os.File, error) {
	fw, err := os.Create(dst)
	if err != nil {
		return nil, fmt.Errorf("failed to create tar file %s: %w", dst, err)
	}

	if err := fw.Chmod(DefaultFilePermission); err != nil {
		log.Warnf("failed to set file permission for %s: %v", dst, err)
	}

	return fw, nil
}

func createCompressedWriter(fw *os.File) (*gzip.Writer, *tar.Writer, error) {
	gw := gzip.NewWriter(fw)
	tw := tar.NewWriter(gw)
	return gw, tw, nil
}

func packDirectory(src string, absolute bool, tw *tar.Writer) error {
	return walkWithHandler(src,
		func(fileName string, fi os.FileInfo) error {
			return packFile(src, fileName, fi, absolute, tw)
		},
		defaultErrorHandler,
	)
}

func walkWithHandler(root string, fn func(string, os.FileInfo) error, errFn func(error) error) error {
	if errFn == nil {
		errFn = func(err error) error { return err }
	}

	return filepath.Walk(root, func(fileName string, fi os.FileInfo, err error) error {
		if err != nil {
			return errFn(err)
		}
		return fn(fileName, fi)
	})
}

func defaultErrorHandler(err error) error {
	return err
}

func packFile(src, fileName string, fi os.FileInfo, absolute bool, tw *tar.Writer) error {
	hdr, err := createTarHeader(src, fileName, fi, absolute)
	if err != nil {
		return err
	}

	if err := tw.WriteHeader(hdr); err != nil {
		return fmt.Errorf("failed to write tar header for %s: %w", fileName, err)
	}

	if fi.Mode().IsRegular() {
		if err := writeFileContent(fileName, tw); err != nil {
			return err
		}
	}

	log.Debugf("Packaged %s successfully\n", fileName)
	return nil
}

func createTarHeader(src, fileName string, fi os.FileInfo, absolute bool) (*tar.Header, error) {
	hdr, err := tar.FileInfoHeader(fi, "")
	if err != nil {
		return nil, fmt.Errorf("failed to create tar header for %s: %w", fileName, err)
	}

	if absolute {
		hdr.Name = filepath.ToSlash(fileName)
	} else {
		relPath, err := filepath.Rel(src, fileName)
		if err != nil {
			return nil, fmt.Errorf("failed to get relative path for %s: %w", fileName, err)
		}
		hdr.Name = filepath.ToSlash(relPath)
	}

	return hdr, nil
}

func writeFileContent(fileName string, tw *tar.Writer) error {
	fr, err := os.Open(fileName)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %w", fileName, err)
	}
	defer fr.Close()

	if _, err := io.Copy(tw, fr); err != nil {
		return fmt.Errorf("failed to copy file %s to tar: %w", fileName, err)
	}

	return nil
}

// UnTar extracts a tar archive to the specified destination directory
func UnTar(src, dst string) error {
	_, err := UnTarWithCount(src, dst)
	return err
}

// UnTarWithCount extracts a tar archive and returns the number of entries processed.
func UnTarWithCount(src, dst string) (int, error) {
	file, err := os.Open(src)
	if err != nil {
		return 0, fmt.Errorf("failed to open archive %s: %w", src, err)
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	stream, err := detectArchiveStream(reader)
	if err != nil {
		return 0, err
	}

	return extractTarStream(stream, dst)
}

func detectArchiveStream(reader *bufio.Reader) (io.Reader, error) {
	// gzip magic bytes: 0x1f 0x8b
	magic, err := reader.Peek(2)
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("failed to detect archive format: %w", err)
	}

	if len(magic) == 2 && magic[0] == 0x1f && magic[1] == 0x8b {
		gzReader, gzErr := gzip.NewReader(reader)
		if gzErr != nil {
			return nil, fmt.Errorf("failed to create gzip reader: %w", gzErr)
		}
		return gzReader, nil
	}

	return reader, nil
}

func extractTarStream(stream io.Reader, dst string) (int, error) {
	tarReader := tar.NewReader(stream)
	count := 0

	for {
		header, err := tarReader.Next()
		if errors.Is(err, io.EOF) {
			return count, nil
		}
		if err != nil {
			return count, fmt.Errorf("failed to read tar entry: %w", err)
		}

		targetPath, err := safeJoin(dst, header.Name)
		if err != nil {
			return count, err
		}

		if err := extractEntry(tarReader, header, targetPath, dst); err != nil {
			return count, err
		}
		count++
	}
}

func safeJoin(dst, name string) (string, error) {
	// Reject absolute entry names; filepath.Join would otherwise discard dst.
	if filepath.IsAbs(name) {
		return "", fmt.Errorf("invalid archive entry path: %s", name)
	}

	cleanDst := filepath.Clean(dst)
	cleanName := filepath.Clean(name)
	if cleanName == ".." || strings.HasPrefix(cleanName, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("invalid archive entry path: %s", name)
	}

	targetPath := filepath.Join(cleanDst, cleanName)
	rel, err := filepath.Rel(cleanDst, targetPath)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("invalid archive entry path: %s", name)
	}

	return targetPath, nil
}

func extractEntry(tarReader *tar.Reader, header *tar.Header, targetPath, dstRoot string) error {
	switch header.Typeflag {
	case tar.TypeDir:
		if err := os.MkdirAll(targetPath, os.FileMode(header.Mode)); err != nil {
			return fmt.Errorf("failed to create directory %s: %w", targetPath, err)
		}
	case tar.TypeReg, tar.TypeRegA:
		if err := os.MkdirAll(filepath.Dir(targetPath), DefaultDirPermission); err != nil {
			return fmt.Errorf("failed to create parent directory for %s: %w", targetPath, err)
		}
		if err := writeFileFromTar(tarReader, targetPath, os.FileMode(header.Mode)); err != nil {
			return err
		}
	case tar.TypeSymlink:
		if err := os.MkdirAll(filepath.Dir(targetPath), DefaultDirPermission); err != nil {
			return fmt.Errorf("failed to create parent directory for symlink %s: %w", targetPath, err)
		}
		if err := os.RemoveAll(targetPath); err != nil {
			return fmt.Errorf("failed to cleanup symlink target %s: %w", targetPath, err)
		}
		if err := os.Symlink(header.Linkname, targetPath); err != nil {
			return fmt.Errorf("failed to create symlink %s -> %s: %w", targetPath, header.Linkname, err)
		}
	case tar.TypeLink:
		if err := os.MkdirAll(filepath.Dir(targetPath), DefaultDirPermission); err != nil {
			return fmt.Errorf("failed to create parent directory for hard link %s: %w", targetPath, err)
		}
		linkTarget, err := safeJoin(dstRoot, header.Linkname)
		if err != nil {
			return err
		}
		if err := os.RemoveAll(targetPath); err != nil {
			return fmt.Errorf("failed to cleanup hard link target %s: %w", targetPath, err)
		}
		if err := os.Link(linkTarget, targetPath); err != nil {
			return fmt.Errorf("failed to create hard link %s -> %s: %w", targetPath, linkTarget, err)
		}
	default:
		// Keep behavior tolerant for non-critical entry types (e.g. pax headers).
		log.Debugf("Skip unsupported tar entry type %d for %s", header.Typeflag, header.Name)
	}

	return nil
}

func writeFileFromTar(tarReader *tar.Reader, targetPath string, mode os.FileMode) error {
	file, err := os.OpenFile(targetPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, mode)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", targetPath, err)
	}
	defer file.Close()

	if _, err = io.Copy(file, tarReader); err != nil {
		return fmt.Errorf("failed to write file %s: %w", targetPath, err)
	}

	return nil
}
