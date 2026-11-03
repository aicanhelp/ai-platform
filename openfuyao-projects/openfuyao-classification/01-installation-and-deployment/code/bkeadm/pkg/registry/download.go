/*
Copyright 2018 The CDI Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Original file: https://github.com/kubevirt/containerized-data-importer/blob/release-v1.34/pkg/importer/format-readers.go
*/

package registry

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/containers/image/v5/image"
	"github.com/containers/image/v5/pkg/blobinfocache"
	"github.com/containers/image/v5/transports/alltransports"
	"github.com/containers/image/v5/types"
	"github.com/klauspost/compress/zstd"
	"github.com/pkg/errors"
	"github.com/ulikunitz/xz"
	"k8s.io/klog/v2"

	"gopkg.openfuyao.cn/bkeadm/pkg/root"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

type OptionsDownload struct {
	root.Options
	SrcTLSVerify        bool   `json:"src-tls-verify"`
	Image               string `json:"image"`
	Username            string `json:"username"`
	Password            string `json:"password"`
	CertDir             string `json:"certDir"`
	DownloadToDir       string `json:"downloadToDir"`
	DownloadInImageFile string `json:"downloadInImageFile"`
}

func (od *OptionsDownload) Download() error {
	if err := od.ensureDownloadDir(); err != nil {
		return err
	}
	downloadFileMap := od.buildDownloadFileMap()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	imageSource, registrySrcCtx, err := od.setupImageSource(ctx)
	if err != nil {
		return err
	}
	defer closeImage(imageSource)

	if err := od.downloadFilesFromLayers(ctx, imageSource, registrySrcCtx, downloadFileMap); err != nil {
		return err
	}

	od.logDownloadResults(downloadFileMap)
	return nil
}

func (od *OptionsDownload) ensureDownloadDir() error {
	if !utils.Exists(od.DownloadToDir) {
		return os.MkdirAll(od.DownloadToDir, os.ModePerm)
	}
	return nil
}

func (od *OptionsDownload) buildDownloadFileMap() map[string]string {
	downloadFileMap := map[string]string{}
	for _, file := range strings.Split(od.DownloadInImageFile, ",") {
		f := file
		if len(f) == 0 {
			continue
		}
		if strings.HasPrefix(f, "/") {
			f = f[1:]
		}
		downloadFileMap[f] = ""
	}
	return downloadFileMap
}

func (od *OptionsDownload) setupImageSource(ctx context.Context) (types.ImageSource, *types.SystemContext, error) {
	imageName := od.Image
	if !strings.HasPrefix(od.Image, "docker://") {
		imageName = "docker://" + od.Image
	}
	ref, err := alltransports.ParseImageName(imageName)
	if err != nil {
		return nil, nil, err
	}

	registrySrcCtx := od.buildSystemContext()
	imageSource, err := ref.NewImageSource(ctx, registrySrcCtx)
	if err != nil {
		log.Errorf("Could not create image reference: %v", err)
		return nil, nil, errors.Wrap(err, "Could not create image reference")
	}
	return imageSource, registrySrcCtx, nil
}

func (od *OptionsDownload) buildSystemContext() *types.SystemContext {
	registrySrcCtx := &types.SystemContext{}
	if od.Username != "" && od.Password != "" {
		registrySrcCtx.DockerAuthConfig = &types.DockerAuthConfig{
			Username: od.Username,
			Password: od.Password,
		}
	}
	if od.CertDir != "" {
		registrySrcCtx.DockerCertPath = od.CertDir
		registrySrcCtx.DockerDaemonCertPath = od.CertDir
	}
	if od.SrcTLSVerify {
		registrySrcCtx.DockerDaemonInsecureSkipTLSVerify = true
		registrySrcCtx.DockerInsecureSkipTLSVerify = types.NewOptionalBool(true)
	}
	return registrySrcCtx
}

func (od *OptionsDownload) downloadFilesFromLayers(ctx context.Context, imageSource types.ImageSource,
	registrySrcCtx *types.SystemContext, downloadFileMap map[string]string) error {
	imgCloser, err := image.FromSource(ctx, registrySrcCtx, imageSource)
	if err != nil {
		klog.Errorf("Error retrieving image: %v", err)
		return errors.Wrap(err, "Error retrieving image")
	}
	defer imgCloser.Close()

	cache := blobinfocache.DefaultCache(registrySrcCtx)
	layers := imgCloser.LayerInfos()

	for _, layer := range reverseLayers(layers) {
		log.Debugf("Processing layer %+v", layer)
		config := &LayerProcessConfig{
			SystemContext:   registrySrcCtx,
			ImageSource:     imageSource,
			Layer:           layer,
			DownloadToDir:   od.DownloadToDir,
			DownloadFileMap: downloadFileMap,
			Cache:           cache,
		}
		err = processLayer(ctx, config)
		if found(downloadFileMap) {
			break
		}
		if err != nil {
			continue
		}
	}
	return nil
}

func (od *OptionsDownload) logDownloadResults(downloadFileMap map[string]string) {
	for k, v := range downloadFileMap {
		if v == "" {
			log.Errorf("%s Failed to find file in the container image", k)
		} else {
			log.Infof("Download complete %s", v)
		}
	}
}

func closeImage(src types.ImageSource) {
	if err := src.Close(); err != nil {
		log.Warnf("Could not close image source: %v ", err)
	}
}

// reverseLayers reverse layer
func reverseLayers(arr []types.BlobInfo) []types.BlobInfo {
	for i, j := 0, len(arr)-1; i < j; i, j = i+1, j-1 {
		arr[i], arr[j] = arr[j], arr[i]
	}
	return arr
}

// ------ transport ------
const (
	whFilePrefix = ".wh."
)

// LayerProcessConfig holds configuration for processing image layers
type LayerProcessConfig struct {
	SystemContext   *types.SystemContext
	ImageSource     types.ImageSource
	Layer           types.BlobInfo
	DownloadToDir   string
	DownloadFileMap map[string]string
	Cache           types.BlobInfoCache
}

func isWhiteout(path string) bool {
	return strings.HasPrefix(filepath.Base(path), whFilePrefix)
}

func isDir(hdr *tar.Header) bool {
	return hdr.Typeflag == tar.TypeDir
}

func processLayer(ctx context.Context, config *LayerProcessConfig) error {
	if config.DownloadFileMap == nil {
		return errors.New("downloadFileMap cannot be nil")
	}

	tarReader, fr, err := openLayerTarReader(ctx, config)
	if err != nil {
		return err
	}
	defer fr.Close()

	return extractFilesFromTar(tarReader, config)
}

func openLayerTarReader(ctx context.Context, config *LayerProcessConfig) (*tar.Reader, *FormatReaders, error) {
	reader, _, err := config.ImageSource.GetBlob(ctx, config.Layer, config.Cache)
	if err != nil {
		log.Errorf("Could not read layer: %v", err)
		return nil, nil, errors.New(fmt.Sprintf("Could not read layer: %v", err))
	}
	fr, err := NewFormatReaders(reader, 0)
	if err != nil {
		return nil, nil, errors.New(fmt.Sprintf("Could not read layer: %v", err))
	}
	return tar.NewReader(fr.TopReader()), fr, nil
}

func extractFilesFromTar(tarReader *tar.Reader, config *LayerProcessConfig) error {
	for {
		if found(config.DownloadFileMap) {
			return nil
		}
		hdr, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Errorf("Error reading layer: %v", err)
			return errors.New(fmt.Sprintf("Error reading layer: %v", err))
		}

		if err := processTargetFiles(tarReader, hdr, config); err != nil {
			return err
		}
	}
	return nil
}

func processTargetFiles(tarReader *tar.Reader, hdr *tar.Header, config *LayerProcessConfig) error {
	df := utils.CopyMap(config.DownloadFileMap)
	for k, v := range df {
		if len(v) > 0 {
			continue
		}
		if shouldExtractFile(hdr, k) {
			if err := extractFile(tarReader, hdr, k, config); err != nil {
				return err
			}
		}
	}
	return nil
}

func shouldExtractFile(hdr *tar.Header, targetFile string) bool {
	return strings.HasSuffix(hdr.Name, targetFile) && !isWhiteout(hdr.Name) && !isDir(hdr)
}

func extractFile(tarReader *tar.Reader, hdr *tar.Header, targetFile string, config *LayerProcessConfig) error {
	log.Debugf("Copying file: %s", hdr.Name)
	hdrPath := strings.Split(hdr.Name, "/")
	destFile := filepath.Join(config.DownloadToDir, hdrPath[len(hdrPath)-1])
	if err := os.Remove(destFile); err != nil && !os.IsNotExist(err) {
		log.Warnf("Failed to remove dest file: %v", err)
	}
	if err := streamDataToFile(tarReader, destFile); err != nil {
		log.Errorf("Error copying file: %v", err)
		return errors.New(fmt.Sprintf("Error copying file: %v", err))
	}
	config.DownloadFileMap[targetFile] = hdr.Name
	return nil
}

func found(m map[string]string) bool {
	for _, v := range m {
		if v == "" {
			return false
		}
	}
	return true
}

// streamDataToFile provides a function to stream the specified io.Reader to the specified local file
func streamDataToFile(r io.Reader, fileName string) error {
	outFile, err := os.OpenFile(fileName, os.O_CREATE|os.O_EXCL|os.O_WRONLY, os.ModePerm)
	if err != nil {
		return err
	}
	defer outFile.Close()
	log.Info("Writing data...")
	if _, err = io.Copy(outFile, r); err != nil {
		log.Errorf("Unable to write file from dataReader: %v", err)
		if rmErr := os.Remove(outFile.Name()); rmErr != nil && !os.IsNotExist(rmErr) {
			log.Warnf("Failed to remove output file: %v", rmErr)
		}
		return errors.New(fmt.Sprintf("Unable to write file from dataReader: %v", err))
	}
	err = outFile.Sync()
	return err
}

// ------ file-fmt ------

// MaxExpectedHdrSize defines the Size of buffer used to read file headers.
// Note: this is the size of tar's header. If a larger number is used the tar unarchive operation
//
//	creates the destination file too large, by the difference between this const and 512.
const MaxExpectedHdrSize = 512

// Headers provides a map for header info, key is file format, eg. "gz" or "tar", value is metadata describing the layout for this hdr
type Headers map[string]Header

var knownHeaders = Headers{
	"gz": Header{
		Format:      "gz",
		magicNumber: []byte{0x1F, 0x8B},
		// TODO: size not in hdr
		SizeOff: 0,
		SizeLen: 0,
	},
	"zst": Header{
		Format:      "zst",
		magicNumber: []byte{0x28, 0xb5, 0x2f, 0xfd},
		SizeOff:     0,
		SizeLen:     0,
	},
	"tar": Header{
		Format:      "tar",
		magicNumber: []byte{0x75, 0x73, 0x74, 0x61, 0x72},
		mgOffset:    0x101,
		SizeOff:     124,
		SizeLen:     8,
	},
	"xz": Header{
		Format:      "xz",
		magicNumber: []byte{0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00},
		// TODO: size not in hdr
		SizeOff: 0,
		SizeLen: 0,
	},
}

// Header represents our parameters for a file format header
type Header struct {
	Format      string
	magicNumber []byte
	mgOffset    int
	SizeOff     int // in bytes
	SizeLen     int // in bytes
}

// CopyKnownHdrs performs a simple map copy since := assignment copies the reference to the map, not contents.
func CopyKnownHdrs() Headers {
	m := make(Headers)
	for k, v := range knownHeaders {
		m[k] = v
	}
	return m
}

// Match performs a check to see if the provided byte slice matches the bytes in our header data
func (h Header) Match(b []byte) bool {
	return bytes.Equal(b[h.mgOffset:h.mgOffset+len(h.magicNumber)], h.magicNumber)
}

// ------ format-readers ------
type reader struct {
	rdrType int
	rdr     io.ReadCloser
}

// FormatReaders contains the stack of readers needed to get information from the input stream (io.ReadCloser)
type FormatReaders struct {
	readers     []reader
	buf         []byte // holds file headers
	Convert     bool
	Archived    bool
	ArchiveXz   bool
	ArchiveGz   bool
	ArchiveZstd bool
}

const (
	rdrGz = iota
	rdrMulti
	rdrXz
	rdrStream
)

// map scheme and format to rdrType
var rdrTypM = map[string]int{
	"gz":     rdrGz,
	"xz":     rdrXz,
	"stream": rdrStream,
}

// NewFormatReaders creates a new instance of FormatReaders using the input stream and content type passed in.
func NewFormatReaders(stream io.ReadCloser, total uint64) (*FormatReaders, error) {
	var err error
	readers := &FormatReaders{
		buf: make([]byte, MaxExpectedHdrSize),
	}
	err = readers.constructReaders(stream)
	return readers, err
}

func (fr *FormatReaders) constructReaders(r io.ReadCloser) error {
	fr.appendReader(rdrTypM["stream"], r)
	knownHdrs := CopyKnownHdrs() // need local copy since keys are removed
	log.Debug("constructReaders: checking compression and archive formats")
	for {
		hdr, err := fr.matchHeader(&knownHdrs)
		if err != nil {
			return errors.New(fmt.Sprintf("could not process image header: %v", err))
		}
		if hdr == nil {
			break // done processing headers, we have the orig source file
		}
		log.Debugf("constructReaders: found header of type %q\n", hdr.Format)
		// create format-specific reader and append it to dataStream readers stack
		fr.fileFormatSelector(hdr)
	}
	return nil
}

// Append to the receiver's reader stack the passed in reader. If the reader type is multi-reader
// then wrap a multi-reader around the passed in reader. If the reader is not a Closer then wrap a
// nop closer.
func (fr *FormatReaders) appendReader(rType int, x interface{}) {
	if x == nil {
		return
	}
	r, ok := x.(io.Reader)
	if !ok {
		log.Error("internal error: unexpected reader type passed to appendReader()")
		return
	}
	if rType == rdrMulti {
		r = io.MultiReader(r, fr.TopReader())
	}
	if _, ok := r.(io.Closer); !ok {
		r = io.NopCloser(r)
	}
	fr.readers = append(fr.readers, reader{rdrType: rType, rdr: r.(io.ReadCloser)})
}

// TopReader return the top-level io.ReadCloser from the receiver Reader "stack".
func (fr *FormatReaders) TopReader() io.ReadCloser {
	return fr.readers[len(fr.readers)-1].rdr
}

// Based on the passed in header, append the format-specific reader to the readers stack,
// and update the receiver Size field. Note: a bool is set in the receiver for qcow2 files.
func (fr *FormatReaders) fileFormatSelector(hdr *Header) {
	var r io.Reader
	var err error
	fFmt := hdr.Format
	switch fFmt {
	case "gz":
		r, err = fr.gzReader()
		if err == nil {
			fr.Archived = true
			fr.ArchiveGz = true
		}
	case "zst":
		r, err = fr.zstReader()
		if err == nil {
			fr.Archived = true
			fr.ArchiveZstd = true
		}
	case "xz":
		r, err = fr.xzReader()
		if err == nil {
			fr.Archived = true
			fr.ArchiveXz = true
		}
	default: // No special handling needed for other formats
	}
	if err == nil && r != nil {
		fr.appendReader(rdrTypM[fFmt], r)
	}
}

// Return the gz reader and the size of the endpoint "through the eye" of the previous reader.
// Assumes a single file was gzipped.
// NOTE: size in gz is stored in the last 4 bytes of the file. This probably requires the file
//
//	to be decompressed in order to get its original size. For now 0 is returned.
func (fr *FormatReaders) gzReader() (io.ReadCloser, error) {
	gz, err := gzip.NewReader(fr.TopReader())
	if err != nil {
		return nil, errors.New(fmt.Sprintf("could not create gzip reader: %v", err))
	}
	log.Debug("gz: reading gzipped file")
	return gz, nil
}

// Return the zst reader.
func (fr *FormatReaders) zstReader() (io.ReadCloser, error) {
	zst, err := zstd.NewReader(fr.TopReader())
	if err != nil {
		return nil, errors.New(fmt.Sprintf("could not create zst reader: %v", err))
	}
	return zst.IOReadCloser(), nil
}

// Return the xz reader and size of the endpoint "through the eye" of the previous reader.
// Assumes a single file was compressed. Note: the xz reader is not a closer so we wrap a
// nop Closer around it.
// NOTE: size is not stored in the xz header. This may require the file to be decompressed in
//
//	order to get its original size. For now 0 is returned.
//
// TODO: support gz size.
func (fr *FormatReaders) xzReader() (io.Reader, error) {
	xz, err := xz.NewReader(fr.TopReader())
	if err != nil {
		return nil, errors.New(fmt.Sprintf("could not create xz reader: %v", err))
	}
	return xz, nil
}

// Return the matching header, if one is found, from the passed-in map of known headers. After a
// successful read append a multi-reader to the receiver's reader stack.
// Note: .iso files are not detected here but rather in the Size() function.
// Note: knownHdrs is passed by reference and modified.
func (fr *FormatReaders) matchHeader(knownHdrs *Headers) (*Header, error) {
	_, err := fr.read(fr.buf) // read current header
	if err != nil {
		return nil, err
	}
	// append multi-reader so that the header data can be re-read by subsequent readers
	fr.appendReader(rdrMulti, bytes.NewReader(fr.buf))

	// loop through known headers until a match
	for format, kh := range *knownHdrs {
		if kh.Match(fr.buf) {
			// delete this header format key so that it's not processed again
			delete(*knownHdrs, format)
			return &kh, nil
		}
	}
	return nil, nil // no match
}

// Read from top-most reader. Note: ReadFull is needed since there may be intermediate,
// smaller multi-readers in the reader stack, and we need to be able to fill buf.
func (fr *FormatReaders) read(buf []byte) (int, error) {
	return io.ReadFull(fr.TopReader(), buf)
}

// Close Readers in reverse order.
func (fr *FormatReaders) Close() error {
	var rtnerr error
	for i := len(fr.readers) - 1; i >= 0; i-- {
		err := fr.readers[i].rdr.Close()
		if err != nil {
			rtnerr = err // tracking last error
		}
	}
	return rtnerr
}
