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

package registry

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/containers/image/v5/docker"
	"github.com/containers/image/v5/image"
	_ "github.com/containers/image/v5/image"
	"github.com/containers/image/v5/transports/alltransports"
	"github.com/containers/image/v5/types"

	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

const (
	// DockerV2Schema2MediaType MIME type represents Docker manifest schema 2
	DockerV2Schema2MediaType = "application/vnd.docker.distribution.manifest.v2+json"
	// DockerV2ListMediaType MIME type represents Docker manifest schema 2 list
	DockerV2ListMediaType = "application/vnd.docker.distribution.manifest.list.v2+json"

	// DockerV1Schema2MediaType MIME type represents Docker manifest schema 1
	DockerV1Schema2MediaType = "application/vnd.oci.image.manifest.v1+json"
	// DockerV1ListMediaType MIME type represents Docker manifest schema 1 list
	DockerV1ListMediaType = "application/vnd.oci.image.index.v1+json"
)

type ImageArch struct {
	Name         string `json:"name"`
	OS           string `json:"os"`
	Architecture string `json:"architecture"`
}

type DockerV2List struct {
	SchemaVersion int        `json:"schemaVersion"`
	MediaType     string     `json:"mediaType"`
	Manifests     []Manifest `json:"manifests"`
}
type Manifest struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int    `json:"size"`
	Platform  struct {
		Architecture string `json:"architecture"`
		OS           string `json:"os"`
	} `json:"platform"`
}

type DockerV2Schema struct {
	SchemaVersion int    `json:"schemaVersion"`
	MediaType     string `json:"mediaType"`
	Config        struct {
		MediaType string `json:"mediaType"`
		Size      int    `json:"size"`
		Digest    string `json:"digest"`
	} `json:"config"`
	Layers []struct {
		MediaType string `json:"mediaType"`
		Size      int    `json:"size"`
		Digest    string `json:"digest"`
	}
}

type imageManifestInfo struct {
	digest    string
	size      int
	mediaType string
}

func (op *Options) Inspect() {
	ctx := context.Background()

	imageSource, destinationCtx, err := op.createInspectImageSource(ctx)
	if err != nil {
		log.Errorf("create inspect image source for %s failed: %v", op.Image, err)
		return
	}
	defer func() { _ = imageSource.Close() }()

	rawManifest, _, err := imageSource.GetManifest(ctx, nil)
	if err != nil {
		log.Errorf("get manifest for %s failed: %v", op.Image, err)
		return
	}

	handled, err := op.handleMultiArchManifest(rawManifest)
	if err != nil {
		log.Errorf("handle multi-arch manifest for %s failed: %v", op.Image, err)
		return
	}
	if handled {
		return
	}

	op.printSingleArchConfig(ctx, destinationCtx, imageSource)
}

// createInspectImageSource 创建检查镜像源
func (op *Options) createInspectImageSource(ctx context.Context) (types.ImageSource, *types.SystemContext, error) {
	destinationCtx, err := newSystemContext()
	if err != nil {
		return nil, nil, err
	}
	if op.DestTLSVerify {
		destinationCtx.DockerInsecureSkipTLSVerify = types.NewOptionalBool(true)
		destinationCtx.DockerDaemonInsecureSkipTLSVerify = true
	}

	imageName := op.Image
	if !strings.HasPrefix(op.Image, "docker://") {
		imageName = "docker://" + op.Image
	}

	ref, err := alltransports.ParseImageName(imageName)
	if err != nil {
		return nil, nil, err
	}
	imageSource, err := ref.NewImageSource(ctx, destinationCtx)
	if err != nil {
		return nil, nil, err
	}
	return imageSource, destinationCtx, nil
}

// handleMultiArchManifest 处理多架构 manifest
func (op *Options) handleMultiArchManifest(rawManifest []byte) (bool, error) {
	dvl := DockerV2List{}
	if err := json.Unmarshal(rawManifest, &dvl); err != nil {
		return false, err
	}
	if dvl.Manifests != nil && len(dvl.Manifests) > 0 {
		fmt.Println(string(rawManifest))
		return true, nil
	}
	return false, nil
}

// printSingleArchConfig 打印单架构配置
func (op *Options) printSingleArchConfig(ctx context.Context, destinationCtx *types.SystemContext,
	imageSource types.ImageSource) {
	img, err := image.FromUnparsedImage(ctx, destinationCtx, image.UnparsedInstance(imageSource, nil))
	if err != nil {
		log.Errorf("load image from source for %s failed: %v", op.Image, err)
		return
	}
	outputData, err := img.OCIConfig(ctx)
	if err != nil {
		log.Errorf("get OCI config for %s failed: %v", op.Image, err)
		return
	}
	out, err := json.MarshalIndent(outputData, "", "    ")
	if err != nil {
		log.Errorf("marshal OCI config for %s failed: %v", op.Image, err)
		return
	}
	if _, err = fmt.Fprintf(os.Stdout, "%s\n", string(out)); err != nil {
		log.Errorf("write OCI config for %s failed: %v", op.Image, err)
	}
}

func getImageSha(name string) (imageManifestInfo, error) {
	ctx := context.Background()
	info := imageManifestInfo{}
	imageName := name
	if !strings.HasPrefix(name, "docker://") {
		imageName = "docker://" + name
	}

	destRef, err := alltransports.ParseImageName(imageName)
	if err != nil {
		return info, err
	}

	destinationCtx, err := newSystemContext()
	if err != nil {
		return info, err
	}

	destinationCtx.DockerInsecureSkipTLSVerify = types.NewOptionalBool(true)
	destinationCtx.DockerDaemonInsecureSkipTLSVerify = true

	dig, err := docker.GetDigest(ctx, destinationCtx, destRef)
	if err != nil {
		return info, err
	}
	info.digest = dig.String()

	imageSource, err := destRef.NewImageSource(ctx, destinationCtx)
	if err != nil {
		return info, err
	}
	mf, mediaType, err := imageSource.GetManifest(ctx, &dig)
	if err != nil {
		return info, err
	}
	info.mediaType = mediaType

	dvs := DockerV2Schema{}
	err = json.Unmarshal(mf, &dvs)
	if err != nil {
		// Not fatal for our purpose (we mainly need digest + size + mediaType).
		// Keep size based on raw manifest bytes and return success.
		info.size = len(mf)
		return info, nil
	}
	info.size = len(mf)
	return info, nil
}

func putManifests(manifests string, name string) error {
	imageName := name
	if !strings.HasPrefix(name, "docker://") {
		imageName = "docker://" + name
	}

	destRef, err := alltransports.ParseImageName(imageName)
	if err != nil {
		return err
	}
	destinationCtx, err := newSystemContext()
	if err != nil {
		return err
	}
	destinationCtx.DockerInsecureSkipTLSVerify = types.NewOptionalBool(true)
	destinationCtx.DockerDaemonInsecureSkipTLSVerify = true

	publicDest, err := destRef.NewImageDestination(context.Background(), destinationCtx)
	if err != nil {
		return err
	}
	err = publicDest.PutManifest(context.Background(), []byte(manifests), nil)
	if err != nil {
		return err
	}
	return nil
}

func CreateMultiArchImage(img []ImageArch, target string) error {
	dvl := DockerV2List{
		SchemaVersion: 2,
		MediaType:     DockerV2ListMediaType,
		Manifests:     []Manifest{},
	}
	// If any child manifest is OCI, prefer producing an OCI index for maximum containerd/nerdctl compatibility.
	preferOCIIndex := false
	for _, im := range img {
		info, err := getImageSha(im.Name)
		if err != nil {
			return err
		}
		if strings.HasPrefix(info.mediaType, "application/vnd.oci.") {
			preferOCIIndex = true
		}
		if info.mediaType == "" {
			info.mediaType = DockerV2Schema2MediaType
		}
		dvl.Manifests = append(dvl.Manifests, Manifest{
			MediaType: info.mediaType,
			Digest:    info.digest,
			Size:      info.size,
			Platform: struct {
				Architecture string `json:"architecture"`
				OS           string `json:"os"`
			}{
				Architecture: im.Architecture,
				OS:           im.OS,
			},
		})
	}
	if preferOCIIndex {
		dvl.MediaType = DockerV1ListMediaType // OCI index
	}

	dvlByte, err := json.MarshalIndent(dvl, "", "    ")
	if err != nil {
		return err
	}
	err = putManifests(string(dvlByte), target)
	if err != nil {
		return err
	}
	return nil
}

func (op *Options) Manifests() {
	var ias []ImageArch
	for _, img := range op.Args {
		_, after, found := strings.Cut(img, op.Image)
		if !found {
			log.Errorf("invalid image name %s", img)
			return
		}
		_, after, found = strings.Cut(after, "-")
		if !found {
			log.Errorf("invalid image name %s", img)
			return
		}
		ias = append(ias, ImageArch{
			Name:         img,
			OS:           "linux",
			Architecture: after,
		})
	}

	err := CreateMultiArchImage(ias, op.Image)
	if err != nil {
		log.Errorf("Create multi-arch image manifest %s failed: %v", op.Image, err)
	}
}

func IsMultiArchManifests(srcTLSVerify bool, imageName string) bool {
	ctx := context.Background()

	destinationCtx, err := newSystemContext()
	if err != nil {
		log.Errorf("create system context for %s failed: %v", imageName, err)
		return false
	}
	if !srcTLSVerify {
		destinationCtx.DockerInsecureSkipTLSVerify = types.NewOptionalBool(true)
		destinationCtx.DockerDaemonInsecureSkipTLSVerify = true
	}
	if !strings.HasPrefix(imageName, "docker://") {
		imageName = "docker://" + imageName
	}

	ref, err := alltransports.ParseImageName(imageName)
	if err != nil {
		log.Errorf("parse image name %s failed: %v", imageName, err)
		return false
	}
	imageSource, err := getNewImageSource(ref, ctx, destinationCtx,
		utils.NewRetryOptions(utils.MaxRetryCount, utils.DelayTime))
	if err != nil {
		log.Errorf("get image source for %s failed: %v", imageName, err)
		return false
	}
	rawManifest, _, err := getManifest(imageSource, ctx, utils.NewRetryOptions(utils.MaxRetryCount, utils.DelayTime))
	if err != nil {
		log.Errorf("get manifest for %s failed: %v", imageName, err)
		return false
	}
	dvl := DockerV2List{}
	err = json.Unmarshal(rawManifest, &dvl)
	if err != nil {
		log.Errorf("unmarshal manifest for %s failed: %v", imageName, err)
		return false
	}
	if dvl.Manifests == nil || len(dvl.Manifests) < 1 {
		return false
	}
	return true
}

func getNewImageSource(ref types.ImageReference, ctx context.Context, destinationCtx *types.SystemContext,
	options utils.RetryOptions) (types.ImageSource, error) {
	var imageSource types.ImageSource
	var err error

	for i := 0; i < options.MaxRetry; i++ {
		imageSource, err = ref.NewImageSource(ctx, destinationCtx)
		if err == nil {
			break
		}
		log.Errorf("failed to get new image source: %v, retrying...", err)
		time.Sleep(options.Delay)
	}

	return imageSource, err
}

func getManifest(imageSource types.ImageSource, ctx context.Context,
	options utils.RetryOptions) ([]byte, string, error) {
	var rawManifest []byte
	var mediaType string
	var err error

	for i := 0; i < options.MaxRetry; i++ {
		rawManifest, mediaType, err = imageSource.GetManifest(ctx, nil)
		if err == nil {
			break
		}
		log.Errorf("failed to get manifest: %v, retrying...", err)
		time.Sleep(options.Delay)
	}

	return rawManifest, mediaType, err
}
