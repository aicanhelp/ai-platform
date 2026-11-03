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
	"bufio"
	"fmt"
	"os"
	"runtime"
	"strings"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

var (
	needRemoveImage = []string{}
)

func (op *Options) MigrateImage() {
	if !infrastructure.IsDocker() {
		log.Error("This migrate instruction only supports running in docker environment.")
		return
	}

	var imageList []string
	if len(op.Image) > 0 {
		imageList = append(imageList, op.Image)
	}
	if len(op.File) > 0 {
		fi, err := os.Open(op.File)
		if err != nil {
			log.Errorf("open migrate image list file %s failed: %v", op.File, err)
			return
		}
		buf := bufio.NewScanner(fi)
		for {
			if !buf.Scan() {
				break
			}
			if len(buf.Text()) == 0 {
				continue
			}
			imageList = append(imageList, buf.Text())
		}
	}
	if len(op.Arch) == 0 {
		op.Arch = runtime.GOARCH
	}
	archs := strings.Split(op.Arch, ",")
	for _, image := range imageList {
		sourceImage := op.Source + "/" + image
		sourceImage = strings.ReplaceAll(sourceImage, "//", "/")
		targetImage := op.Target + "/" + image
		targetImage = strings.ReplaceAll(targetImage, "//", "/")
		err := syncImage(sourceImage, targetImage, archs)
		if err != nil {
			log.Errorf("sync image %s to %s failed: %v", sourceImage, targetImage, err)
			return
		}
	}
	log.Info("Image migration completed. ")
	// cleanLocalImage
	cleanBuildImage()
}

// syncImage Synchronize an image between mirror repositories
// image collection of multiple architectures is supported in the following scenarios
// 1. The image itself is multi-architecture
// 2. The image is of a single architecture, and the suffix contains the -ARCH architecture
/*
multi-architecture image alpine:3.15
or
alpine:3.15-amd64
alpine:3.15-arm64
*/
func syncImage(source, target string, arch []string) error {
	if len(arch) == 1 {
		return syncSingleArchImage(source, target, arch[0])
	}
	return syncMultiArchImage(source, target, arch)
}

// syncSingleArchImage 同步单架构镜像
func syncSingleArchImage(source, target, arch string) error {
	imageAddress, err := pullImageWithRetry(source, arch)
	if err != nil {
		return err
	}

	if err = global.Docker.Tag(imageAddress, target); err != nil {
		log.Errorf("docker tag %s %s error: %v", imageAddress, target, err)
		return err
	}

	if err = pushImage(target); err != nil {
		return err
	}

	needRemoveImage = append(needRemoveImage, imageAddress)
	needRemoveImage = append(needRemoveImage, target)
	return nil
}

// syncMultiArchImage 同步多架构镜像
func syncMultiArchImage(source, target string, arch []string) error {
	manifestCreateCmd := fmt.Sprintf("docker manifest create --insecure %s", target)
	var manifestAnnotate []string

	for _, ar := range arch {
		targetArch, err := processArchImage(source, target, ar)
		if err != nil {
			return err
		}
		manifestCreateCmd += fmt.Sprintf("  --amend %s", targetArch)
		manifestAnnotate = append(manifestAnnotate,
			fmt.Sprintf("docker manifest annotate %s --os linux --arch %s %s", target, ar, targetArch))
	}
	log.Infof("Multi-arch image sync completed: target=%s, archs=%d", target, len(arch))

	return global.ExecuteManifestCommands(target, manifestCreateCmd, manifestAnnotate)
}

// pullImageWithRetry 尝试拉取镜像
func pullImageWithRetry(source, arch string) (string, error) {
	imageAddress := source
	log.Infof("Try pulling away the mirror image %s", imageAddress)
	if err := global.Docker.Pull(docker.ImageRef{
		Image:    imageAddress,
		Platform: arch,
	}, utils.RetryOptions{MaxRetry: 3, Delay: 1}); err != nil {
		log.Warnf("Failed to pull the mirror %v", err)
		imageAddress = source + "-" + arch
		log.Debugf("Try pulling away the mirror image %s", imageAddress)
		if err = global.Docker.Pull(docker.ImageRef{
			Image:    imageAddress,
			Platform: arch,
		}, utils.RetryOptions{MaxRetry: 3, Delay: 1}); err != nil {
			log.Errorf("Failed to pull the mirror %v", err)
			return "", err
		}
	}
	log.Infof("Pull image %s success", imageAddress)
	return imageAddress, nil
}

// processArchImage 处理特定架构的镜像
func processArchImage(source, target, arch string) (string, error) {
	imageAddress, err := pullImageWithRetry(source, arch)
	if err != nil {
		return "", err
	}

	targetArch := fmt.Sprintf("%s-%s", target, arch)
	if err = global.Docker.Tag(imageAddress, targetArch); err != nil {
		log.Errorf("docker tag %s %s error: %v", imageAddress, target, err)
		return "", err
	}
	log.Infof("docker tag %s %s", imageAddress, targetArch)

	if err = global.Docker.Remove(docker.ImageRef{Image: imageAddress}); err != nil {
		log.Errorf("Image cannot be removed %s %v", imageAddress, err)
		return "", err
	}

	if err = pushImage(targetArch); err != nil {
		return "", err
	}

	needRemoveImage = append(needRemoveImage, targetArch)
	return targetArch, nil
}

// pushImage 推送镜像
func pushImage(target string) error {
	log.Infof("docker push %s", target)
	push := fmt.Sprintf("docker push %s", target)
	if result, err := global.Command.ExecuteCommandWithOutput("/bin/sh", "-c", push); err != nil {
		log.Errorf("Image push failed %s %s %v", target, result, err)
		return err
	}
	log.Infof("Image push %s success", target)
	return nil
}

func cleanBuildImage() {
	for _, image := range needRemoveImage {
		if err := global.Docker.Remove(docker.ImageRef{
			Image: image,
		}); err != nil {
			log.Warnf("Remove build image %s failed: %v", image, err)
		}
	}
}
