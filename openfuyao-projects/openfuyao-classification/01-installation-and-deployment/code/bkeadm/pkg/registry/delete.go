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
	"fmt"
	"strings"

	"github.com/containers/common/pkg/retry"
	"github.com/containers/image/v5/transports/alltransports"
	"github.com/containers/image/v5/types"

	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

func (op *Options) Delete() {
	if len(op.Args) != 1 {
		log.Error("only one mirror can be deleted at a time")
		return
	}
	imageName := op.Args[0]
	if !strings.HasPrefix(imageName, "docker://") {
		imageName = fmt.Sprintf("docker://%s", imageName)
	}
	ref, err := alltransports.ParseImageName(imageName)
	if err != nil {
		log.Errorf("invalid image name %s: %v", imageName, err)
		return
	}

	destinationCtx, err := newSystemContext()
	if err != nil {
		log.Errorf("create system context for delete %s failed: %v", imageName, err)
		return
	}
	if op.DestTLSVerify {
		destinationCtx.DockerInsecureSkipTLSVerify = types.NewOptionalBool(true)
		destinationCtx.DockerDaemonInsecureSkipTLSVerify = true
	}

	ctx := context.Background()
	err = retry.IfNecessary(ctx, func() error {
		return ref.DeleteImage(ctx, destinationCtx)
	}, &retry.Options{
		MaxRetry: 0,
		Delay:    0,
	})
	if err != nil {
		log.Errorf("Delete image %s failed: %v", op.Image, err)
		return
	}
	return
}
