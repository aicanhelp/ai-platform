/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package containerd

import (
	"fmt"
	"os"

	"github.com/containerd/console"
	"github.com/containerd/containerd"
	"github.com/containerd/containerd/cio"
	"github.com/containerd/containerd/cmd/ctr/commands"
	"github.com/containerd/containerd/cmd/ctr/commands/tasks"
	"github.com/containerd/containerd/defaults"
	"github.com/containerd/containerd/oci"
	"github.com/containerd/containerd/runtime/v2/runc/options"
	"github.com/containerd/containerd/snapshots"
	"github.com/opencontainers/runtime-spec/specs-go"
	"github.com/pkg/errors"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

type ContainerSpec struct {
	Image       string            `json:"image"`
	Args        []string          `json:"args"`
	Env         []string          `json:"env"`
	Mount       []specs.Mount     `json:"mount"`
	Privileged  bool              `json:"privileged"`
	HostNetwork bool              `json:"hostNetwork"`
	Annotation  map[string]string `json:"annotation"`
}

func (c *Client) Run(cs ContainerSpec) error {
	container, err := c.newContainer(cs)
	if err != nil {
		return err
	}
	log.Infof("Running container %s", container.ID())
	var con console.Console
	var opts []containerd.NewTaskOpts
	var ioOpts []cio.Opt

	task, err := tasks.NewTask(c.ctx, c.condClient, container, "", con, false, "", ioOpts, opts...)
	if err != nil {
		return err
	}
	if err := task.Start(c.ctx); err != nil {
		return err
	}
	return nil
}

func (c *Client) newContainer(cs ContainerSpec) (containerd.Container, error) {
	id := GenerateID()
	image, err := c.prepareImage(cs.Image)
	if err != nil {
		return nil, err
	}

	opts, err := c.buildOCISpecOptions(cs, image)
	if err != nil {
		return nil, err
	}
	cOpts := c.buildContainerOptions(cs, image, id, opts)

	// oci.WithImageConfig (WithUsername, WithUserID) depends on access to rootfs for resolving via
	// the /etc/{passwd,group} files. So cOpts needs to have precedence over opts.
	return c.condClient.NewContainer(c.ctx, id, cOpts...)
}

func (c *Client) prepareImage(imageName string) (containerd.Image, error) {
	snapshotter := ""
	var image containerd.Image
	i, err := c.condClient.ImageService().Get(c.ctx, imageName)
	if err != nil {
		return nil, err
	}
	image = containerd.NewImage(c.condClient, i)
	unpacked, err := image.IsUnpacked(c.ctx, snapshotter)
	if err != nil {
		return nil, err
	}
	if !unpacked {
		if err := image.Unpack(c.ctx, snapshotter); err != nil {
			return nil, err
		}
	}
	return image, nil
}

func (c *Client) buildOCISpecOptions(cs ContainerSpec, image containerd.Image) ([]oci.SpecOpts, error) {
	var opts []oci.SpecOpts
	opts = append(opts, oci.WithDefaultSpec(), oci.WithDefaultUnixDevices)
	opts = append(opts, oci.WithEnv(cs.Env))
	opts = append(opts, oci.WithMounts(cs.Mount))
	opts = append(opts, oci.WithImageConfig(image))

	if len(cs.Args) > 0 {
		opts = append(opts, oci.WithProcessArgs(cs.Args...))
	}
	if cs.Privileged {
		opts = append(opts, oci.WithPrivileged, oci.WithAllDevicesAllowed, oci.WithHostDevices)
	}
	if cs.HostNetwork {
		hostNetworkOpts, err := c.buildHostNetworkOptions()
		if err != nil {
			return nil, err
		}
		opts = append(opts, hostNetworkOpts...)
	}
	opts = append(opts, oci.WithAnnotations(cs.Annotation))
	return opts, nil
}

// buildHostNetworkOptions builds OCI spec options for host network mode
func (c *Client) buildHostNetworkOptions() ([]oci.SpecOpts, error) {
	networkOpts := []oci.SpecOpts{
		oci.WithHostNamespace(specs.NetworkNamespace),
		oci.WithHostHostsFile,
		oci.WithHostResolvconf,
	}
	currentHostname, err := os.Hostname()
	if err != nil {
		return nil, errors.Wrap(err, "failed to retrieve system hostname")
	}
	hostnameEnv := fmt.Sprintf("HOSTNAME=%s", currentHostname)
	networkOpts = append(networkOpts, oci.WithEnv([]string{hostnameEnv}))

	return networkOpts, nil
}

func (c *Client) buildContainerOptions(cs ContainerSpec, image containerd.Image, id string, opts []oci.SpecOpts) []containerd.NewContainerOpts {
	var cOpts []containerd.NewContainerOpts
	snapshotter := ""

	cOpts = append(cOpts,
		containerd.WithImage(image),
		containerd.WithImageConfigLabels(image),
		containerd.WithAdditionalContainerLabels(image.Labels()),
		containerd.WithSnapshotter(snapshotter))
	cOpts = append(cOpts, containerd.WithNewSnapshot(id, image,
		snapshots.WithLabels(commands.LabelArgs([]string{"snapshotter-label=snapshotter-label"}))))
	cOpts = append(cOpts, containerd.WithImageStopSignal(image, "SIGTERM"))

	runtimeOpts := &options.Options{}
	cOpts = append(cOpts, containerd.WithRuntime(defaults.DefaultRuntime, runtimeOpts))

	var s specs.Spec
	spec := containerd.WithSpec(&s, opts...)
	cOpts = append(cOpts, spec)

	return cOpts
}
