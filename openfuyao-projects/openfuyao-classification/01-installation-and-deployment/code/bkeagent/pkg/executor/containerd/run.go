package containerd

import (
	"fmt"
	"log"
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
	log.Printf("[INFO] Running container %s", container.ID())
	var con console.Console
	opts := []containerd.NewTaskOpts{}
	ioOpts := []cio.Opt{}

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
	var (
		opts  []oci.SpecOpts
		cOpts []containerd.NewContainerOpts
		spec  containerd.NewContainerOpts
		id    = GenerateID()
	)
	opts = append(opts, oci.WithDefaultSpec(), oci.WithDefaultUnixDevices)
	opts = append(opts, oci.WithEnv(cs.Env))
	opts = append(opts, oci.WithMounts(cs.Mount))

	// rootfs
	snapshotter := ""
	var image containerd.Image
	i, err := c.condClient.ImageService().Get(c.ctx, cs.Image)
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
	opts = append(opts, oci.WithImageConfig(image))
	cOpts = append(cOpts,
		containerd.WithImage(image),
		containerd.WithImageConfigLabels(image),
		containerd.WithAdditionalContainerLabels(image.Labels()),
		containerd.WithSnapshotter(snapshotter))
	cOpts = append(cOpts, containerd.WithNewSnapshot(id, image,
		snapshots.WithLabels(commands.LabelArgs([]string{"snapshotter-label=snapshotter-label"}))))
	cOpts = append(cOpts, containerd.WithImageStopSignal(image, "SIGTERM"))

	if len(cs.Args) > 0 {
		opts = append(opts, oci.WithProcessArgs(cs.Args...))
	}
	if cs.Privileged {
		opts = append(opts, oci.WithPrivileged, oci.WithAllDevicesAllowed, oci.WithHostDevices)
	}
	if cs.HostNetwork {
		hostname, err := os.Hostname()
		if err != nil {
			return nil, errors.Wrap(err, "get hostname")
		}
		opts = append(opts,
			oci.WithHostNamespace(specs.NetworkNamespace),
			oci.WithHostHostsFile,
			oci.WithHostResolvconf,
			oci.WithEnv([]string{fmt.Sprintf("HOSTNAME=%s", hostname)}),
		)
	}
	runtimeOpts := &options.Options{}

	cOpts = append(cOpts, containerd.WithRuntime(defaults.DefaultRuntime, runtimeOpts))

	opts = append(opts, oci.WithAnnotations(cs.Annotation))
	var s specs.Spec
	spec = containerd.WithSpec(&s, opts...)

	cOpts = append(cOpts, spec)

	// oci.WithImageConfig (WithUsername, WithUserID) depends on access to rootfs for resolving via
	// the /etc/{passwd,group} files. So cOpts needs to have precedence over opts.
	return c.condClient.NewContainer(c.ctx, id, cOpts...)
}
