package mfutil

type BKEComponent struct {
	Name       string
	LongName   string
	MfPath     string
	RenderFunc func(c *BKEComponent, cfg *BootScope) error
}

type BKEHAComponent struct {
	Name       string
	LongName   string
	ConfName   string
	ConfPath   string
	MfPath     string
	RenderFunc func(c *BKEHAComponent, cfg map[string]interface{}) error
}

type HANode struct {
	Hostname string
	IP       string
	Port     string
}

type Components []*BKEComponent

type HAComponents []*BKEHAComponent

func (h *HAComponents) SetMfPath(path string) {
	for _, c := range *h {
		c.MfPath = path
	}
}

func (c *Components) SetMfPath(path string) {
	for _, component := range *c {
		component.MfPath = path
	}
}

func GetHAComponentList() HAComponents {
	return HAComponents{
		HAProxyComponent(),
		KeepalivedComponent(),
	}
}

func GetIngressHaComponentList() HAComponents {
	return HAComponents{
		KeepalivedComponent(),
	}
}

func GetDefaultComponentList() Components {
	return Components{
		// required components
		APIServerComponent(),
		SchedulerComponent(),
		ControllerComponent(),
		// etcd
		EtcdComponent(),
	}
}

func GetComponentListWithOutEtcd() Components {
	return Components{
		APIServerComponent(),
		SchedulerComponent(),
		ControllerComponent(),
	}
}

func APIServerComponent() *BKEComponent {
	return &BKEComponent{
		Name:       KubeAPIServer,
		LongName:   "APIServer static pod yaml",
		RenderFunc: renderAPIServer,
	}
}

func SchedulerComponent() *BKEComponent {
	return &BKEComponent{
		Name:       KubeScheduler,
		LongName:   "Scheduler static pod yaml",
		RenderFunc: renderScheduler,
	}
}

func ControllerComponent() *BKEComponent {
	return &BKEComponent{
		Name:       KubeControllerManager,
		LongName:   "Controller manager static pod yaml",
		RenderFunc: renderController,
	}
}

func EtcdComponent() *BKEComponent {
	return &BKEComponent{
		Name:       Etcd,
		LongName:   "Etcd static pod yaml",
		RenderFunc: renderEtcd,
	}
}

// HA Components

func HAProxyComponent() *BKEHAComponent {
	return &BKEHAComponent{
		Name:       HAProxy,
		LongName:   "HAProxy static pod yaml",
		RenderFunc: renderHAProxy,
		ConfName:   HAProxyConfName,
		ConfPath:   HAProxyConfPath,
	}
}

func KeepalivedComponent() *BKEHAComponent {
	return &BKEHAComponent{
		Name:       Keepalived,
		LongName:   "Keepalived static pod yaml",
		RenderFunc: renderKeepalived,
		ConfName:   KeepAlivedConfName,
		ConfPath:   KeepAlivedConfPath,
	}
}
