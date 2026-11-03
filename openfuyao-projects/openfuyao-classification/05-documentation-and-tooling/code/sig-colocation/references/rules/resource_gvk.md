# 资源GVK
| Group | Version | Kind | Spec | 描述 | 申请人 | 审批 |  
| - | - | - | - | - | - | - | 
| console.openfuyao.com | v1beta  | ConsolePlugin | [#ConsolePlugin](#consoleplugin) | 用于描述扩展前端的加载与提供方式 | 杨煦庭 |      |

## GVK.Spec

### ConsolePlugin

```go
// ConsolePlugin specifies the mechanism that an openFuyao extended component serves its
// own UI resources, as well as how the resources will be rendered on the console webpage.
type ConsolePlugin struct {
    metav1.TypeMeta `json:",inline"`
    metav1.ObjectMeta `json:"metadata"`

    // +kubebuilder:validation:Required
    Spec ConsolePluginSpec `json:"spec"`
}

// ConsolePluginSpec is the desired configuration for the console plugin.
type ConsolePluginSpec struct {
    // displayName is the unique name of the plugin. The name should only include
    // alphabets, digits and '-'
    // +kubebuilder:validation:Required
    PluginName string `json:"pluginName"`

    // entrypoint is the location where the entrypoint of the plugin will be
    // rendered on the console webpage.
    // +kubebuilder:validation:Required
    Entrypoint ConsolePluginEntrypoint `json:entrypoint"`

    // displayName is the display name of the plugin on the UI entrypoint.
    // The dispalyName should be between 1 and 128 characters.
    // +kubebuilder:validation:Required
    DisplayName string `json:"displayName"`

    // backend holds the configuration of backend which is serving console's plugin.
    // +kubebuilder:validation:Required
    Backend *ConsolePluginBackend `json:"backend"`

    // enabled specifies whether the plugin would be loaded on console webpage.
    // +kubebuilder:default:=true
    // +optional
    Enabled bool `json:"enabled"`
}

// ConsolePluginEntrypoint is an enumeration of entrypoint location
type ConsolePluginEntrypoint string
const (
    // The plugin entrypoint will be rendered in the top navigation bar.
    NavEntrypoint ConsolePluginEntrypoint = "Nav"

    // The plugin entrypoint will be rendered in the side menu.
    SideEntrypoint ConsolePluginEntrypoint = "Side"
)

// ConsolePluginBackend holds information about the endpoint which serves the plugin.
type ConsolePluginBackend struct {
    // type is the type of the backend that supplies the plugin UI resources. 
    // Currently only service is supported.
    // +kubebuilder:validation:Required
    // +unionDiscriminator
    Type ConsolePluginBackendType `json:"type"`

    // service is the kubernetes service that exposes the plugin UI resources using a
    // deployment with an HTTP server.
    // +optional
    Service *ConsolePluginService `json:"service"`
}

// ConsolePluginBackendType is an enumeration of types of the backend that serves 
// the plugin UI resource.
type ConsolePluginBackendType string
const (
    // The UI resource of the plugin is supplied by a kubernetes service resource.
    ServiceBackendType ConsolePluginBackendType = "Service"
)

// ConsolePluginService holds information of the service that is serving
// plugin UI resources.
type ConsolePluginService struct {
    // name of the service serving the plugin UI resources.
    // +kubebuilder:validation:Required
    Name string `json:"name"`

    // namespace of the service serving the plugin UI resources.
    // +kubebuilder:validation:Required
    Namespace string `json:"namespace"`

    // port on which the service serving the plugin is listening to.
    // +kubebuilder:default:=80
    // +optional
    Port int32 `json:"port"`

    // basePath is the base path to the plugin UI resource in the HTTP server.
    // +kubebuilder:default:="/"
    // +optional
    BasePath string `json:"basePath"`
}
```