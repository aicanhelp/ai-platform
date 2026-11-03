# golang版本
- [Kubernetes version](https://github.com/kubernetes/kubernetes/tree/v1.28.8):1.28.8
- [.go-version](https://github.com/kubernetes/kubernetes/blob/v1.28.8/.go-version):1.21.8
- [client-go version](https://github.com/kubernetes/client-go?tab=readme-ov-file#compatibility-client-go---kubernetes-clusters):0.28.8
> 目前kubernetes 1.28.8不在版本火车里，等许丹确认kubernetes是否在送检范围，如果在需要切换到1.28.4版本，
> kubernetes1.28.4的.go-version为1.20.11
## ref
### .go-version
The .go-version file is not specifically mentioned in the official Kubernetes documentation or in the Kubernetes GitHub repository12. However, in general, a .go-version file in a Go project is typically used to specify the version of Go that the project should be built with. This can help ensure consistency across different environments where the code might be built or run.

If you’re looking at a specific Kubernetes project or repository that includes a .go-version file, it would be best to refer to any accompanying documentation or contact the project maintainers for the most accurate information. If you have more specific questions or if there’s a particular context you’re interested in, feel free to ask! I’m here to help. 
### [client-go](https://github.com/kubernetes/client-go/blob/master/INSTALL.md)