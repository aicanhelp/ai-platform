package downloader

import (
	"os"

	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/bkeagent/utils/download"
)

const Name = "Downloader"

type DownloaderPlugin struct{}

func New() plugin.Plugin {
	return &DownloaderPlugin{}
}

func (d *DownloaderPlugin) Name() string {
	return Name
}

func (d *DownloaderPlugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"url": {
			Key:         "url",
			Value:       "",
			Required:    true,
			Default:     "",
			Description: "download url",
		},
		"rename": {
			Key:         "rename",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "rename downloaded file",
		},
		"chmod": {
			Key:         "perm",
			Value:       "",
			Required:    false,
			Default:     "0644",
			Description: "file permission",
		},
		"saveto": {
			Key:         "saveto",
			Value:       "",
			Required:    true,
			Default:     os.TempDir(),
			Description: "save to directory",
		},
	}
}

func (d *DownloaderPlugin) Execute(commands []string) ([]string, error) {
	commandMap, err := plugin.ParseCommands(d, commands)
	if err != nil {
		return nil, err
	}
	url := commandMap["url"]
	rename := commandMap["rename"]
	saveto := commandMap["saveto"]
	chmod := commandMap["chmod"]

	if err := download.ExecDownload(url, saveto, rename, chmod); err != nil {
		return nil, err
	}
	return nil, nil
}
