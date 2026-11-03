/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package app

import (
	"context"
	"errors"
	"net/http"

	"github.com/spf13/cobra"
	"k8s.io/apiserver/pkg/server"

	"openfuyao/oauth-server/cmd/oauth-server/app/config"
	"openfuyao/oauth-server/cmd/oauth-server/app/options"
	"openfuyao/oauth-server/pkg/apiserver"
	"openfuyao/oauth-server/pkg/fuyaoerrors"
	"openfuyao/oauth-server/pkg/zlog"
)

// NewOAuthServerCommand is the cobra command for the whole service
func NewOAuthServerCommand() *cobra.Command {
	options := options.NewOAuthServerOption()

	cmd := &cobra.Command{
		Use:   "openfuyao-oauth-server",
		Short: "The authorization server to generate the oauth2 access token and manage the users.",
		RunE: func(cmd *cobra.Command, args []string) error {
			if err := options.Validate(); err != nil {
				zlog.LogFatal(err)
			}

			oAuthAPIServerConfigs, err := options.ReadConfig()
			if err != nil {
				zlog.LogFatal(err)
			}

			if errs := oAuthAPIServerConfigs.Complete().Validate(); len(errs) != 0 {
				for _, err = range errs {
					zlog.LogError(err)
				}
				zlog.LogFatal(fuyaoerrors.ErrIntExitSignal)
			}

			return wrapRunOAuthServerServer(oAuthAPIServerConfigs, server.SetupSignalContext())
		},
		SilenceUsage: true,
	}

	// Handle flags
	flags := cmd.Flags()

	// Read config path from cli
	flags.StringVar(&options.ConfigFile, "configFile", "", "Location of the authserver configuration file to run from.")

	return cmd
}

func wrapRunOAuthServerServer(c *config.OAuthServerAPIServerConfig, ctx context.Context) error {
	innerCtx, cancelFunc := context.WithCancel(context.TODO())
	errCh := make(chan error)
	defer close(errCh)
	go func() {
		if err := runOAuthServerServer(c, innerCtx); err != nil {
			errCh <- err
		}
	}()

	for {
		select {
		case <-ctx.Done():
			cancelFunc()
			return nil
		case err := <-errCh:
			cancelFunc()
			return err
		}
	}
}

func runOAuthServerServer(c *config.OAuthServerAPIServerConfig, ctx context.Context) error {
	oAuthServerAPIServer, err := apiserver.NewOAuthServerAPIServer(c, ctx.Done())
	if err != nil {
		return err
	}

	err = oAuthServerAPIServer.PrepareRun(ctx.Done())
	if err != nil {
		return err
	}

	err = oAuthServerAPIServer.Run(ctx)
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}

	return err
}
