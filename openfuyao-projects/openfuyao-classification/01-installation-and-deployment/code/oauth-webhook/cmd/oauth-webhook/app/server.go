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

// Package app configures the whole oauth-webhook app
package app

import (
	"context"
	"errors"
	"net/http"

	"github.com/spf13/cobra"
	"k8s.io/apiserver/pkg/server"

	"openfuyao/oauth-webhook/cmd/oauth-webhook/app/config"
	"openfuyao/oauth-webhook/cmd/oauth-webhook/app/options"
	"openfuyao/oauth-webhook/pkg/apiserver"
	"openfuyao/oauth-webhook/pkg/fuyaoerrors"
	"openfuyao/oauth-webhook/pkg/zlog"
)

// NewOAuthWebhookCommand is the cobra command for the whole service
func NewOAuthWebhookCommand() *cobra.Command {
	options := options.NewOAuthWebhookOption()

	cmd := &cobra.Command{
		Use:   "openfuyao1-oauth-webhook",
		Short: "The authentication webhook to validate the access token.",
		RunE: func(cmd *cobra.Command, args []string) error {
			if err := options.Validate(); err != nil {
				zlog.LogFatal(err)
			}

			configs, err := options.ReadConfig()
			if err != nil {
				zlog.LogFatal(err)
			}

			if errs := configs.Complete().Validate(); len(errs) != 0 {
				for _, err = range errs {
					zlog.LogError(err)
				}
				zlog.LogFatal(fuyaoerrors.ErrIntExitSignal)
			}

			return wrapRunOAuthWebhookServer(configs, server.SetupSignalContext())
		},
		SilenceUsage: true,
	}

	// Handle flags
	flags := cmd.Flags()

	// Read config path from cli
	flags.StringVar(&options.ConfigFile, "configFile", "", "Location of the authwebhook configuration file to run from.")

	return cmd
}

func wrapRunOAuthWebhookServer(c *config.OAuthWebhookConfig, ctx context.Context) error {
	innerCtx, cancelFunc := context.WithCancel(context.TODO())
	errCh := make(chan error)
	defer close(errCh)
	go func() {
		if err := runOAuthWebhookServer(c, innerCtx); err != nil {
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

func runOAuthWebhookServer(c *config.OAuthWebhookConfig, ctx context.Context) error {
	oAuthWebhookAPIServer, err := apiserver.NewOAuthWebhookAPIServer(c, ctx.Done())
	if err != nil {
		return err
	}

	err = oAuthWebhookAPIServer.PrepareRun(ctx.Done())
	if err != nil {
		return err
	}

	err = oAuthWebhookAPIServer.Run(ctx)
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}

	return err
}
