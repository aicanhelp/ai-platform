package rpc

import (
	"io"
	"testing"
	"time"
)

func TestGetHelmChartBytes_ReturnsErrorWhenServerUnavailable(t *testing.T) {
	type result struct {
		reader io.Reader
		err    error
	}

	ch := make(chan result, 1)
	go func() {
		reader, err := GetHelmChartBytes("repo", "chart", "1.0.0")
		ch <- result{reader: reader, err: err}
	}()

	select {
	case res := <-ch:
		if res.err == nil {
			t.Fatalf("expected error when grpc server is unavailable, got nil (reader=%T)", res.reader)
		}
		// reader should be nil on error (current implementation).
		if res.reader != nil {
			t.Fatalf("expected nil reader on error, got %T", res.reader)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("GetHelmChartBytes took too long (possible grpc dial/handshake hang)")
	}
}

func TestGetHelmChart_ReturnsErrorWhenServerUnavailable(t *testing.T) {
	type result struct {
		chart interface{}
		err   error
	}

	ch := make(chan result, 1)
	go func() {
		c, err := GetHelmChart("repo", "chart", "1.0.0")
		ch <- result{chart: c, err: err}
	}()

	select {
	case res := <-ch:
		if res.err == nil {
			t.Fatalf("expected error when grpc server is unavailable, got nil (chart=%T)", res.chart)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("GetHelmChart took too long (possible grpc dial/handshake hang)")
	}
}

