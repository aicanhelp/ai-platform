package contentprovider

import (
	"context"
	"testing"
)

// TestNewEmptyContentProvider verifies the constructor returns a non-nil
// DynamicContentProvider.
func TestNewEmptyContentProvider(t *testing.T) {
	p := NewEmptyContentProvider()
	if p == nil {
		t.Fatal("NewEmptyContentProvider() returned nil, want non-nil")
	}
	// Ensure it satisfies the DynamicContentProvider interface at compile time.
	var _ DynamicContentProvider = p
}

// TestEmptyContentProviderCurrentContent verifies CurrentContent returns nil.
func TestEmptyContentProviderCurrentContent(t *testing.T) {
	p := NewEmptyContentProvider()
	if got := p.CurrentContent(); got != nil {
		t.Fatalf("CurrentContent() = %v, want nil", got)
	}
}

// TestEmptyContentProviderAddListener verifies AddListener does not panic when
// given a (no-op) listener.
func TestEmptyContentProviderAddListener(t *testing.T) {
	p := NewEmptyContentProvider()
	called := false
	p.AddListener(func(content []byte) {
		called = true
	})
	// AddListener is a no-op; it must never invoke the listener.
	if called {
		t.Fatal("AddListener invoked the listener, expected no-op")
	}
}

// TestEmptyContentProviderRunOnce verifies RunOnce returns nil.
func TestEmptyContentProviderRunOnce(t *testing.T) {
	p := NewEmptyContentProvider()
	if err := p.RunOnce(); err != nil {
		t.Fatalf("RunOnce() = %v, want nil", err)
	}
}

// TestEmptyContentProviderRun verifies Run returns nil for a non-cancelled
// context.
func TestEmptyContentProviderRun(t *testing.T) {
	p := NewEmptyContentProvider()
	ctx := context.Background()
	if err := p.Run(ctx); err != nil {
		t.Fatalf("Run(ctx) = %v, want nil", err)
	}
}

// TestEmptyContentProviderRunCancelledContext verifies Run accepts a cancelled
// context without panicking.
func TestEmptyContentProviderRunCancelledContext(t *testing.T) {
	p := NewEmptyContentProvider()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := p.Run(ctx); err != nil {
		t.Fatalf("Run(cancelled ctx) = %v, want nil", err)
	}
}
