package executor

type Executor interface {
	Exec(cmd string, opts ...ExecOption) (*ExecResult, error)
}

type ExecOption struct{}

type ExecResult struct {
	Command  string
	Stdout   string
	Stderr   string
	ExitCode int
}
