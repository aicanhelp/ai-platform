package utils

import (
	"bytes"
	"os/exec"
)

func ExecLocal(cmd string) (stdout, stderr string, err error) {
	c := exec.Command("bash", "-lc", cmd)
	var out, errBuf bytes.Buffer
	c.Stdout = &out
	c.Stderr = &errBuf
	err = c.Run()
	return out.String(), errBuf.String(), err
}
