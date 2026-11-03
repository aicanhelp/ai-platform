//go:build e2e

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You can obtain a copy of PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package warmup

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	ginkgo "github.com/onsi/ginkgo/v2"
)

// warmDB wraps a direct PostgreSQL connection to snapshot_warm_task /
// superpod_warm_detail, mirroring the "psql 直查" assertions of the case
// matrix. The connection goes through the postgres port-forward.
type warmDB struct {
	conn *pgx.Conn
}

func (e *warmEnv) db(tb ginkgo.GinkgoTInterface) *warmDB {
	tb.Helper()
	if e.dbConn == nil {
		e.dbConn = &warmDB{}
	}
	e.dbConn.ensureConnected(tb, e)
	return e.dbConn
}

// ensureConnected verifies the PostgreSQL connection is alive and reconnects
// (re-establishing the port-forward first) when it is not. The outage specs
// replace the warm-postgres pod, which invalidates both the port-forward and
// the cached connection.
func (d *warmDB) ensureConnected(tb ginkgo.GinkgoTInterface, e *warmEnv) {
	tb.Helper()
	if d.conn != nil {
		if err := d.conn.Ping(context.Background()); err == nil {
			return
		}
		_ = d.conn.Close(context.Background())
		d.conn = nil
	}
	e.restartPGPortForward(tb)
	conn, err := pgx.Connect(context.Background(), wPGDSN)
	if err != nil {
		tb.Fatalf("connect postgres: %v", err)
	}
	d.conn = conn
}

type warmTaskRow struct {
	TaskID             string          `json:"task_id"`
	State              int16           `json:"state"`
	TotalSnapshotCount int64           `json:"total_snapshot_count"`
	SnapshotIDs        json.RawMessage `json:"snapshot_ids"`
}

func (d *warmDB) taskRow(tb ginkgo.GinkgoTInterface, taskID string) warmTaskRow {
	tb.Helper()
	var row warmTaskRow
	err := d.conn.QueryRow(context.Background(),
		`SELECT task_id, state, total_snapshot_count, snapshot_ids FROM snapshot_warm_task WHERE task_id = $1`,
		taskID).Scan(&row.TaskID, &row.State, &row.TotalSnapshotCount, &row.SnapshotIDs)
	if err != nil {
		tb.Fatalf("query task %s: %v", taskID, err)
	}
	return row
}

// taskState returns the current snapshot_warm_task.state for the task.
func (e *warmEnv) taskState(tb ginkgo.GinkgoTInterface, taskID string) int {
	tb.Helper()
	var state int16
	err := e.db(tb).conn.QueryRow(context.Background(),
		`SELECT state FROM snapshot_warm_task WHERE task_id = $1`, taskID).Scan(&state)
	if err != nil {
		tb.Fatalf("query task state: %v", err)
	}
	return int(state)
}

// waitForTaskTerminal polls snapshot_warm_task.state until it reaches a
// terminal value (2=SUCCEED, 3=FAILED) or the timeout elapses.
func (e *warmEnv) waitForTaskTerminal(tb ginkgo.GinkgoTInterface, taskID string, timeout time.Duration) int {
	tb.Helper()
	var final int
	waitFor(tb, fmt.Sprintf("task %s reaching terminal state", taskID), timeout, func() bool {
		final = e.taskState(tb, taskID)
		return final == 2 || final == 3
	})
	return final
}

// observeTaskStates polls the task state at high frequency until terminal,
// returning every observed state. A fast dispatch window means a 1s poll can
// miss WARMING, so this uses 100ms ticks.
func (e *warmEnv) observeTaskStates(tb ginkgo.GinkgoTInterface, taskID string, timeout time.Duration) map[int]bool {
	tb.Helper()
	observed := map[int]bool{}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		state := e.taskState(tb, taskID)
		observed[state] = true
		if state == 2 || state == 3 {
			return observed
		}
		time.Sleep(100 * time.Millisecond)
	}
	tb.Fatalf("task %s did not reach terminal state, last observed: %v", taskID, observed)
	return observed
}

type warmDetailRow struct {
	SnapshotID       string          `json:"snapshot_id"`
	Category         int16           `json:"category"`
	SuperPodID       json.RawMessage `json:"superpod_id"`
	WarmedSuperPodID json.RawMessage `json:"warmed_superpod_id"`
	State            int16           `json:"state"`
}

// detailRow returns the superpod_warm_detail row for the snapshot. found is
// false when no row exists yet (the executor writes rows per snapshot, so
// mid-task some snapshots have no row).
func (d *warmDB) detailRow(tb ginkgo.GinkgoTInterface, snapshotID string) (warmDetailRow, bool) {
	tb.Helper()
	var row warmDetailRow
	err := d.conn.QueryRow(context.Background(),
		`SELECT snapshot_id, category, superpod_id, warmed_superpod_id, state FROM superpod_warm_detail WHERE snapshot_id = $1`,
		snapshotID).Scan(&row.SnapshotID, &row.Category, &row.SuperPodID, &row.WarmedSuperPodID, &row.State)
	if err == pgx.ErrNoRows {
		return warmDetailRow{SnapshotID: snapshotID, SuperPodID: []byte("[]"), WarmedSuperPodID: []byte("[]")}, false
	}
	if err != nil {
		tb.Fatalf("query detail %s: %v", snapshotID, err)
	}
	return row, true
}

// detailRowMust returns the detail row, failing when it does not exist.
func (d *warmDB) detailRowMust(tb ginkgo.GinkgoTInterface, snapshotID string) warmDetailRow {
	tb.Helper()
	row, found := d.detailRow(tb, snapshotID)
	if !found {
		tb.Fatalf("detail row for %s does not exist", snapshotID)
	}
	return row
}

// deleteDetails removes detail rows for the given snapshots so a spec can
// start from a clean cache state (test reset, mirrors a pristine cluster).
func (d *warmDB) deleteDetails(tb ginkgo.GinkgoTInterface, snapshotIDs ...string) {
	tb.Helper()
	if len(snapshotIDs) == 0 {
		return
	}
	_, err := d.conn.Exec(context.Background(),
		`DELETE FROM superpod_warm_detail WHERE snapshot_id = ANY($1)`, snapshotIDs)
	if err != nil {
		tb.Fatalf("delete details %v: %v", snapshotIDs, err)
	}
}

// purgeLeftoverTasks cancels tasks left in a non-terminal state by a previous
// suite run that was killed mid-flight (their workers would otherwise keep
// retrying and starve the pool while later specs run). Terminal-state rows
// are left in place.
func (d *warmDB) purgeLeftoverTasks(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	tag, err := d.conn.Exec(context.Background(),
		`UPDATE snapshot_warm_task SET state = 3, updated_at = now() WHERE state IN (0, 1)`)
	if err != nil {
		tb.Fatalf("purge leftover tasks: %v", err)
	}
	if n := tag.RowsAffected(); n > 0 {
		tb.Logf("purged %d leftover non-terminal warm task(s) from a previous run", n)
	}
}

// jsonStrings decodes a JSONB array column into a []string.
func jsonStrings(tb ginkgo.GinkgoTInterface, raw json.RawMessage) []string {
	tb.Helper()
	var out []string
	if err := json.Unmarshal(raw, &out); err != nil {
		tb.Fatalf("decode json array %s: %v", raw, err)
	}
	return out
}
