/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package metrics

import (
	"testing"
	"time"
)

func TestTimeNow(t *testing.T) {
	result := TimeNow()
	if result == "" {
		t.Error("TimeNow returned empty string")
	}
	_, err := time.Parse(TimeFormat, result)
	if err != nil {
		t.Errorf("TimeNow returned invalid format: %v", err)
	}
}

func TestTimeNowUnix(t *testing.T) {
	result := TimeNowUnix()
	if result == "" {
		t.Error("TimeNowUnix returned empty string")
	}
}

func TestFormatTime(t *testing.T) {
	testTime := time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC)
	result := FormatTime(testTime)
	if result != "2024-01-01 12:00:00" {
		t.Errorf("expected '2024-01-01 12:00:00', got %s", result)
	}
}

func TestParseTimeUnix(t *testing.T) {
	tests := []struct {
		name      string
		timestamp string
		wantErr   bool
	}{
		{"valid timestamp", "1704110400", false},
		{"invalid timestamp", "invalid", true},
		{"empty timestamp", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := ParseTimeUnix(tt.timestamp)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseTimeUnix() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && result.IsZero() {
				t.Error("ParseTimeUnix returned zero time")
			}
		})
	}
}

func TestParseTimeFromTo(t *testing.T) {
	tests := []struct {
		name    string
		from    string
		to      string
		wantErr bool
	}{
		{"both empty", "", "", false},
		{"from empty", "", "1704110400", false},
		{"to empty", "1704110400", "", false},
		{"both valid", "1704110400", "1704196800", false},
		{"invalid from", "invalid", "1704196800", true},
		{"invalid to", "1704110400", "invalid", true},
		{"from after to", "1704196800", "1704110400", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startTime, endTime, err := ParseTimeFromTo(tt.from, tt.to)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseTimeFromTo() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr {
				if startTime.IsZero() {
					t.Error("startTime is zero")
				}
				if endTime.IsZero() {
					t.Error("endTime is zero")
				}
				if startTime.After(endTime) {
					t.Error("startTime is after endTime")
				}
			}
		})
	}
}
