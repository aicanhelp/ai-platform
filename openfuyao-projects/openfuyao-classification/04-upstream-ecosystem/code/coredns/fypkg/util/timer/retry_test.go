// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package timer

import (
	"context"
	"fmt"
	"testing"
	"time"
)

func TestRunUtil(t *testing.T) {
	tcTemp := []struct {
		result bool
		count  int
	}{
		{
			result: false,
			count:  0,
		},
		{
			result: false,
			count:  0,
		},
		{
			result: false,
			count:  0,
		},
	}

	tcs := []struct {
		name      string
		ctx       context.Context
		period    time.Duration
		timeout   time.Duration
		condition ConditionFunc
		res       struct {
			result bool
			count  int
		}
	}{
		{
			name:    "condition return with no error first time",
			ctx:     context.Background(),
			period:  300 * time.Millisecond,
			timeout: 2 * time.Second,
			condition: func(ctx context.Context) error {
				tcTemp[0].count++
				return nil
			},
			res: struct {
				result bool
				count  int
			}{result: false, count: 1},
		},
		{
			name:    "condition return with errors third time",
			ctx:     context.Background(),
			period:  300 * time.Millisecond,
			timeout: 2 * time.Second,
			condition: func(ctx context.Context) error {
				tcTemp[1].count++
				if tcTemp[1].count == 3 {
					return nil
				}
				return fmt.Errorf("fake failed")
			},
			res: struct {
				result bool
				count  int
			}{result: false, count: 3},
		},
		{
			name:    "condition return with error always",
			ctx:     context.Background(),
			period:  300 * time.Millisecond,
			timeout: 2 * time.Second,
			condition: func(ctx context.Context) error {
				tcTemp[2].count++
				return fmt.Errorf("fake failed")
			},
			res: struct {
				result bool
				count  int
			}{result: false, count: 8},
		},
	}

	for i, tc := range tcs {
		r := &RetryableRunner{}
		t.Run(tc.name, func(t *testing.T) {
			r.RunUtil(tc.ctx, tc.timeout, tc.period, tc.condition)
			time.Sleep(tc.timeout + time.Second)
			if tcTemp[i] != tc.res {
				t.Errorf("expect call %d times, but call %d times", tc.res.count, tcTemp[i].count)
			}
		})
	}
}

func TestRunUtilWhenLastCallNoFinished(t *testing.T) {
	r := &RetryableRunner{}

	count1 := 0
	r.RunUtil(context.Background(), 2*time.Second, 300*time.Millisecond, func(ctx context.Context) error {
		count1++
		return fmt.Errorf("fake failed")
	})

	time.Sleep(1 * time.Second)
	count2 := 0
	r.RunUtil(context.Background(), 2*time.Second, 300*time.Millisecond, func(ctx context.Context) error {
		count2++
		if count2 >= 3 {
			return nil
		}
		return fmt.Errorf("fake failed")
	})

	time.Sleep(2 * time.Second)
	if count1 >= 8 {
		t.Errorf("expect call less %d times, but call %d times", 8, count1)
	}
	if count2 != 3 {
		t.Errorf("expect call %d times, but call %d times", 3, count2)
	}
}

func TestRunUtilWhenLastCallFinished(t *testing.T) {
	r := &RetryableRunner{}

	count1 := 0
	r.RunUtil(context.Background(), 2*time.Second, 300*time.Millisecond, func(ctx context.Context) error {
		count1++
		return fmt.Errorf("fake failed")
	})

	time.Sleep(3 * time.Second)
	count2 := 0
	r.RunUtil(context.Background(), 2*time.Second, 300*time.Millisecond, func(ctx context.Context) error {
		count2++
		if count2 >= 3 {
			return nil
		}
		return fmt.Errorf("fake failed")
	})

	time.Sleep(3 * time.Second)
	if count1 != 8 {
		t.Errorf("expect call %d times, but call %d times", 8, count1)
	}
	if count2 != 3 {
		t.Errorf("expect call %d times, but call %d times", 3, count2)
	}
}
