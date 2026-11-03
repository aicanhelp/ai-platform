package v1beta1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

func RemoveCondition(cond []*Condition, con *Condition) []*Condition {
	if con == nil {
		return cond
	}
	c := GetCondition(cond, con)
	if c == nil {
		return cond
	}
	newCond := []*Condition{}
	for _, c1 := range cond {
		if c1.ID == con.ID {
			continue
		}
		newCond = append(newCond, c1)
	}
	return newCond
}

func GetCondition(cond []*Condition, con *Condition) *Condition {
	if con == nil || len(cond) == 0 {
		return nil
	}
	for _, c := range cond {
		if c.ID == con.ID {
			return c
		}
	}
	return nil
}

func ReplaceCondition(cond []*Condition, con *Condition) []*Condition {
	if con == nil {
		return cond
	}
	c := GetCondition(cond, con)
	if c == nil {
		cond = append(cond, con)
		return cond
	}
	newCond := []*Condition{}
	for _, c1 := range cond {
		if c1.ID == con.ID {
			newCond = append(newCond, con)
			continue
		}
		newCond = append(newCond, c1)
	}
	return newCond
}

func ConditionCount(cond []*Condition, commandCount int) (succeeded, failed int, status metav1.ConditionStatus, phase CommandPhase) {
	if len(cond) == 0 {
		return 0, 0, metav1.ConditionUnknown, CommandRunning
	}
	for _, c := range cond {
		if c.Status == metav1.ConditionFalse {
			failed++
			continue
		}
		succeeded++
	}

	latestCond := cond[len(cond)-1]
	if len(cond) < commandCount {
		status = metav1.ConditionUnknown
	} else if latestCond.Status == metav1.ConditionTrue {
		status = metav1.ConditionTrue
	} else if latestCond.Status == metav1.ConditionFalse {
		status = metav1.ConditionFalse
	} else {
		status = metav1.ConditionUnknown
	}
	phase = CommandUnKnown
	if latestCond.Phase == CommandFailed {
		phase = CommandFailed
		status = metav1.ConditionFalse
	}
	if len(cond) < commandCount {
		if latestCond.Phase == CommandRunning || latestCond.Phase == CommandSkip || latestCond.Phase == CommandComplete {
			phase = CommandRunning
		}
	}
	if len(cond) == commandCount {
		if latestCond.Phase == CommandRunning {
			phase = CommandRunning
		}
		if latestCond.Phase == CommandSkip || latestCond.Phase == CommandComplete {
			phase = CommandComplete
		}
	}
	return succeeded, failed, status, phase
}
