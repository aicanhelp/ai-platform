package errors

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTokenExpiredErrorError(t *testing.T) {
	errWithEmptyMsg := &TokenExpiredError{Message: ""}
	assert.Equal(t, "", errWithEmptyMsg.Error())

	errWithMsg := &TokenExpiredError{Message: "any"}
	assert.Equal(t, "token has expired", errWithMsg.Error())
}

func TestCommonHttpErrorTypes(t *testing.T) {
	httpErr := &HttpResponseNotOKError{Message: "status not ok"}
	assert.Equal(t, "status not ok", httpErr.Error())

	jsonErr := &InvalidJsonHttpBodyError{Message: "bad json"}
	assert.Equal(t, "bad json", jsonErr.Error())

	fieldErr := &FieldNotFoundError{Message: "missing field", Field: "name"}
	assert.Equal(t, "missing field", fieldErr.Error())
}

