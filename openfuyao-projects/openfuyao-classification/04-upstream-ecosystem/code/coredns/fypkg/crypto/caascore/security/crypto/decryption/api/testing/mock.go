// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package testing

type Service struct {
	IsRelease bool
}

func (*Service) DecryptSecret(data []byte) ([]byte, error) {
	result := "[DecryptSecret]" + string(data)
	return []byte(result), nil
}

func (*Service) DecryptCert(data []byte) ([]byte, error) {
	result := "[DecryptCert]" + string(data)
	return []byte(result), nil
}

func (s *Service) Release() {
	if s.IsRelease {
		panic("release more than once")
	}
	s.IsRelease = true
}
