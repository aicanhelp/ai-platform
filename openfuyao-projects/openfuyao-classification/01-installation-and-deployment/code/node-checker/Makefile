# 构建OsChecktool
.PHONY: oschecktool
oschecktool:
	CGO_ENABLED=0 GOOS=linux  go build -trimpath -buildmode=pie -ldflags="-w -s" -o oschecktool .
	
# 清理
.PHONY: clean
clean:
	rm -f oschecktool

# 测试
.PHONY: test
test:
	go test ./...

