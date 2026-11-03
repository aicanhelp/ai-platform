package performance_test

// ==================== 性能测试常量 ====================

// BKE 二进制文件下载指令
// 示例："curl -o bke.tar.gz https://example.com/bke.tar.gz"
const BKE_DOWNLOAD_COMMAND = "curl -sfL https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh | bash"

// BKE 初始化指令
const BKE_INIT_COMMAND = "bke init --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest --confirm"
