package contentprovider

import (
	"context"
	"os"
	"sync"
	"time"

	"go.etcd.io/etcd/client/pkg/v3/ofpkg/util/timer"
)

var (
	defaultFileWatcher FileWatcher
	lock               sync.Mutex
)

// DefaultCallBackDuration 默认文件变化回调检查间隔
const DefaultCallBackDuration = 30 * time.Second

// ChangedFilesChanSize 文件变化通道缓冲区大小
const ChangedFilesChanSize = 100

// DefaultFileWatcher 获取默认的单例文件监视器（双重检查锁模式）
func DefaultFileWatcher() (FileWatcher, error) {
	if defaultFileWatcher == nil {
		lock.Lock()
		defer lock.Unlock()
		if defaultFileWatcher == nil {
			if watcher, err := CreateAndRunFileWatcher(DefaultCallBackDuration); err != nil {
				return nil, err
			} else {
				defaultFileWatcher = watcher
			}
		}
	}
	return defaultFileWatcher, nil
}

// CreateAndRunFileWatcher 创建并启动文件监视器
func CreateAndRunFileWatcher(callBack time.Duration) (FileWatcher, error) {
	watcher := NewFileWatcher(callBack)
	if err := watcher.Run(); err != nil {
		return nil, err
	}
	return watcher, nil
}

// FileChangedListener 文件变化监听器回调函数类型
type FileChangedListener func()

// FileWatcher should call back multiple changes for same file sequentially
type FileWatcher interface {
	// 添加文件监听器
	AddListener(filePath string, listener FileChangedListener)

	// 启动文件监听器
	Run() error

	// 停止文件监听器
	Stop() error
}

type fileWatcher struct {
	cancelFunc       func()
	callbackDuration time.Duration

	filePaths    map[string][]FileChangedListener
	lastModTime  map[string]int64
	changedFiles chan string

	filePathLock sync.Mutex
}

// NewFileWatcher 创建新的文件监视器实例
func NewFileWatcher(callBack time.Duration) FileWatcher {
	return &fileWatcher{
		filePaths:        map[string][]FileChangedListener{},
		lastModTime:      map[string]int64{},
		changedFiles:     make(chan string, ChangedFilesChanSize),
		callbackDuration: callBack,
	}
}

// AddListener 向指定文件路径添加变化监听器
func (f *fileWatcher) AddListener(filePath string, listener FileChangedListener) {
	f.filePathLock.Lock()
	defer f.filePathLock.Unlock()

	listeners, ok := f.filePaths[filePath]
	if !ok {
		f.filePaths[filePath] = []FileChangedListener{listener}
	} else {
		f.filePaths[filePath] = append(listeners, listener)
	}
}

// Run 启动文件监视器，开始监控文件变化
func (f *fileWatcher) Run() error {
	ctx, cancelFunc := context.WithCancel(context.Background())
	f.cancelFunc = cancelFunc

	go timer.Until(f.watchFile, f.callbackDuration, ctx.Done())
	go timer.Until(func() { f.handleFileChanged(ctx.Done()) }, time.Second, ctx.Done())

	return nil
}

// Stop 停止文件监视器，释放相关资源
func (f *fileWatcher) Stop() error {
	f.cancelFunc()
	return nil
}

func (f *fileWatcher) watchFile() {
	f.filePathLock.Lock()
	defer f.filePathLock.Unlock()

	for filePath := range f.filePaths {
		if f.checkFileChange(filePath) {
			f.changedFiles <- filePath
		}
	}
}

func (f *fileWatcher) checkFileChange(filePath string) bool {
	getFileModTime := func(path string) int64 {
		fi, err := os.Stat(path)
		if err != nil {
			return time.Now().Unix()
		}
		return fi.ModTime().Unix()
	}

	modTimeNew := getFileModTime(filePath)
	if modTime, ok := f.lastModTime[filePath]; !ok || modTime != modTimeNew {
		f.lastModTime[filePath] = modTimeNew
		return true
	}
	return false
}

func (f *fileWatcher) handleFileChanged(stopCh <-chan struct{}) {
	for {
		select {
		case filePath := <-f.changedFiles:
			f.fireFileChangeEventFor(filePath)
		case <-stopCh:
			return
		}
	}
}

func (f *fileWatcher) fireFileChangeEventFor(filePath string) {
	f.filePathLock.Lock()
	defer f.filePathLock.Unlock()

	listeners, ok := f.filePaths[filePath]
	if !ok {
		return
	}
	for _, listener := range listeners {
		listener()
	}
}
