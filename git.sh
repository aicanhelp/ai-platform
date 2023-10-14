#!/usr/bin/env bash

cur_dir=`pwd`

function do_del() {
    MOD_NAME=$1

   echo "deinit ${MOD_NAME} "
  # 逆初始化模块，其中{MOD_NAME}为模块目录，执行后可发现模块目录被清空
  git submodule deinit resources/${MOD_NAME}

  echo "remove ${MOD_NAME} "
  # 删除.gitmodules中记录的模块信息（--cached选项清除.git/modules中的缓存）
  git rm -f --cached resources/${MOD_NAME}

  rm -rf .git/module/resources/${MOD_NAME}
  rm -rf resources/${MOD_NAME}

  # 提交更改到代码库，可观察到'.gitmodules'内容发生变更
  git commit -am "Remove a submodule resources/${MOD_NAME}"
  git push
}

function do_add() {
    git submodule add $1 resources/$2
}

function do_list() {
    git submodule|awk '{print $2}'|sed 's/resources\///'
}

function do_update(){
   cd resources/$1
   git pull
   cd ${cur_dir}
}

function do_update_all(){
   git submodule update --recursive
}

function do_mv(){
  mkdir -p resources/$2
  git mv resources/$1 resources/$2/$1
}

do_$1 $2 $3