#!/usr/bin/env bash

cur_dir=`pwd`

function do_del() {
  local MOD_NAME=${1:?'[ERROR]argument name required!'}

  echo "deinit ${MOD_NAME} "
  # 逆初始化模块，其中{MOD_NAME}为模块目录，执行后可发现模块目录被清空
  git submodule deinit ${MOD_NAME}

  echo "remove ${MOD_NAME}"
  # 删除.gitmodules中记录的模块信息（--cached选项清除.git/modules中的缓存）
  git rm -f --cached ${MOD_NAME}

  rm -rf .git/module/${MOD_NAME}
  rm -rf ${MOD_NAME}

  # 提交更改到代码库，可观察到'.gitmodules'内容发生变更
  git commit -am "Remove a submodule ${MOD_NAME}"
  git push
}

function do_add() {
   local repo=${1:?'[ERROR]argument repo required!'}
   local type=${2:?'[ERROR]argument type required!'}
   local name=$3

   test -z $name && name=`echo ${repo} | sed 's/.*\///' | sed 's/\..*//'`
   git submodule add ${repo} ${type}-projects/$name
}

function do_list() {
    git submodule|awk '{print $2}'
}

function do_update(){
  local type=${1:?'[ERROR]argument type required!'}
  local name=${2:?'[ERROR]argument name required!'}

  MOD_NAME=$1-projects/$2
  cd ${MOD_NAME}
  git pull
  cd ${cur_dir}
}

function do_update_all(){
   git submodule update --recursive
}

function do_mv(){
  local src=${1:?'[ERROR]argument src required!'}
  local type=${2:?'[ERROR]argument target_type required!'}

  local name=`echo ${src} | sed 's/.*\///'`

  MOD_NAME=$type-projects/${name}
  mkdir -p ${type}-projects
  git mv ${src} ${MOD_NAME}
}

test -z $1 && echo "Please input action: del add list update mv upate_all" && exit 0

do_$1 $2 $3 $4 $5
