#!/usr/bin/python
# -*- coding:utf-8 -*-

import subprocess
import time
import sys
import re
import argparse
import logging
import shlex

LOGFILE = "/var/log/clean_docker_images.log"
IMAGE_COMMAND_TEMPLATE = "docker image ls -a -q"
IMAGE_GREP_COMMAND_TEMPLATE = "docker image ls -a | grep %s"
IMAGE_CREATED_INFO_COMMAND_TEMPLATE = "docker inspect %s --format='{{ .Created }}'"
IMAGE_REMOVE_COMMAND_TEMPLATE = "docker rmi %s"


class CleanDockerImagesError(Exception):
    """Custom exception for clean docker images errors"""
    pass


class Logger:
    def __init__(self, logger_name):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        logfile = logger_name
        fh = logging.FileHandler(logfile, mode='a')
        fh.setLevel(logging.INFO)  # 用于写到file的等级开关
        # 第三步，再创建一个handler,用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # 输出到console的log等级的开关
        # 第四步，定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # 第五步，将logger添加到handler里面
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def get_log(self):
        return self.logger


# Initialize logger
logger = Logger(LOGFILE).get_log()


class Cmd(object):
    def __init__(self):
        self._return_code = None
        self._message = None

    def run(self, cmd_):
        if not isinstance(cmd_, str):
            msg = 'cmd is not available! expect Union[bytes, str] not %s' % type(cmd_)
            logger.error(msg)
            raise CleanDockerImagesError(msg)
        cmd_list = shlex.split(cmd_)
        result = subprocess.Popen(cmd_list, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result.wait()
        self._return_code = result.returncode
        self._message = result.stdout.read()
        # self._return_code, self._message = subprocess.getstatusoutput(cmd_)
        # if self._return_code != 0:
        #     msg = 'execute cmd error! error: %s' % self._message
        #     _print(msg)
        #     sys.exit(1)
        logger.info('execute cmd: %s success' % cmd_)

    @property
    def return_code(self):
        return self._return_code

    @property
    def message(self):
        if isinstance(self._message, bytes):
            return self._message.decode(self, encoding='utf-8')
        return self._message


cmd = Cmd()


def command_rmi(remove_ids):
    for remove_id in remove_ids:
        # Replace shell pipe with Python filtering for SCA compliance
        cmd.run("docker image ls -a")
        if cmd.return_code == 0:
            # Filter lines containing remove_id in Python instead of using grep
            output_lines = cmd.message.split('\n')
            matching_lines = [line for line in output_lines if remove_id in line]
            if matching_lines:
                cmd.run(IMAGE_REMOVE_COMMAND_TEMPLATE % remove_id)
                if cmd.return_code == 0:
                    logger.info("%s" % cmd.message)


def transfer_timedelta(timedelta):
    if not re.match('^\d', timedelta):
        msg = "error format ! timedelta: %s " % timedelta
        logger.error(msg)
        raise CleanDockerImagesError(msg)
    if timedelta.endswith('S'):
        return int(timedelta.split('S')[0])
    elif timedelta.endswith('M'):
        return int(timedelta.split('M')[0]) * 60
    elif timedelta.endswith('H'):
        return int(timedelta.split('H')[0]) * 60 * 60
    elif timedelta.endswith('d'):
        return int(timedelta.split('d')[0]) * 24 * 60 * 60
    elif timedelta.endswith('m'):
        return int(timedelta.split('m')[0]) * 30 * 24 * 60 * 60
    elif timedelta.endswith('y'):
        return int(timedelta.split('y')[0]) * 12 * 30 * 24 * 60 * 60
    else:
        msg = "error format ! timedelta: %s " % timedelta
        logger.error(msg)
        raise CleanDockerImagesError(msg)


class DockerImage(object):
    def __init__(self, ignore_ids=None):
        if ignore_ids is None:
            ignore_ids = []
        if not isinstance(ignore_ids, list):
            msg = "[ERROR] expect list!"
            logger.error(msg)
            raise CleanDockerImagesError(msg)
        self.ignore_ids = ignore_ids
        self.image_ids = []
        self.remove_ids = []
        self._get_image_ids()

    def _get_image_ids(self):
        cmd.run(IMAGE_COMMAND_TEMPLATE)
        if cmd.return_code != 0:
            msg = ('[ERROR] can not get docker image info! cmd: %s . '
                   'please check docker status on your host' % IMAGE_COMMAND_TEMPLATE)
            logger.error(msg)
            raise CleanDockerImagesError(msg)
        self.image_ids = cmd.message.split('\n')[:-1]
        return self.image_ids

    def _check(self):
        pass


class DataParser(object):
    def parse(self, docker_image, timedelta):
        remove_ids = []
        image_id_string = ' '.join(docker_image.image_ids)
        timedelta = transfer_timedelta(timedelta)

        created_timestamp_list = self._get_created_timestamp_list(image_id_string)
        if len(docker_image.image_ids) != len(created_timestamp_list):
            msg = ('Data inconsistency: image_ids count (%d) does not match '
                   'created_timestamp_list count (%d)' % (
                       len(docker_image.image_ids), len(created_timestamp_list)))
            logger.error(msg)
            raise CleanDockerImagesError(msg)
        for image_id, created_timestamp in zip(docker_image.image_ids,
                                               created_timestamp_list):
            if image_id in docker_image.ignore_ids:
                continue
            if created_timestamp <= time.time() - timedelta:
                remove_ids.append(image_id)
        docker_image.remove_ids = remove_ids

    def _get_created_timestamp_list(self, image_id_string):
        cmd.run(IMAGE_CREATED_INFO_COMMAND_TEMPLATE % image_id_string)
        if cmd.return_code != 0:
            msg = ('[ERROR] can not get docker image id  info! cmd: %s . '
                   'please check docker status on your host' % IMAGE_CREATED_INFO_COMMAND_TEMPLATE)
            logger.error(msg)
            raise CleanDockerImagesError(msg)
        return [self._transfer_timestring_to_timestamp(x) for x in cmd.message.split('\n')[:-1]]

    def _transfer_timestring_to_timestamp(self, timestring):
        return time.mktime(time.strptime(timestring.split(r'.')[0], "%Y-%m-%dT%H:%M:%S"))


data_parser = DataParser()
if __name__ == '__main__':
    try:
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("-t", "--timedelta", default='3d', help="timedelta example 1S, 1M, 1H, 1d, 1m, 1y")
        arg_parser.add_argument("-g", "--ignore", nargs='*', help="ids in ignore_id_list will not be deleted")
        args = arg_parser.parse_args()

        docker_image = DockerImage(ignore_ids=args.ignore)
        data_parser.parse(docker_image, timedelta=args.timedelta)
        if len(docker_image.remove_ids) == 0:
            logger.info("finish clean")
        command_rmi(docker_image.remove_ids)
    except CleanDockerImagesError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s" % str(e))
        sys.exit(1)
