# -*- coding: utf-8 -*-

from __future__ import print_function
import json
import os
import struct
import sys
import platform
import re
import time
import traceback
import requests
import socket


def get_ip():
    """
    获取本机IP
    :return:
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.223.30.51', 53))
        ip = s.getsockname()[0]
    except Exception as e:
        ip = socket.gethostbyname(socket.gethostname())
    finally:
        s.close()
    return ip

report_ip = os.environ.get('CHIEF_IP', '')
if not report_ip:
    report_ip = get_ip()


def report_progress(progress):
    """
    向worker上报训练进度
    :return:
    """
    url = "http://%s:%s/v1/worker/report-progress" % (report_ip, 8080)
    try:
        response = requests.post(url, json=json.dumps(progress))
    except Exception as e:
        print("send progress info to worker failed!\nprogress_info: %s, \n%s" % (progress, traceback.format_exc()))
        return False, str(e)
    if response.status_code != 200:
        print("send progress info to worker failed!\nprogress_info: %s, \nreason: %s" % (progress, response.reason))
        return False, response.text
    return True, ""


def report_error(code, msg=""):
    """
    向worker上报错误信息
    :param code: 错误码
    :param msg: 出错原因
    :return:
    """
    progress = {"type": "error", "code": code, "msg": msg}
    return report_progress(progress)


def job_completed():
    """
    通知worker，训练任务结束
    :return:
    """
    progress = {"type": "completed"}
    return report_progress(progress)


if __name__ == "__main__":
  ## 不同上报的测试样例
    ##1 测试上报心跳
    for i in range(3):
        # heartbeat()
        print("yes")
    ## 2 上报训练进度
    progress = {"step": 0, "type": "train", "loss": 1.5}
    for i in range(3):
        time.sleep(3*60)
        ret, msg = report_progress(progress)
        print("ret: %s, %s" % (ret, msg))
    
    ## 3 上报测试进度
    progress = {"step": 0, "type": "test", "accracy": 0.3}
    for i in range(3):
        time.sleep(3*60)
        ret, msg = report_progress(progress)
        print("ret: %s, %s" % (ret, msg))
    
    ## 4 上报结束信息
    job_completed()
