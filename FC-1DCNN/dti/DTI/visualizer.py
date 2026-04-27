import logging
import socket

import visdom
import torch
import numpy as np
# 新建一个连接客户端
# 指定env = 'test1'，默认是'main',注意在浏览器界面做环境的切换


LOG = logging.getLogger('visualizer')


class Visualizer:
    def __init__(self, opt):
        self.opt = opt
        self.name = opt.RECORD_NAME
        self.vis = None
        self.enabled = self._is_visdom_server_available()
        if self.enabled:
            self.vis = visdom.Visdom(env='main', use_incoming_socket=False, raise_exceptions=False)
        else:
            LOG.info("Visdom server is not available on localhost:8097, visualization is disabled.")

    def _is_visdom_server_available(self, host='127.0.0.1', port=8097, timeout=0.2):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            return sock.connect_ex((host, port)) == 0
        finally:
            sock.close()

    def display_train_result(self, loss, tacc, vacc, epoch):  # h
        if not self.enabled or self.vis is None:
            return
        self.vis.line(Y=[[loss, tacc, vacc]], X=[epoch], win=self.name,
                      opts=dict(title=self.name, legend=['train_loss', 'train_acc', 'val_acc']),
                      update=None if epoch == 0 else 'append')


class Nothing:
    pass
