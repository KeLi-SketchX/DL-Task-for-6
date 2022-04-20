import logging
import time
import os
class Log(object):
    def __init__(self, loggername=None,loglevel=logging.DEBUG,log_cate='search',file_path=None):

        self.logger = logging.getLogger(loggername)
        self.logger.setLevel(loglevel)
        # 创建一个handler，用于写入日志文件
        self.log_time = time.strftime("%Y-%m-%d-%H-%M")
        file_dir = os.getcwd() + '/./logs'
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        self.log_path = file_dir
        if file_path is None:
            self.log_name = self.log_path + "/" + log_cate + "." + self.log_time + '.log'
        else:
            self.log_name = file_path
        fh = logging.FileHandler(self.log_name, 'a')
        fh.setLevel(loglevel)
        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(loglevel)
        # 定义handler的输出格式
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('[%(levelname)s]%(asctime)s %(filename)s:%(lineno)d: %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        #  添加下面一句，在记录日志之后移除句柄
        # self.logger.removeHandler(ch)
        # self.logger.removeHandler(fh)
        # 关闭打开的文件
        fh.close()
        ch.close()

    def getlog(self):
        return self.logger


