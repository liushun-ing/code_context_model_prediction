from os.path import join

import paramiko
import os
from stat import S_ISDIR as isdir


def down_from_remote(sftp_obj, remote_dir_name, local_dir_name):
    """远程下载文件"""
    remote_file = sftp_obj.stat(remote_dir_name)
    if isdir(remote_file.st_mode):
        # 文件夹，不能直接下载，需要继续循环
        check_local_dir(local_dir_name)
        for remote_file_name in sftp.listdir(remote_dir_name):
            sub_remote = os.path.join(remote_dir_name, remote_file_name)
            sub_remote = sub_remote.replace('\\', '/')
            sub_local = os.path.join(local_dir_name, remote_file_name)
            sub_local = sub_local.replace('\\', '/')
            down_from_remote(sftp_obj, sub_remote, sub_local)
    else:
        sftp.get(remote_dir_name, local_dir_name)


def check_local_dir(local_dir_name):
    """本地文件夹是否存在，不存在则创建"""
    if not os.path.exists(local_dir_name):
        os.makedirs(local_dir_name)


if __name__ == "__main__":
    """程序主入口"""
    # 服务器连接信息
    host_name = '115.236.22.158'
    user_name = 'shunliu'
    password = '@@010311'
    port = 9997
    # 远程文件路径（需要绝对路径）
    remote_root = '/data0/shunliu/pythonfile/code_context_model_prediction/params_validation/git_repo_code/my_mylyn/repo_first_3/'
    # 本地文件存放路径（绝对路径或者相对路径都可以）
    local_root = 'D:/git_code/model_code/'
    # 连接远程服务器
    t = paramiko.Transport((host_name, port))
    t.connect(username=user_name, password=password)
    sftp = paramiko.SFTPClient.from_transport(t)
    need_to_download = sftp.listdir(remote_root)
    need_to_download.sort(key=lambda x: int(x))
    index = need_to_download.index('5368')
    for need in need_to_download[index:]:
        print('downloading ', need)
        remote_down = remote_root + need
        local_down = join(local_root, need)
        # remote_down_dir = sftp.stat(remote_down)
        for remote_dir in sftp.listdir(remote_down):
            if 'doxygen' in remote_dir:
                continue
            remote_file = sftp.stat(remote_down + "/" + remote_dir)
            if not isdir(remote_file.st_mode):
                continue
            repo_remote = join(remote_down, remote_dir)
            repo_remote = repo_remote.replace('\\', '/')
            repo_local = join(local_down, remote_dir)
            check_local_dir(repo_local)
            # 远程文件开始下载
            down_from_remote(sftp, repo_remote, repo_local)
    # 关闭连接
    t.close()


