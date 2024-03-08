import os

from bugzilla import Bugzilla
from bugzilla.bug import Bug


def get_mylyn_attachment(bug: Bug):
    """
    获取mylyn标签

    :param bug: bug对象
    :return: 返回mylyn的所有标签id
    """
    att_ids: list[int] = []
    attachments = bug.get_attachments()
    for att in attachments:
        print('att', att)
        if 'mylyn' in att['summary'] and 'zip' in att['summary']:
            att_ids.append(att['id'])
    return att_ids


def make_dir(directory):
    """
    创建一个目录

    :param directory: 目录地址
    :return: 无返回值，创建目录
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_mylyn(bz_api: Bugzilla, bug: Bug, output_dir: str, att_ids: list):
    """
    写文件，保存 mylyn 匹配的

    :param bz_api: Bugzilla对象
    :param bug: bug对象
    :param output_dir: 输出目录
    :param att_ids: 包含attachment的bugId集合
    :return: 无返回值，写文件
    """
    bug_dir = output_dir + "/" + str(bug.id)
    make_dir(bug_dir)
    for att_id in att_ids:
        attach = bz_api.openattachment(att_id)
        att_name = bug_dir + '/' + str(bug.id) + '_' + str(att_id) + '.zip'
        with open(att_name, 'wb') as out:
            out.write(attach.read())


def filter_mylyn_bugs(bz_api: Bugzilla, component_bugs: dict[str, list[Bug]]):
    """
    step3: filter by existence of mylyn attachment
    根据 bug id和标签 id 获取 zip 内容，并保存到本地文件

    :return: 无返回值，写文件
    """
    count = 1
    has_att = 0
    file_dir = '0_paper_dataset/'
    product = 'Platform/'
    output_dir = file_dir + 'mylyn_zip/' + product
    make_dir(output_dir)
    mylyn_zip_bugs = list()
    component: str
    fixed_bugs: list[Bug]
    for component, fixed_bugs in sorted(component_bugs.items()):
        print(component)
        print("Till now, mylyn zip #:" + str(has_att))
        for bug in fixed_bugs:
            if count % 10 == 0:
                print('Total: {}, current bug ID: {}'.format(count, bug.id))
            #         print(bug)
            #         print(bug.assigned_to)
            #         print("Now processing bug ID:" + str(bug.id))
            count = count + 1
            #         if count < 22680:
            #             continue
            att_ids = get_mylyn_attachment(bug)
            # no mylyn attachment
            if len(att_ids) == 0:
                continue
            mylyn_zip_bugs.append(bug)
            has_att = has_att + 1
            save_mylyn(bz_api, bug, output_dir, att_ids)
