import time

from bugzilla import Bugzilla
from bugzilla.bug import Bug

from data_extraction.step3 import make_dir
from data_extraction.step4 import get_att_ids_and_patch_ids


def save_mylyn_and_patch(bz_api: Bugzilla, bug: Bug, output_dir_3: str, output_dir_4: str, att_ids: list,
                         patch_ids: list, patch: bool):
    """
    写文件，保存符合 mylyn 或者 patch 的数据

    :param bz_api: Bugzilla对象
    :param bug: Bug对象
    :param output_dir_3: mylyn的输出目录
    :param output_dir_4: patch的输出目录
    :param att_ids: 符合mylyn的bugId集合
    :param patch_ids: 符合patch的bugId集合
    :param patch: 是否需要保存 patch
    :return: 写文件，没有返回值
    """
    bug_dir_3 = output_dir_3 + "/" + str(bug.id)
    make_dir(bug_dir_3)
    patch_ids_copy = patch_ids.copy()
    bug_dir_4 = output_dir_4 + "/" + str(bug.id)
    if patch:
        make_dir(bug_dir_4)
    for att_id in att_ids:
        attach = bz_api.openattachment(att_id)
        att_name = bug_dir_3 + '/' + str(bug.id) + '_' + str(att_id) + '.zip'
        with open(att_name, 'wb') as out:
            out.write(attach.read())
        if patch:
            # 如果是存在的，就不重复获取了，直接使用现成的
            if att_id in patch_ids_copy:
                att_name_patch = bug_dir_4 + '/' + str(bug.id) + '_' + str(att_id) + '.txt'
                with open(att_name_patch, 'wb') as out_patch:
                    out_patch.write(attach.read())
                patch_ids_copy.remove(att_id)
    # 再遍历剩下的
    if patch:
        for att_id in patch_ids_copy:
            attach = bz_api.openattachment(att_id)
            att_name = bug_dir_4 + '/' + str(bug.id) + '_' + str(att_id) + '.txt'
            with open(att_name, 'wb') as out:
                out.write(attach.read())


def merged_3_and_4(bz_api: Bugzilla, component_bugs: dict[str, list[Bug]], product: str, interval: int):
    """
    step3: filter by existence of mylyn attachment
    根据 bug id和标签 id 获取 zip 内容，并保存到本地文件

    :return: 无返回值，写文件
    """
    count = 1
    has_att = 0
    file_dir = '2023_dataset/'
    output_dir_3 = file_dir + 'mylyn_zip/' + product
    make_dir(output_dir_3)

    output_dir_4 = file_dir + 'patch_txt/' + product
    make_dir(output_dir_4)
    # patched_context_bug_ids = []
    # context_bug_ids = []

    mylyn_zip_bugs = list()
    component: str
    fixed_bugs: list[Bug]
    for component, fixed_bugs in component_bugs.items():
        if len(fixed_bugs) <= 0:
            continue
        print(component, len(fixed_bugs))
        print("Till now, mylyn zip #:" + str(has_att))
        i = interval
        while i < len(fixed_bugs):  # 可以在这限制处理的bug区间
            bug = fixed_bugs[i]
            if count % 10 == 0:
                print('Total: {}, current bug ID: {}, index: {}'.format(count, bug.id, fixed_bugs.index(bug)))
            #         print(bug)
            #         print(bug.assigned_to)
            #         print("Now processing bug ID:" + str(bug.id))
            count = count + 1
            try:
                att_ids, patch_ids = get_att_ids_and_patch_ids(bug)
                if (len(att_ids)) > 0:
                    # mylyn存在，保存进文件
                    mylyn_zip_bugs.append(bug)
                    has_att = has_att + 1
                    if (len(patch_ids)) > 0:
                        print('Patch & mylyn: %d' % bug.id)
                        # patched_context_bug_ids.append(bug.id)
                        save_mylyn_and_patch(bz_api, bug, output_dir_3, output_dir_4, att_ids, patch_ids, True)
                    else:
                        print('    Only mylyn, no patch: %d' % bug.id)
                        save_mylyn_and_patch(bz_api, bug, output_dir_3, output_dir_4, att_ids, patch_ids, False)
                        # context_bug_ids.append(bug.id)
                else:
                    print('               No mylyn: %d' % bug.id)
            except Exception as error:
                print(bug.id, "index:", i, "出错啦，间隔二十秒之后，重新请求")
                time.sleep(20)
                i -= 1
            finally:
                i += 1
