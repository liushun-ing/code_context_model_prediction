from bugzilla import Bugzilla
from bugzilla.bug import Bug

from data_extraction.step3 import make_dir


def get_att_ids_and_patch_ids(bug: Bug):
    """
    获取 attachment 和 patch id 集合

    :param bug: bug对象
    :return: att_ids, patch_ids，两个 id 集合
    """
    # attachment of patch (ids)
    patch_ids = []
    # mylyn.zip ids
    att_ids = []
    attachments = bug.get_attachments()
    for att in attachments:
        if 'mylyn' in att['summary'] and 'zip' in att['summary']:
            att_ids.append(att['id'])
        elif 'mylar' in att['summary'] and 'zip' in att['summary']:
            att_ids.append(att['id'])
        elif att['is_patch'] == 1:
            patch_ids.append(att['id'])
    return att_ids, patch_ids


def save_patch(bz_api: Bugzilla, bug: Bug, output_dir: str, patch_ids: list):
    """
    写文件，保存 patch 匹配的

    :param bz_api: Bugzilla对象
    :param bug: bug对象
    :param output_dir: 输出目录
    :param patch_ids: patchId集合
    :return: 无返回值，写文件
    """
    bug_dir = output_dir + "/" + str(bug.id)
    make_dir(bug_dir)
    for att_id in patch_ids:
        attach = bz_api.openattachment(att_id)
        att_name = bug_dir + '/' + str(bug.id) + '_' + str(att_id) + '.txt'
        with open(att_name, 'wb') as out:
            out.write(attach.read())


def filter_br_with_patch(bz_api: Bugzilla, component_bugs: dict[str, list[Bug]]):
    """
    step5: Filter BR with patch

    :return: 无返回值
    """
    file_dir = '0_paper_dataset/'
    output_dir = file_dir + 'patch_txt'
    make_dir(output_dir)
    patched_context_bug_ids = []
    context_bug_ids = []
    count = 0
    # bug_id = 263919
    component: str
    fixed_bugs: list[Bug]
    for component, fixed_bugs in sorted(component_bugs.items()):
        print('component = {}, count = {}'.format(component, count))
        for bug in fixed_bugs:
            # if bug.id < bug_id:
            #     continue
            att_ids, patch_ids = get_att_ids_and_patch_ids(bug)
            if (len(att_ids)) > 0:
                count += 1
                if (len(patch_ids)) > 0:
                    print('Patch & mylyn: %d' % bug.id)
                    patched_context_bug_ids.append(bug.id)
                    save_patch(bz_api, bug, output_dir, patch_ids)
                else:
                    print('    Only mylyn, no patch: %d' % bug.id)
                    context_bug_ids.append(bug.id)
            else:
                print('               No mylyn: %d' % bug.id)
