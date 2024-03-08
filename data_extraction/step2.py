from bugzilla import Bugzilla


def get_all_bugs(bz_api: Bugzilla, query, components: list, index: int):
    """
    Step 2: initialize the output

    :param index: 当前处理的组件的下标
    :param bz_api: bugzilla对象
    :param query: query对象
    :param components: 所有的components
    :return: component_bugs: 返回components和它的bugs集合
    """
    component_bugs = dict()
    for component in components:
        component_bugs[component] = list()
    # query['component'] = components
    # print(len(bz_api.query(query)))

    # 每次爬取一个
    for component in components[index:index + 1]:
        component_query = query.copy()
        component_query['component'] = component
        print('component_query', component_query)
        bugs = bz_api.query(component_query)
        print('single component', component, len(bugs), bugs)
        component_bugs[component].extend(bugs)

    # for component, bugs in component_bugs.items():
    #     print('{} has {} bugs'.format(component, len(bugs)))

    return component_bugs


"""
Ant has 29 bugs
CVS has 21 bugs
Compare has 8 bugs
Debug has 46 bugs
Doc has 26 bugs
IDE has 41 bugs
Incubator has 0 bugs
PMC has 4 bugs
Releng has 205 bugs
Resources has 25 bugs
Runtime has 17 bugs
SWT has 356 bugs
Scripting has 0 bugs
Search has 15 bugs
Team has 76 bugs
Text has 51 bugs
UI has 1335 bugs
Update  (deprecated - use Eclipse>Equinox>p2) has 20 bugs
User Assistance has 60 bugs
WebDAV has 1 bugs
Website has 5 bugs
"""
