from data_extraction import step1, step2, step_merge
from data_extraction.bugzilla_data.component import Mylyn

if __name__ == '__main__':
    # 0:2006 1:806 2:1197 3:3605 4:705 5:700 7:86 8:4977(2565) 10:1149 12:537 13:10467(6804，7623，9536)
    # 14:3479 15:2135 16:15989(15791) 19:24 20:61
    # index = 21
    # while True:
    #     Platform.set_index(index)
    #     interval = 0
    #     url = concat_component_url(Platform.url, Platform.components)
    #     bz_api, query = step1.get_all_components(url)
    #     component_bugs = step2.get_all_bugs(bz_api, query, Platform.components, Platform.index)
    #     step_merge.merged_3_and_4(bz_api, component_bugs, Platform.name, interval)
    #     index += 1

    # 1:1008 2:251 4:7113
    # index = 0
    # while True:
    #     PDE.set_index(index)
    #     interval = 0
    #     url = concat_component_url(PDE.url, PDE.components)
    #     bz_api, query = step1.get_all_components(PDE.url)
    #     component_bugs = step2.get_all_bugs(bz_api, query, PDE.components, PDE.index)
    #     step_merge.merged_3_and_4(bz_api, component_bugs, PDE.name, interval)
    #     index += 1

    # 0:18 1:335 4:33 15:47 18:105
    # index = 0
    # while True:
    #     ECF.set_index(19)
    #     interval = 0
    #     url = concat_component_url(ECF.url, ECF.components)
    #     bz_api, query = step1.get_all_components(ECF.url)
    #     component_bugs = step2.get_all_bugs(bz_api, query, ECF.components, ECF.index)
    #     step_merge.merged_3_and_4(bz_api, component_bugs, ECF.name, interval)
    #     index += 1

    # 0:54 1:24 2:1
    # index = 0
    # while True:
    #     MDT.set_index(3)
    #     interval = 0
    #     url = concat_component_url(MDT.url, MDT.components)
    #     bz_api, query = step1.get_all_components(MDT.url)
    #     component_bugs = step2.get_all_bugs(bz_api, query, MDT.components, MDT.index)
    #     step_merge.merged_3_and_4(bz_api, component_bugs, MDT.name, interval)
    #     index += 1

    # 6221
    bz_api, query = step1.get_all_components(Mylyn.url)
    component_bugs = step2.get_all_bugs(bz_api, query, ['Mylyn'], 0)
    step_merge.merged_3_and_4(bz_api, component_bugs, Mylyn.name, 0)




