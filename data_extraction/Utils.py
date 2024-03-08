def concat_component_url(origin_url: str, components: list[str]):
    """
    拼接 components 到 url 中

    :param origin_url: 原来的 url
    :param components: 组件列表
    :return: 拼接好的 url
    """
    component_str = ''
    for component in components:
        component_str += '&component=' + component
    new_url = origin_url.replace('&limit', component_str + '&limit')
    return new_url
