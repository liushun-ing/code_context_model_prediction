import bugzilla


def get_all_components(specific_url: str):
    """
    step1: identify all the components involved URL from eclipse bugzilla_data site

    :return: bz_api, query, components: bugzilla对象，query对象和所有的components
    """
    url = "https://bugs.eclipse.org/bugs/xmlrpc.cgi"
    bz_api = bugzilla.Bugzilla(url)

    query = bz_api.url_to_query(specific_url)
    print('query', query)

    # components = query['component']
    # print('components', components, 'len =', len(components))
    return bz_api, query
