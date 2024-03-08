url = 'https://bugs.eclipse.org/bugs/buglist.cgi?bug_status=CLOSED&bug_status=RESOLVED&bug_status=VERIFIED&limit=0' \
      '&product=Platform&query_format=advanced&resolution=FIXED'

name = 'Platform'

total_bugs = 53882  # 47923-5

index = 0

components = [
    'Ant',
    'Compare',
    'CVS',
    'Debug',
    'Doc',
    'IDE',
    'Incubato',
    'PMC',
    'Releng',
    'Resource',
    'Runtime',
    'Scriptin',
    'Search',
    'SWT',
    'Team',
    'Text',
    'UI',
    'Update',
    'User Ass',
    'WebDAV',
    'Website'
]


def add_index():
    global index
    if index >= len(components):
        index = 0
    else:
        index += 1


def set_index(new_index: int):
    global index
    if new_index >= len(components):
        raise Exception('index over')
    else:
        index = new_index
