url = 'https://bugs.eclipse.org/bugs/buglist.cgi?bug_status=CLOSED&bug_status=RESOLVED&bug_status=VERIFIED&limit=0' \
      '&product=PDE&query_format=advanced&resolution=FIXED'

name = 'PDE'

total_bugs = 9801 # 8372

index = 0

components = [
    'API Tool',
    'Build',
    'Doc',
    'Incubato',
    'UI',
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

