url = 'https://bugs.eclipse.org/bugs/buglist.cgi?bug_status=CLOSED&bug_status=RESOLVED&bug_status=VERIFIED&limit=0' \
      '&product=MDT&query_format=advanced&resolution=FIXED'

name = 'MDT'

total_bugs = 79 # 79

index = 0

components = [
    'Releng',
    'Website',
    'XSD'
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
