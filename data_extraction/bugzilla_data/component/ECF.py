url = 'https://bugs.eclipse.org/bugs/buglist.cgi?bug_status=CLOSED&bug_status=RESOLVED&bug_status=VERIFIED&limit=0' \
      '&product=ECF&query_format=advanced&resolution=FIXED'

name = 'ECF'

total_bugs = 1634  # 538

index = 0

components = [
    'ecf.cola',
    'ecf.core',
    'ecf.data',
    'ecf.disc',
    'ecf.doc',
    'ecf.exam',
    'ecf.file',
    'ecf.news',
    'ecf.pres',
    'ecf.prot',
    'ecf.prov',
    'ecf.rele',
    'ecf.remo',
    'ecf.serv',
    'ecf.tele',
    'ecf.test',
    'ecf.tool',
    'ecf.twit',
    'ecf.ui'
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


