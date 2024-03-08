from dateutil.parser import parse
from datetime import timezone

rfc_timezones = {
    'CET': '+1:00',
    'CDT': '-5:00',
    'CEST': '+2:00',
    'COT': '-5:00',
    'PET': '-5:00',
    'IST': '+5:30',
    'CST': '-6:00',
    'BST': '+1:00',
    'GMT': '+0:00',
    'PDT': '-7:00',
    'PST': '-8:00',
    'EET': '+2:00',
    'EDT': '-4:00',
    'MDT': '-6:00',
    'EST': '-5:00',
    'MST': '-7:00',
    'BRT': '-3:00',
    'EEST': '+3:00',
    'BRST': '-2:00',
    'JST': '+9:00',
    'YAKT': '+9:00',
    'ICT': '+7:00'
}


def get_common_time(time_str: str):
    time_list = time_str.split(' ')
    i = time_list[2].find('+')
    if i != -1:
        time_list[2] = time_list[2].replace(time_list[2][i:], '')
    i = time_list[2].find('-')
    if i != -1:
        time_list[2] = time_list[2].replace(time_list[2][i:], '')
    offset = rfc_timezones.get(time_list[2])
    time_list[2] = offset
    res = parse(' '.join(time_list))
    return res.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
