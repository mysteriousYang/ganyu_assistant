#-*- encoding utf-8 -*-
import lxml.html
import requests
import json
import os
import lxml
import shutil

proxies = {
    "http":"127.0.0.1:7890",
    "https":"127.0.0.1:7890"
}

headers = {
    'accept':'text/html',
    'accept-encoding':'gzip',
    'accept-language':'zh-CN,zh;q=0.9,en;q=0.8',
    'cache-control':'max-age=0',
    # 'cookie':'wikia_beacon_id=YOEhSC1vFW; _b2=i-SMheEF_3.1705396132015; Geo={%22region%22:%22NO REGION%22%2C%22city%22:%22wong tai sin%22%2C%22country_name%22:%22hong kong%22%2C%22country%22:%22HK%22%2C%22continent%22:%22AS%22}; fandom_global_id=27cdda04-8247-4e69-b046-d65adbfe3fc4; _pubcid=2e08abb4-b5ea-4963-b200-9a41e1ea366d; _au_1d=AU1D-0100-001709386642-XVW9NLB3-AWWA; _ga=GA1.1.1299219373.1709388927; wikia_session_id=ne5ST5c_CU; _cc_id=2587324a1b5699a3c053a0e284cbd4e; ac_cclang=; eb=80; basset=icFeaturedVideoPlayer-0_B_25:true|icConnatixPlayer-0_A_25:false; _ga_LVKNCJXRLW=GS1.1.1720835707.58.1.1720835707.0.0.0; _ga_LFNSP5H47X=GS1.1.1720835709.53.0.1720835709.0.0.0; fan_visited_wikis=2025468,31618,1460983,1542703; _pubcid_cst=kSylLAssaw%3D%3D; _ce.irv=new; cebs=1; _ga_FVWZ0RM4DH=GS1.1.1726759912.28.0.1726759912.60.0.0; exp_bucket=79; active_cms_notification=454; exp_bucket_2=v2-99; AMP_MKTG_6765a55f49=JTdCJTIycmVmZXJyZXIlMjIlM0ElMjJodHRwcyUzQSUyRiUyRnd3dy5nb29nbGUuY29tLmhrJTJGJTIyJTJDJTIycmVmZXJyaW5nX2RvbWFpbiUyMiUzQSUyMnd3dy5nb29nbGUuY29tLmhrJTIyJTdE; _ce.clock_data=337%2C38.150.11.187%2C1%2Cf51bb482c660d0eeadd1f058058a2b35%2CChrome%2CJP; cebsp_=12; AMP_6765a55f49=JTdCJTIyZGV2aWNlSWQlMjIlM0ElMjJiODk1MzNkMy0zZWM4LTQxNWItYjQ1My00ZTRkODI0MDlhYWMlMjIlMkMlMjJzZXNzaW9uSWQlMjIlM0ExNzM1MTczOTUyMTI3JTJDJTIyb3B0T3V0JTIyJTNBZmFsc2UlMkMlMjJsYXN0RXZlbnRUaW1lJTIyJTNBMTczNTE3Mzk3MzI3NiUyQyUyMmxhc3RFdmVudElkJTIyJTNBNTQ5JTdE; _ce.s=v~79ad92d921056dea60ec60ea77c0e02f2609ee28~lcw~1735183388782~lva~1735137746414~vpv~0~v11.fhb~1735173953355~v11.lhb~1735183388566~vir~returning~v11.cs~362001~v11.s~c2a396e0-c322-11ef-9bec-2d4a9201b55b~v11.sla~1735174858995~gtrk.la~m54rdzla~v11.send~1735183392492~lcw~1735183392492',
    'priority':'u=0, i',
    'referer':'https://genshin-impact.fandom.com/wiki/Character_and_Weapon_Enhancement_Material',
    'sec-ch-ua':'"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    'sec-ch-ua-mobile':'?0',
    'sec-ch-ua-platform':'"Windows"',
    'sec-fetch-dest':'document',
    'sec-fetch-mode':'navigate',
    'sec-fetch-site':'same-origin',
    'sec-fetch-user':'?1',
    'upgrade-insecure-requests':'1',
    'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
}

fetch_url = r'https://act-api-takumi-static.mihoyo.com/hoyowiki/genshin/wapi/entry_page?app_sn=ys_obc&entry_page_id=1342&lang=zh-cn'
preDFN_file = ".\\collective_item_train\\predefined_classes.txt"

def pull_metadata():
    response = requests.get(url=fetch_url,headers=headers,proxies=proxies).json()
    response:str = response["data"]["page"]["modules"][0]["components"][0]["data"]

    # response = response.encode('utf-8').decode('unicode_escape')
    # print(response)

    response_json = json.loads(response)
    rows = response_json["table"]["row"]

    obj_name = list()

    for row in rows:
        xml_obj = lxml.html.fromstring(row[0])
        obj_name.append(xml_obj.xpath("//p/span/a/span/text()")[0])

        # print(row[0],"\n")
    print(obj_name)
    return obj_name

def update_preDFN(obj_name,dfn_file=preDFN_file):
    exist_items = set()
    if os.path.exists(dfn_file):
        print("Predefined Classes Detected!")
        with open(dfn_file,encoding="utf-8",mode='r') as fp:
            raw_str_list = fp.readlines()
            for line in raw_str_list:
                exist_items.add(line.strip())

    all_objs = set(obj_name)
    update_obj = all_objs - exist_items

    with open(dfn_file,encoding="utf-8",mode='w') as fp:
        for item in update_obj:
            fp.write(item)
            fp.write('\n')
    # shutil.copy(dfn_file,)
    print("Done")

def run():
    names = pull_metadata()
    update_preDFN(names)

if __name__ == "__main__":
    pass