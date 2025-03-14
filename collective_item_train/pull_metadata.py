#-*- encoding utf-8 -*-
import requests
import os

from lxml import html
from lxml import etree
from typing import List

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

url_local_specialty = r"https://genshin-impact.fandom.com/wiki/Local_Specialty"
url_cooking = r"https://genshin-impact.fandom.com/wiki/Cooking_Ingredient"

response_local = requests.get(url=url_local_specialty,headers=headers,proxies=proxies).text
response_cooking = requests.get(url=url_cooking,headers=headers,proxies=proxies).text
# with open(".\\test.html",mode="w",encoding="utf-8") as fp:
#     fp.write(response)

'''
/html/body/div[6]/div[4]/div[4]/main/div[3]/div[2]/div/table[1]

// local_specialty
/html/body/div[6]/div[4]/div[4]/main/div[3]/div[2]/div/table[1]/tbody/tr[1]/td[2]/a

//cooking
/html/body/div[6]/div[4]/div[4]/main/div[3]/div[2]/div/table[1]/tbody/tr[3]/td[2]/a
'''

html_local_obj = etree.HTML(response_local)
html_cooking_obj = etree.HTML(response_cooking)
#                                   这里的@class匹配也可以换成索引                                                                                                                                                          这里的position()<=6是指获取前6个表格,因为后面是一些探索类技能的表格
tr_list_l : List[str] = html_local_obj.xpath("/html/body/div[@class='main-container']/div[@class='resizable-container']/div[@class='page has-right-rail']/main/div[@id='content']/div[@id='mw-content-text']/div[@class='mw-parser-output']/table[position() <= 6]/tbody/tr/td[2]/a/text()")
tr_list_c : List[str] = html_cooking_obj.xpath("/html/body/div[@class='main-container']/div[@class='resizable-container']/div[@class='page has-right-rail']/main/div[@id='content']/div[@id='mw-content-text']/div[@class='mw-parser-output']/table[position() <= 6]/tbody/tr/td[2]/a/text()")

# print(tr_list_c)

with open("collective_item.csv",mode="w",encoding="utf-8") as fp:
    for obj in tr_list_l:
        obj = obj.lower()
        obj = obj.replace(" ","_")
        # print(obj)
        fp.write(obj)
        fp.write("\n")
    # print(etree.tostring(obj, pretty_print=True).decode("utf-8"))

    for obj in tr_list_c:
        obj = obj.lower()
        obj = obj.replace(" ","_")
        # print(obj)
        fp.write(obj)
        fp.write("\n")

# for obj in tr_list:
#     print(obj.xpath("//td"))
