"""
prompt:
1. 遍历53个web页面，地址为”https://www.aibj.fun/forum.php?mod=forumdisplay&fid=16&filter=typeid&typeid=111&page={number}“
其中number变化从1-54.
2. 针对每一个页面，获取页面上所有以"thread-"开头的超链接，得到一个地址列表，然后对该列表去重
3. 针对第2步得到的地址列表，访问每个地址。每个地址之前加上前缀”https://www.aibj.fun/“进行访问。
4. 针对第3步中每个地址的内容，查找是否存在目标字符串，例如”素衣“，如果存在则保存改地址，并输出到终端。

"""

import sys
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

def search_pages(target_string):
    # 存储所有找到的链接
    all_links = set()
    found_pages = []
    
    # 遍历53个页面
    for page_num in range(1, 54):
        # url = f"https://www.aibj.fun/forum.php?mod=forumdisplay&fid=16&filter=typeid&typeid=112&page={page_num}" # 丰台
        url = f"https://www.aibj.fun/forum.php?mod=forumdisplay&fid=16&filter=typeid&typeid=111&page={page_num}" # 朝阳
        try:
            # 获取页面内容
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找所有以thread-开头的链接
            links = soup.find_all('a', href=lambda x: x and x.startswith('thread-'), onclick=lambda x: x is not None)
            
            
            # 提取链接并去重
            for link in links:
                all_links.add(link['href'])
                
            print(f"已处理第{page_num}页，找到{len(links)}个链接")
            
            # 添加延时避免请求过快
            time.sleep(0.5)
            
        except Exception as e:
            print(f"处理第{page_num}页时出错: {str(e)}")
            continue
    
    print(f"\n总共找到{len(all_links)}个唯一链接")
    
    # 访问每个链接并搜索目标字符串
    
    total_links = len(all_links)
    for link in tqdm(all_links, desc="处理链接", total=total_links):
        full_url = f"https://www.aibj.fun/{link}"
        try:
            response = requests.get(full_url)
            if target_string in response.text:
                found_pages.append(full_url)
                print(f"找到匹配页面: {full_url}")
            
            # 添加延时避免请求过快
            time.sleep(0.2)
            
        except Exception as e:
            print(f"访问{full_url}时出错: {str(e)}")
            continue

        # print(f'full_url = {full_url}')
        # print(f'content = {response.text}')
        # break
    
    return found_pages

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None  # 从命令行参数获取搜索目标
    if target is None:
        print("错误: 请提供搜索目标字符串作为命令行参数")
        sys.exit(1)
    print(f"开始搜索包含'{target}'的页面...")
    found = search_pages(target)
    print(f"\n搜索完成，共找到{len(found)}个匹配页面")
    print(f'page list: {found}')
