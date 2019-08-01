# -*- coding: UTF-8 -*-
import  requests
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')#python处理字符流改成utf-8形式

def translate_text(target, content):
	language_type = ""
	url = "https://translation.googleapis.com/language/translate/v2"
	data = {
	    'key':"AI*******mpI",
	    'source': language_type,
	    'target': target,
	    'q': content,
	    'format': "text"
	}
	#headers = {'X-HTTP-Method-Override': 'GET'}
	#response = requests.post(url, data=data, headers=headers)
	response = requests.post(url, data)
	# print(response.json())
	print(response)
	res = response.json()
	print(res["data"]["translations"][0]["translatedText"])
	result = res["data"]["translations"][0]["translatedText"]
	print(result)
	return result


if __name__ == '__main__':
	content = "Teknoloji haberleri ve ürün incelemeleri"
	target = 'zh-cn'
	print (translate_text(target,content))
