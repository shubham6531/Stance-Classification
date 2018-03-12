import re
import sys
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

## processed output file: hashtags, http, stopwords removed. Abbreviations substituted with actual words and lemmatized
f_write=open(sys.argv[2],'w')

## dic contains a dictionary of slangs/abbreviations usually used in tweets, and their interpretations 
dic = {}
with open('emnlp_dict.txt','r') as f3:
	for sent in f3:
		norm = sent.strip('\n').split('\t')
		dic[norm[0]] = norm[1]

## processing data in the dataset file
with open(sys.argv[1]) as f:
	for line in f:
		line=line.strip('\n')
		tabs=line.split(',')
		words = tabs[2].split()

		words = [re.sub('[^A-Za-z0-9]+', '', word.lower()) for word in words if not word.startswith('http') and not word.startswith('#')]
		if len(words)==0:
			continue

		for l in range(len(words)):
			try:
				words[l] = dic[words[l]]
			except KeyError:
				continue

		stopword_set = set(stopwords.words('english'))
		cleaned_tokens = [i for i in words if i not in stopword_set]

		wnl = nltk.WordNetLemmatizer()
		cleaned_tokens = [wnl.lemmatize(t) for t in cleaned_tokens]
	
		tabs[2] = ' '.join(cleaned_tokens)
		if tabs[2] == "":
			continue

		f_write.write(tabs[0]+","+tabs[1]+","+tabs[2]+"\n")

f_write.close()
