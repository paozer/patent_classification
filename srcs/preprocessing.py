import os

import xml.etree.ElementTree as ET
	
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from textblob.wordnet import ADV, NOUN, VERB, ADJ
from contractions import fix

def extract_xml(root, file_):
	
	"""Extract patent text and ipcs from given xml file.
	
	Parameters
	----------
	root : string
		Root path for all files
	
	file_ : string
		Specific file path w/o root path
	
	Returns
	-------
	ipcs : list of strings
		All ipcs that have been found in the xml file by scanning for 'ipc' and 'ic' tags
	
	text : string
		Patent text found by scanning for 'txt' tag
	"""
	
	
	
	ipcs = []
	text = ""
	
	path = os.path.join(root, file_)

	# construct a tree of xml file and get its root
	xml_tree = ET.parse(path)
	root_tree = xml_tree.getroot()

	# extract main ipc
	for elem in root_tree.findall('ipcs'):
		ipcs.append(str(elem.attrib['mc']))

	for elem in root_tree:

		# extract additional ipcs and patent description
		for subelem in elem.findall('ipc'):
			ipcs.append(str(subelem.attrib['ic']))

		for subelem in elem.findall('txt'):
			text = subelem.text
			
	return ipcs, text
	

def replace_contractions(text):
	"""Replace all contractions with respective words
	
	Parameters
	----------
	text : string
		Original text
	
	Returns
	-------
	fixed_text : string
		Text with replaced contractions
	"""
	
	
	fixed_text = fix(text)
	return fixed_text

def tokenise_text(text):
	"""Tokenise text into word-tokens
	
	Parameters
	----------
	text : string
		Original text
	
	Returns
	-------
	word_token_list : list of strings
		List with all word-tokens
	"""
	
	word_token_list = word_tokenize(text)
	return word_token_list

def remove_numbers(words):
	"""Remove all numbers from word list.
	
	Unlike words, numbers without any context are not expected to provide any explanatory value for topic classification.
	
	Parameters
	----------
	words : string
		Original word-token list
	
	Returns
	-------
	new_words : list of strings
		List with all remaining word-tokens
	"""
	
	new_words = []
	for word in words:
		if not word.isdigit():
			new_words.append(word)
		else:
			pass

	return new_words

def remove_stopwords(words): 
	"""Remove all stopwords from word list.
	
	The word list is compared against two stopword lists:
	
		1. Stopword list provided in python module nltk, which is using a stopword corpus based on Porter et al [1]. 
	       This list covers 2400 stopwords regularly appear frequently in texts without regarding a specific topic.
	
		2. Stopword list provided by the United States Patent and Trademark Office [2].
		   This list covers terms that appear so frequently in patent texts that they provide no value for distinguishing different patents. 
	
	Parameters
	----------
	words : string
		Original word-token list
	
	Returns
	-------
	new_words : list of strings
		List with all remaining word-tokens
	
	References
	----------
	[1] Porter, M. F. 1980. "An Algorithm for Suffix Stripping." Program, 14(3), 130-37
	[2] United States Patent and Trademark Office, 2017, "Stopwords", patft.uspto.gov/netahtml/PTO/help/stopword.htm without nltk.corpus.words('english') words
	"""
	
	
	new_words = []
	stopwords_uspto = ['accordance', 'according', 'also', 'another', 'claim', 'comprises', 'corresponding', 'could', 'described', 'desired', 'embodiment', 'fig', 'figs', 'generally', 'herein', 'however', 'invention', 'means', 'onto', 'particularly', 'preferably', 'preferred', 'present', 'provide', 'provided', 'provides', 'relatively', 'respectively', 'said', 'since', 'suitable', 'thereby', 'therefore', 'thereof', 'thereto', 'thus', 'use', 'various', 'whereby', 'wherein', 'would']
	
	for word in words:
		if word not in (stopwords.words('english') + stopwords_uspto):
			new_words.append(word)
		else:
			continue

	return new_words


def remove_punctuation(words):
	
	"""Remove all punctuations and words that contain punctuations.
	
	Parameters
	----------
	words : string
		Original word-token list
	
	Returns
	-------
	new_words : list of strings
		List with all remaining word-tokens
	"""
	
	new_words = []
	punctuation = ['.', ',', '?', '!', ';', "'", '"', ':', '(', ')', '§', '$', '%']
    
	for word in words:
		# Remove pure punctuations
		if word not in punctuation:
			
			# Remove words containing punctuations
			p_flag = False
			for p in punctuation:
				
				if p in word:
					p_flag = True
			if not p_flag:
				new_words.append(word)
    
	
	return new_words

def to_lowercase(words):
	"""Convert all given words to lowercase words.
	
	Parameters
	----------
	words : string
		Original word-token list
	
	Returns
	-------
	new_words : list of strings
		List with all remaining word-tokens
	"""
	
	new_words = []

	for word in words:
		new_words.append(word.lower())
	
	return new_words # works

def lemmatize(words):
	"""Lemmatize all words.
	
	Lemmatazation is applied based on the large lexical database WordNet via the NLTK interface with additionally using appropriate POS tags
	in order to improve lemmatization results. This lemmatazation implementation has been chosen since it delivered the most reliable results when
	tested on a small sample of sentences compared to other lemmatization implementations like TextBlob Lemmatizer or Standford CoreNLP.
	
	Parameters
	----------
	words : string
		Original word-token list
	
	Returns
	-------
	new_words : list of strings
		List with all remaining word-tokens
	"""
	
	tag_words = pos_tag(words)

	dic = {
		'JJ': ADJ,
		'JJR': ADJ, 
		'JJS': ADJ, 

		'RB': ADV, 
		'RBR': ADV, 
		'RBS': ADV,

		'NN': NOUN,
		'NNP': NOUN,
		'NNS': NOUN,
		'NNPS': NOUN,

		'VB': VERB,
		'VBG': VERB,
		'VBD': VERB,
		'VBN': VERB,
		'VBP': VERB,
		'VBZ': VERB
	}

	lemmatizer = WordNetLemmatizer()
	new_words = []

	for word in tag_words:
		tag = word[1]
		
		try:
			new_tag = dic[tag]
			lemmatized_word = lemmatizer.lemmatize(word[0], new_tag)
			new_words.append(lemmatized_word)

		except KeyError:
			#print('KeyError on tag @ ', word[0]) 
			pass
		
	return new_words

def preprocess(text, word_limit):
	
	"""Apply preprocessing to a given text and return a list of words not larger than the given word limit
	
	Parameters
	----------
	text : string
		Original text
	
	word_limit : integer
		Limits the number of words returned from the final word list
	
	Returns
	-------
	words : list of strings
		List with all remaining word-tokens
	
	"""
	
	text = replace_contractions(text)
	words = tokenise_text(text)
	
	words = remove_numbers(words)
	words = remove_stopwords(words)
	words = remove_punctuation(words)
	words = to_lowercase(words)
	words = lemmatize(words)
	words = words[:(min(word_limit, len(words)))]

	return words
	
def ensure_dir(file_path):
	"""Check if the given directory is already existents and make directory if not
	
	Parameters
	----------
	file_path : string
		File path to be checked and made if not existent
	"""
	dir = os.path.dirname(file_path)
	if not os.path.exists(dir):
		os.makedirs(dir)	
	
def preprocess_directory(directory, path_split, faulty_files, target_directory, word_limit):
	"""Apply preprocessing to all files in given directory and write resulting files to target directory with the same directory structure.
	
	Parameters
	----------
	source_directory : string
		Source directory
	
	path_split : string
		Sequence to identify the split point in a given file path
	
	faulty_files : list of strings
		List of files that have been identified as faulty in advance
	
	target_directory : string
		Target directory
	
	word_limit : integer
		Limits the number of words returned from the final word list in the preprocess method
	"""
	
	
	counter = 0

	# iterate through provided directory, extract and preprocess all files and save them into target_directory
	for root, dirs, files in os.walk(directory):

		for file_ in files:
		
			counter += 1

			#if counter%100 == 0:
			print(counter, root, file_)

			if '.xml' in file_:

				if file_ in faulty_files:
					continue
				
				# Extracting and preprocessing xml file
				ipcs, text = extract_xml(root, file_)
				words = preprocess(text, word_limit)			
			
				# Writing preprocessed file into new directory with same structure
				new_path = target_directory + root.split(path_split)[1] + '/'
				ensure_dir(new_path)
				f_destination = open('%s%s.txt' % (new_path, file_[0:-4]), 'a', encoding='utf-8')
				f_destination.write(' '.join(ipcs))
				f_destination.write('\n')
				f_destination.write(' '.join(words))



	


