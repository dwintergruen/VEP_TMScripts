from past.builtins.misc import raw_input
from fpdf.php import print_r
__author__ = 'ealexand'

import os
import shutil
import sys
import codecs
import argparse
import time
import csv
from scipy.sparse import dok_matrix
import random
import json
import math
import numpy as np
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
import sklearn.feature_extraction.text as text
import pickle
from sklearn import decomposition
from RegexTokenizer import RegexTokenizer as RegT

#from nonnegfac.nmf import NMF



# Helper function that creates new directories, overwriting old ones if necessary and desired.
def createDir(name, force=False):
    if os.path.exists(name):
        if force:
            shutil.rmtree(name)
        else:
            response = raw_input('%s already exists. Do you wish to overwrite it? (y/n) ' % name)
            if response.lower() == 'y' or response.lower() == 'yes':
                shutil.rmtree(name)
            elif response.lower() == 'n' or response.lower() == 'no':
                print ('Modeler aborted.')
                exit(0)
            else:
                print ('Response not understood.')
                print ('Modeler aborted.')
                exit(1)
    os.makedirs(name, exist_ok=True)

def buildGSmodel(args):
    fullStartTime = time.time()
    model_name = "%s_%s_%s-%s"%(args.model_name,args.num_topics,args.start_year,args.end_year)
    modelDir = os.path.join(args.output_path, model_name)
    
    if not args.silent:
        print ('Creating  corpus...')
        start = time.time()
    
    vectorizer = text.CountVectorizer(stop_words="english", min_df=2)  
    
    #load texts
    pss = {}
    pss_all = []
    abstracts = []
    for y in range(args.start_year,args.end_year+1):
        pss[y] = pickle.load(open("/var/tmp/ps_%s.pickle"%y,"rb"))
        abstr = []
        for p in pss[y]:
            titles = p.title
            if titles is not None:
                titles = " ".join(p.title)
            else:
                titles = ""
            
         
            abstr.append(p.abstract + titles if not p.abstract is None else titles)
        pss_all += pss[y]
        abstracts += abstr
    
    # Now build the model!
    if not args.silent:
        print ('Build NMF model...')
        start = time.time()
    
    dtm = vectorizer.fit_transform(abstracts).toarray()
    wordList = np.array(vectorizer.get_feature_names())   
    clf = decomposition.NMF(n_components=args.num_topics, random_state=1)
    doctopic = clf.fit_transform(dtm)
    
    topic_words = []
    for topic in clf.components_:
        word_idx = np.argsort(topic)[::-1][0:100]
        topic_words.append([wordList[i] for i in word_idx])
    
    #norm alize
    doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
       
    #article_names = ["%s_%s_%s"%(p.author,p.date,p.title) for p in pss_all]
    #article_names = np.asarray(article_names)
    doctopic_orig = doctopic.copy()
    #num_of_articles = len(set(article_names))
    
    import pandas
    theta = pandas.DataFrame(doctopic)
    thetaT = theta.transpose()
    # Build Serendip Files
    serendipDir = os.path.join(modelDir, 'TopicModel')
    createDir(serendipDir)
    with open(os.path.join(serendipDir, 'theta.csv'), 'w',encoding="utf-8") as tFile:
        for row in thetaT:
            print_row = []
            for x,y in dict(thetaT[row]).items():  
                if math.isnan(y):
                    y = 0 
                print_row.append("%s,%s"%(x,y))
            tFile.write(",".join(print_row))
            tFile.write("\n")
            
          
    # Build theta, both writing out theta.csv and the object theta
    # theta is a list of dictionaries that map topic -> prop for each doc

    writeTopicCSVs(modelDir,doctopic, wordList,clf,serendipDir)
    
    writeDefaultMeta(pss_all, serendipDir)
    tag_corpus(modelDir,pss_all, clf, wordList, theta, doctopic, serendipDir)
    
    if not args.silent:
        print ('Total time elapsed: %.2f seconds' % (time.time() - fullStartTime))

    return {
        'fnames': pss,
        'model': doctopic
    }

def bow2matrix(bow, numDocs, numWords):
    s = dok_matrix((numWords, numDocs))
    for docNum in range(len(bow)):
        for wordId, count in bow[docNum]:
            s[wordId, docNum] = count
    return s

def getYear(art):
    splitted = art.date.split("-")
    return splitted[0]

def join_save(jstr,stri):
    if stri is None:
        return ""
    
    try:
        return jstr.join(stri).replace('"','')
    except TypeError:
        return stri
    
def writeDefaultMeta(pss_all, modelDir):
    with open(os.path.join(modelDir, 'metadata.csv'), 'w',encoding="utf-8") as mFile:
        metaWriter = csv.writer(mFile)
        metaWriter.writerow(['id','filename',"title","date","year","author"])
        metaWriter.writerow(['int','str','str',"str","int","str"])
        for i in range(len(pss_all)):
            metaWriter.writerow([i, 
                                 str(i),
                                 " ".join(pss_all[i].title).replace('"',''),
                                 pss_all[i].date,
                                 getYear(pss_all[i]),
                                 join_save(" ",pss_all[i].author),
                                 ])

# Given GenSim model and containing director, write topics to CSV files for use in Serendip
def writeTopicCSVs(modelDir, doctopic, wordList, clf, serendipDir , wordThreshold=None, densityThreshold=0.99, silent=False):
    if not silent:
        print ('Writing topics to CSV files...')
        topicStart = time.time()

    # If they don't give us a wordThreshold, set it to max (vocab size)
    if wordThreshold is None:
        wordThreshold = len(wordList)

    # Create directory for CSVs
    csvDir = os.path.join(serendipDir, 'topics_freq') # TODO: sal, ig?
    createDir(csvDir)

    # Loop through topics, writing words and proportions to CSV file
    for topicNum in range(len(clf.components_)):
        with open(os.path.join(csvDir, 'topic_%d.csv' % topicNum), 'w',encoding="utf-8") as csvF:
            topicWriter = csv.writer(csvF)
            
            cutoffIndex = 0
            currDensity = 0.0
            if clf.components_[topicNum].sum() != 0:
                topicArray = clf.components_[topicNum]/clf.components_[topicNum].sum()
            else:
                topicArray = clf.components_[topicNum]
            word_idx = np.argsort(topicArray)[::-1]
            while currDensity < densityThreshold and cutoffIndex < wordThreshold:
            
                prob = topicArray[word_idx[cutoffIndex]]
                word = wordList[word_idx[cutoffIndex]]
                topicWriter.writerow([word, prob])
                currDensity += prob
                cutoffIndex += 1

    if not silent:
        print ('Done writing topic CSV files. (%.2f seconds)' % (time.time() - topicStart))

# Tag the files with the trained model, creating tokens.csv files and rules.json files
def tag_corpus(modelDir,pss_all, clf, wordlist, theta, doctopic, serendipDir,  nameToChunkNums=None, silent=False):

    if not silent:
        print ('Tagging texts and writing token CSVs...')
        tokenStart = time.time()

    # Make the HTML directory for Serendip
    htmlDir = os.path.join(serendipDir, 'HTML')
    createDir(htmlDir)
    
    word2id ={ y:x for x,y in enumerate(wordlist)}

    # Create dictionaries for each topic mapping word to proportion
    topicDicts = [ {} for i in range(len(clf.components_)) ]
    for t in range(len(clf.components_)):
        topicArray = clf.components_[t]
        word_idx = np.argsort(topicArray)[::-1]           
        
        for i in range(len(wordlist)):
            topicDicts[t][word_idx[i]] = topicArray[i]
            
    if not silent:
        print ('Done making topic-word dictionaries. (%.2f seconds)' % (time.time() - tokenStart))

    # Query this p_tGwd[d][w] = ordered list of (topic, prop) descending by prop
    p_tGwd = [ {} for d in range(len(theta)) ]

    # Build tokenizer for files to be tagged
    taggingTokenizer = RegT(case_sensitive=False,
                             preserve_original_strs=True)#,
                             #excluded_token_types=(1,2,3))

    # Loop through the texts and tag 'em
    
    for textNum in range(len(pss_all)):
        
        textStr = " ".join(pss_all[textNum].title) + "\n" + pss_all[textNum].abstract if pss_all[textNum].abstract is not None else ""
        try:
            currTokens = taggingTokenizer.tokenize(textStr)
        except TypeError:
            print("textNum: %s : %s"%(textNum,textStr))
            continue
        rules = {}
        outList = []
        if args.chunk_size:
            wordIndex = 0
            #textChunks = nameToChunkNums[textName]
            chunkIndex = 0

        # Loop through all the tokens in the file
        for tokenIndex in range(len(currTokens)):
            token = currTokens[tokenIndex]
            # If it's a word or punc, write it out
            isWord = token[RegT.INDEXES['TYPE']] == RegT.TYPES['WORD']
            isPunc = token[RegT.INDEXES['TYPE']] == RegT.TYPES['PUNCTUATION']
            if isWord or isPunc:
                # Set joiner (how tokens will be pieced back together)
                try:
                    if currTokens[tokenIndex+1][RegT.INDEXES['TYPE']] == RegT.TYPES['WHITESPACE']:
                        joiner = 's'
                    elif currTokens[tokenIndex+1][RegT.INDEXES['TYPE']] == RegT.TYPES['NEWLINE']:
                        joiner = 'n'
                    else:
                        joiner = ''
                except IndexError: # Presumably this means that we're at the last token
                    if (len(currTokens)) == tokenIndex + 1:
                        joiner = ''
                    else:
                        raise
                csvLine = [token[RegT.INDEXES['STRS']][-1], token[RegT.INDEXES['STRS']][0], joiner]

                # If the word is in our model, get the tag
                word = token[RegT.INDEXES['STRS']][0]
                if isWord and word in wordlist:
                    
                    relevantTextIndex = textNum

                    # If we haven't already calculated p of topic given word, doc, do so
                    if not word in p_tGwd[relevantTextIndex]:
                        tmpTopicPropList = []
                        tot = 0.0
                        for t in range(len(clf.components_)):
                            try:
                                wordnum = word2id[word]
                                tmpProp = topicDicts[t][wordnum] * theta[relevantTextIndex][t]
                                tmpTopicPropList.append([t, tmpProp])
                                tot += tmpProp
                                # May get a KeyError if word isn't in topicDicts[t] or t isn't in theta[textNum]

                            except KeyError:
                                continue
                        for i in range(len(tmpTopicPropList)):
                            tmpTopicPropList[i][1] /= tot
                        tmpTopicPropList.sort(key=lambda x: x[1], reverse=True)
                        p_tGwd[relevantTextIndex][word] = tmpTopicPropList

                    # Randomly sample from the p_tGwd distribution to create a tag for this instance
                    rand = random.random()
                    densityTot = 0.0
                    i = 0
                    while densityTot < rand:
                        try:
                            topic, prop = p_tGwd[relevantTextIndex][word][i]
                        except IndexError:
                            pass
                            
                        densityTot += prop
                        i += 1
                        # TODO: also get the rank_bin

                    # Add rule to rules for sampled topic
                    rule_name = 'topic_%d' % topic
                    if rule_name in rules:
                        rules[rule_name]['num_tags'] += 1
                        rules[rule_name]['num_included_tokens'] += 1
                    else:
                        rules[rule_name] = {
                            'name': rule_name,
                            'full_name': rule_name,
                            'num_tags': 1,
                            'num_included_tokens': 1
                        }

                    # Now, add the tag to the end
                    csvLine.append(rule_name) # TODO: freq, sal, ig

                # Finally, append the tagged token to the outList, which will be written to file
                outList.append(csvLine)

        # Build directory for it
        nameSansExtension = "%s"%textNum
        currHTMLdir = os.path.join(htmlDir, nameSansExtension)
        createDir(currHTMLdir)
        # Write rules to json file
        with open(os.path.join(currHTMLdir, 'rules.json'), 'w',encoding="utf-8") as jsonF:
            jsonF.write(json.dumps(rules))
        # Write the tokens to CSV file
        with open(os.path.join(currHTMLdir, 'tokens.csv'), 'w', encoding="utf-8") as tokensF:
            tokensWriter = csv.writer(tokensF)
            tokensWriter.writerows(outList)

    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A gensim-based topic modeler for Serendip')

    parser.add_argument('-m','--model_name', help='name of the model', required=True)
    #parser.add_argument('--corpus_path', help='path to corpus directory', required=True)
    parser.add_argument('-o','--output_path', help='path to output directory (new directory by will be made for model_name', required=True)
    parser.add_argument('-n', '--num_topics', help='number of topics to infer', type=int, required=True)
    parser.add_argument('-s', '--start_year', help='start-year', type=int, required=True)
    parser.add_argument('-e', '--end_year', help='end-year', type=int, required=True)
    parser.add_argument('--chunk_size', help='chunk size of a document', type=int)
    parser.add_argument('--nltk_stopwords', help='extract nltk English stopwords from documents before modeling', action='store_true')
    parser.add_argument('--extra_stopword_path', help='path to file containing space-delimited stopwords to exclude before modeling')
    parser.add_argument('--silent', help='if set, will suppress console output', action='store_true')

    if 0:
        args = parser.parse_args([
            '--model_name', 'gr_1950_1975',
            '--output_path', '/var/tmp/models',
            '--num_topics', '50',
            '--nltk_stopwords',
            '--extra_stopword_path', '',
            '--chunk_size', '2000'
        ])
    if 1:
        args = parser.parse_args()

    buildGSmodel(args)
