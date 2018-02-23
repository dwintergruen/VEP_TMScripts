import os
from collections import  defaultdict
import pandas
from DTMModeler import createDir, join_safe
from RegexTokenizer import RegexTokenizer as RegT
import pickle
import random
import json
import csv
import argparse
import time
def tag_corpus(topicFolder,start_year,end_year,articleData_file_pattern,silent=False):
    pss = {}
    pss_all = []
    abstracts = []
    try:
        topicFolder = topicFolder%(start_year,end_year)
        print(topicFolder)
    except:
        print(topicFolder)
    for y in range(start_year, end_year + 1):
        print(y)
        pss[y] = pickle.load(open(articleData_file_pattern % y, "rb"))
        abstr = []
        for p in pss[y]:
            titles = p.title
            if titles is not None:
                titles = " ".join(p.title)
            else:
                titles = ""
            if p.abstract is None:
                txt = titles
            else:
                txt = p.abstract + titles
            abstr.append(txt)
        pss_all += pss[y]
        abstracts += abstr

    topics = defaultdict(dict)
    words2topics = defaultdict(list)
    docs2topicFreq = defaultdict(dict)
    with open(os.path.join(topicFolder,"theta.csv"),"r") as inf:
        cnt = 0
        for t in inf.readlines():
            t = t.replace("\n","")
            splitted = t.split(",")
            for i in range(0,len(splitted),2):
                docs2topicFreq[cnt][int(splitted[i])] = float(splitted[i+1]) ## topic -> prep

            cnt += 1


    for t in os.listdir(os.path.join(topicFolder,"topics_freq")):

        num = int(os.path.splitext(t)[0].split("_")[1])
        print("topic:",num)
        with open(os.path.join(topicFolder,"topics_freq",t),"r") as inf:
            for l in inf.readlines():
                l.replace("\n","")
                splitted = l.split(",")
                try:
                    topics[num][splitted[0]] = float(splitted[1])
                    words2topics[splitted[0]].append((num,float(splitted[1]))) #sammle zu jedem word topic und prop
                except ValueError:
                    continue

    # Make the HTML directory for Serendip
    htmlDir = os.path.join(topicFolder, 'HTML')
    createDir(htmlDir)



    # Create dictionaries for each topic mapping word to proportion
    topicDicts = [{} for i in range(len(topics))]
      #    topicDicts[t][word_idx[i]] = topicArray[i]


    # Query this p_tGwd[d][w] = ordered list of (topic, prop) descending by prop
    p_tGwd = [{} for d in range(len(pss_all))]

    # Build tokenizer for files to be tagged
    taggingTokenizer = RegT(case_sensitive=False,
                            preserve_original_strs=True)  # ,
    # excluded_token_types=(1,2,3))

    # Loop through the texts and tag 'em

    for textNum in range(len(pss_all)):
        textAbstr = (pss_all[textNum].abstract if pss_all[textNum].abstract is not None else "")
        textStr = join_safe(" ", pss_all[textNum].title) + "\n" + textAbstr

        try:
            currTokens = taggingTokenizer.tokenize(textStr)
        except TypeError:
            print("textNum: %s : %s" % (textNum, textStr))
            continue
        rules = {}
        outList = []

        # Loop through all the tokens in the file
        for tokenIndex in range(len(currTokens)):
            token = currTokens[tokenIndex]
            # If it's a word or punc, write it out
            isWord = token[RegT.INDEXES['TYPE']] == RegT.TYPES['WORD']
            isPunc = token[RegT.INDEXES['TYPE']] == RegT.TYPES['PUNCTUATION']
            if isWord or isPunc:
                # Set joiner (how tokens will be pieced back together)
                try:

                    if currTokens[tokenIndex + 1][RegT.INDEXES['TYPE']] == RegT.TYPES['WHITESPACE']:
                        joiner = 's'
                    elif currTokens[tokenIndex + 1][RegT.INDEXES['TYPE']] == RegT.TYPES['NEWLINE']:
                        joiner = 'n'
                    else:
                        joiner = ''
                except IndexError:  # Presumably this means that we're at the last token
                    if (len(currTokens)) == tokenIndex + 1:
                        joiner = ''
                    else:
                        raise
                csvLine = [token[RegT.INDEXES['STRS']][-1], token[RegT.INDEXES['STRS']][0], joiner]

                # If the word is in our model, get the tag
                word = token[RegT.INDEXES['STRS']][0]
                if isWord and word in words2topics:

                    relevantTextIndex = textNum

                    # If we haven't already calculated p of topic given word, doc, do so
                    if not word in p_tGwd[relevantTextIndex]:
                        tmpTopicPropList = []
                        tot = 0.0
                        for t in range(len(topics)):
                            try:

                                tmpProp = topics[t][word] * docs2topicFreq[relevantTextIndex][t]
                                tmpTopicPropList.append([t, tmpProp])
                                tot += tmpProp
                                # May get a KeyError if word isn't in topicDicts[t] or t isn't in theta[textNum]
                            except KeyError:
                                continue
                        for i in range(len(tmpTopicPropList)):
                            if tot != 0:
                                tmpTopicPropList[i][1] /= tot
                            else:
                                tmpTopicPropList[i][1] = 0
                        tmpTopicPropList.sort(key=lambda x: x[1], reverse=True)
                        p_tGwd[relevantTextIndex][word] = tmpTopicPropList

                    # Randomly sample from the p_tGwd distribution to create a tag for this instance
                    rand = random.random()
                    densityTot = 0.0
                    i = 0
                    topic = None
                    while densityTot < rand and i < len(p_tGwd[relevantTextIndex][word]):
                        try:
                            topic, prop = p_tGwd[relevantTextIndex][word][i]
                            densityTot += prop
                        except IndexError:
                            pass
                        i += 1
                        # TODO: also get the rank_bin

                    # Add rule to rules for sampled topic
                    if topic is None:
                        print("NO TOPIC")
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
                    csvLine.append(rule_name)  # TODO: freq, sal, ig

                # Finally, append the tagged token to the outList, which will be written to file
                outList.append(csvLine)

        # Build directory for it
        nameSansExtension = "%s" % textNum
        currHTMLdir = os.path.join(htmlDir, nameSansExtension)
        createDir(currHTMLdir)
        # Write rules to json file
        with open(os.path.join(currHTMLdir, 'rules.json'), 'w', encoding="utf-8") as jsonF:
            jsonF.write(json.dumps(rules))
        # Write the tokens to CSV file
        with open(os.path.join(currHTMLdir, 'tokens.csv'), 'w', encoding="utf-8") as tokensF:
            tokensWriter = csv.writer(tokensF)
            tokensWriter.writerows(outList)



if __name__ == "__main__":
    print("start")
    parser = argparse.ArgumentParser(description='A gensim-based topic modeler for Serendip')

    parser.add_argument('-p', '--model_path', help='path to the models', required=True)
    # parser.add_argument('--corpus_path', help='path to corpus directory', required=True)
    parser.add_argument('-s', '--start_year',
                        help='start year', required=True)
    parser.add_argument('-e', '--end_year',
                        help='end year', required=True)
    parser.add_argument('-fp', '--data_file_pattern',
                        help='pattern for the articles per year z.b. var/tmp/ps_%s.pickle ', required=True)

    args = parser.parse_args()
    tag_corpus(args.model_path,args.start_year,args.end_year,args.data_file_pattern)
