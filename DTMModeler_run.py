from VEP_TMScripts.DTMModeler import *
import argparse



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A gensim-based topic modeler for Serendip')

    parser.add_argument('-m','--model_name', help='name of the model', required=True)
    #parser.add_argument('--corpus_path', help='path to corpus directory', required=True)
    parser.add_argument('-o','--output_path', help='path to output directory (new directory by will be made for model_name', required=True)
    parser.add_argument('-fp','--data_file_pattern', help='pattern for the articles per year z.b. var/tmp/ps_%s.pickle ', required=True)
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
