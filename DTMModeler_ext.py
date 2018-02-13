from VEP_TMScripts.DTMModeler import *

def buildGSmodel_intervall(args):
    new_args = args
    sy = args.start_year
    ey = args.end_year

    if args.start_year_for_output is not None:
        print("Startyear output: %s"%args.start_year_for_output)
        sty = args.start_year_for_output
    else:
        sty =sy

    if args.growing == "true":
        print("growing_mode")
        intervall =  args.intervall
    else:
        intervall = 1

    for start_year in range(sty,ey,intervall):
        print(start_year)
        if args.growing == "true":
            new_args.start_year = sy
            new_args.end_year = start_year

        else:
            new_args.start_year=start_year
            new_args.end_year = start_year+args.intervall
        buildGSmodel(new_args)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A gensim-based topic modeler for Serendip')

    parser.add_argument('-m','--model_name', help='name of the model', required=True)
    #parser.add_argument('--corpus_path', help='path to corpus directory', required=True)
    parser.add_argument('-o','--output_path', help='path to output directory (new directory by will be made for model_name', required=True)
    parser.add_argument('-fp','--data_file_pattern', help='pattern for the articles per year z.b. var/tmp/ps_%s.pickle ', required=True)
    parser.add_argument('-n', '--num_topics', help='number of topics to infer', type=int, required=True)
    parser.add_argument('-s', '--start_year', help='start-year', type=int, required=True)
    parser.add_argument('-e', '--end_year', help='end-year', type=int, required=True)
    parser.add_argument('-i', '--intervall', help='intervall', type=int, required=True)
    parser.add_argument('--chunk_size', help='chunk size of a document', type=int)
    parser.add_argument('--nltk_stopwords', help='extract nltk English stopwords from documents before modeling', action='store_true')
    parser.add_argument('--extra_stopword_path', help='path to file containing space-delimited stopwords to exclude before modeling')
    parser.add_argument('--silent', help='if set, will suppress console output', action='store_true')
    parser.add_argument('--growing',help='if set "true", will create topics based on a growing corpus, from start_year to end_year, with steps set by the intervall')
    parser.add_argument('--start_year_for_output',type=int,
                        help='is meant to be used in connection with growing. This gives the first year which will be calculated, if not set equals start_year')

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

    buildGSmodel_intervall(args)
