import datetime
import pandas as pd
import os
import sys
import nltk.data
from ast import literal_eval
from dateutil import parser
from pathlib import Path

from arxiv_topics.config import Config
from arxiv_topics.db import DB


MAXLENGTH = 256 * 4  # 256 tokens * ~4 chars

pd.options.mode.copy_on_write = True # to employ inplace replace


def cleanup(col):
    """ clean up text """

    #replace selected accents from https://info.arxiv.org/help/prep.html#note-on-latex-accent-commands
    col.replace(regex={r'\\"A':'Ä',r'\\"a':'ä',r'\\\'A':'Á',r'\\\'a':'á',r'\\^A':'Â',r'\\^a':'â'
                      ,r'\\`A':'À',r'\\`a':'à',r'\\r{A}':'Å',r'\\r{a}':'å'
                      ,r'\\c{C}':'Ç',r'\\c{c}':'ç'
                      ,r'\\\'E':'É',r'\\\'e':'é',r'\\^E':'Ê',r'\\^e':'ê',r'\\`E':'È',r'\\`e':'è'
                      ,r'\\~N':'Ñ',r'\\~n':'ñ'
                      ,r'\\"O':'Ö',r'\\"o':'ö',r'\\\'O':'Ó',r'\\\'o':'ó',r'\\`O':'Ò',r'\\`o':'ò'
                      ,r'\\"U':'Ü',r'\\"u':'ü',r'\\\'U':'Ú',r'\\\'u':'ú',r'\\`U':'Ù',r'\\`u':'ù'
                      ,r'{\\O}':'Ø',r'{\\o}':'ø',r'{\\oe}':'œ',r'{\\ss}':'ß'}, inplace=True)
    
    #clean miscellaneous artifacts and LATex formulae to facilitate sentence embedding
    col.replace(regex={r'\n':' ' #new lines
                      ,r'  +':' ' #multiple spaces
                      ,r'[\(\[][aA](bridged|BRIDGED)(\.)?[\)\]]':'' #common start
                      ,r'(\()?(Context|CONTEXT):(\))?':'' #another common start
                      ,r'\$\[?([a-zA-Z])\]?\$':r'\1' #variables / formulae
                      ,r'\$\\\w+\{([a-zA-Z])\}\$':r'\1'
                      ,r'\$\\(mbf|mit|mbfit|mscr|mbfscr|mfrak|Bbb|mbffrak|mtt)(sans)?([a-zA-Z])\$':r'\3'
                      ,r'\$\\(\w+)\$':r'\1'
                      ,r'(\${1,2})(?:(?!\1)[\s\S])*\1':'(formula)'}, inplace=True)


def trim_sentences(sentences):
    """ trim the text to end after the last complete sentence before reaching MAXLENGTH """
    paragraph = sentences[0]

    for sent in sentences[1:]:
        if len(paragraph) + len(sent) < MAXLENGTH:
            paragraph = paragraph + ' ' + sent
        else:            
            break

    return paragraph


def get_creation_date(versions):
    """ extract first date from version dict """
    if isinstance(versions, str):
        versions = literal_eval(versions)
    
    for version_info in versions:
        if version_info['version'] == 'v1':
            return parser.parse(version_info['created']).date()


def write_chunk_to_sql(chunk, db, year_min, last_update):
    """ writes papers to database
    
      Arguments:
        chunk: chunk from pandas iterable
        db: arxiv-topics database
        year_min: exclude papers before this year
        last_update: exclude papers before this update_date
       
      Returns:
        number of papers after exclusions according to year_min and last_update
        number of imported papers
        number of withdrawn papers
        number of papers with abstracts shorter than 64 chars
        number of papers that couldn't be split into sentences by nltk
        maximum update_date in chunk
    """
    
    df = pd.DataFrame(chunk)

    update = df['update_date'].max()

    #extract year & month of creation
    df['creation_date'] = pd.to_datetime(df['versions'].apply(get_creation_date))
    df['creation_year'] = pd.DatetimeIndex(df['creation_date']).year
    df['creation_month'] = pd.DatetimeIndex(df['creation_date']).month

    #filter by year and update
    df = df.query(f'creation_year >= {year_min} and update_date > "{last_update}"')

    if df.empty:
        return 0, 0, 0, 0, 0, update


    count = len(df.index)

    #replace accents, newlines, common starting words and formulae
    cleanup(df['title'])
    cleanup(df['abstract'])   

    #filter withdrawn papers (filtering 'withdrawn' in the whole abstract is too broad)
    df = df[~df['abstract'].str.startswith((' This article has been removed'
                                           ,' This paper has been removed'
                                           ,' This submission has been removed'
                                           ,' This manuscript has been removed'
                                           ,' This article has been withdrawn'
                                           ,' The paper has been withdrawn'
                                           ,' This paper has been withdrawn'
                                           ,' This submission has been withdrawn'
                                           ,' This manuscript has been withdrawn'
                                           ,' This submission is a duplicate'
                                           ,' Withdrawn'
                                           ,' Paper withdrawn'
                                           ,' The article was withdrawn'
                                           ,' The paper was withdrawn'
                                           ,' This paper was withdrawn'
                                           ,' The paper is being withdrawn'
                                           ,' This paper is being withdrawn'))]

    count_withdrawn = len(df.index)    

    #filter papers with too short abstracts (mostly comments etc. inapplicable for topic extraction)
    df['abstract_length'] = df['abstract'].apply(lambda x: len(x))
    df = df[df['abstract_length'] > 63]

    count_short = len(df.index)    

    if df.empty:        
        return count, count_short, count - count_withdrawn, count_withdrawn - count_short, 0, update

    #trim abstracts to approximately fit into 256 tokens
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    df['abstract'] = df['abstract'].apply(lambda x: trim_sentences(sent_detector.tokenize(x.strip())))

    #remove abstracts with unknown punctuation
    df['abstract_length'] = df['abstract'].apply(lambda x: len(x))

    df = df[df['abstract_length'] <= MAXLENGTH]

    count_long = len(df.index)
    
    if not df.empty:

        #expand first three categories
        df = df.join(df['categories'].str.split(n=3,expand=True).add_prefix('cat_'))

        if not 'cat_1' in df.columns:
            df['cat_1'] = None
            
        if not 'cat_2' in df.columns:
            df['cat_2'] = None

        df = df[['id','title','abstract','abstract_length','creation_year','creation_month','update_date','cat_0','cat_1','cat_2']].set_index('id')

        #save to sql
        df.to_sql(con=db.connection(), name='import_abstracts', index_label='id', if_exists='append')


    #return counts & update date
    return count, count_long, count - count_withdrawn, count_withdrawn - count_short, count_short - count_long, update


if __name__ == "__main__":
    """ Read in arXiv dataset, preprocess data and save to sqlite database """

    config = Config()    
    
    print('Assuring database')
    db_path = Path(config.database)
    if len(db_path.parents) > 0:
        db_path.parents[0].mkdir(parents=True, exist_ok=True)
    
    db = DB(config.database)
    db.assure_database()
    
    last_update = '1990-01-01'
    update_date = '1990-01-01'

    if os.path.isfile(config.input_update_date):
        with open(config.input_update_date, 'r') as txt:
            last_update = txt.read()

    count_total = 0
    count_imported = 0
    count_withdrawn = 0
    count_short = 0
    count_long = 0

    print('Reading in json', end='')
    sys.stdout.flush()
    
    with pd.read_json(config.input_snapshot, lines=True, dtype={'id':str}, chunksize=50000) as jr:

        for chunk in jr:            
            total, imported, withdrawn, short, long, update = write_chunk_to_sql(chunk, db, config.pre_year_min, last_update)

            count_total += total
            count_imported += imported
            count_withdrawn += withdrawn
            count_short += short
            count_long += long
            
            if update > update_date:
                update_date = update
            
            print('.', end='')
            sys.stdout.flush()            

    print()
    print('Importing to sqlite')
    db.import_snapshot()

    with open(config.input_update_date, 'w') as txt:
        txt.write(update_date)

    print(f'Imported {count_imported} papers from a total of {count_total} published since {config.pre_year_min}.')
    print(f'Excluded {count_withdrawn} withdrawn papers, {count_short} papers with too short abstracts and {count_long} with unknown punctuation.')
    
