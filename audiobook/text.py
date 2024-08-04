list()#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 01:23:05 2021

Generate text transcript from the time-aligned TextGrid data annotations.

This allow us to compute word-level feature in line with the annotations when
using an LSTM or Transformer model on the text.

@author: hugwei
"""
import string
import os, sys
# import textgrid as tg  # not needed anymore
import pandas as pd
import numpy as np
import argparse
import glob
from tqdm import tqdm
from string import punctuation
import requests
import json

PORT = 8900

def get_pos(text, port=PORT, annotators='pos'):
    """
    Get POS tag from stanford core NLP server.
    """
    try:
        out = requests.post(f'http://[::]:{port}/?properties='+'{"annotators":"'+f'{annotators}","outputFormat":"json"' + '}', data=text.encode('utf-8')).text
    except requests.ConnectionError:
        print("""Try running the following command in another process:
        java -Xmx5G -cp .:./* edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 8900 -timeout 30000 -threads 5 -maxCharLength 100000 -quiet False -serverProperties StanfordCoreNLP-french.properties -preload -outputFormat json
        """)
        raise ConnectionError("Could not connect to the coreNLP server.")
    d = json.JSONDecoder().decode(out)
    return d['sentences'][0]['tokens'][0]['pos']

from .config import STIM_PATH

utf8map = {"â":"Ã¢",
           "é":"Ã©",
           "è":"Ã¨",
           "ê":"Ãª",
           "ë":"Ã«",
           "î":"Ã®",
           "ï":"Ã¯",
           "ô":"Ã´",
           "ö":"Ã¶",
           "ù":"Ã¹",
           "É":"Ãī",
           "û":"Ã»",
           "ü":"Ã¼",
           "ç":"Ã§",
           "œ":"Å?",
           "€":"â¬",
           "°":"Â°",
           "à":"Ãł",
          "À":"ÃĢ"}
latin1map = {v:k for k,v in utf8map.items()}  | {'Ã':'À'}

def repair_char(token):
    newtok = []
    for k,c in enumerate(token):
        if c == 'Ã':
            if len(token) == 1:
                return latin1map[c]
            newtok.append(latin1map[c + token[k+1]])
            if k+2 < len(token):
                return ''.join(newtok) + repair_char(token[k+2:])
            else:
                return ''.join(newtok)
        else:
            newtok.append(c)
    return ''.join(newtok)

def generate_text(wordlist, add_eos=False, eos='</s>'):
    """
    Generate a raw text string from a list of words, detecting new sentences
    simply from capitalisation of first character in words.
    """
    text = ""
    for k, w in enumerate(wordlist):
        w = w.replace('-', '') # clean a bit
        if w.istitle() or (len(w)==1 and wordlist[k+1].istitle()): # deal with "t'" or "s'"
            if k == 0:
                text += w
            else:
                if len(wordlist[k-1]) != 1:
                    text += (eos if add_eos else "") + "\n" + w
                else:
                    text += " " + w # technically "'"+w ...
        else:
            text += " " + w
    return text

def remove_punctuation(text: str, punct=punctuation)->str:
    """
    Remove punctuation from a string of text.
    """
    table = str.maketrans('', '', punct)
    return text.translate(table)

def get_wordlevel_df(storyname, use=(), annot_type='GPT2',
                     annot_dir='/project/3027007.01/Stimuli/Annotations',
                     transcript='normalized') -> pd.DataFrame:
    """
    Return the data frame. Can subselect some columns with `use`.
    
    Default directory is '/project/3027007.01/Stimuli/Annotations'.
    
    Type of annotsations available:
        - GPT2 (contains surprisal, entorpy, KL, and PE)
        - dep (dependancy parse from UDPipe)
        - stan (stanford dependency parse)
        - wf or wordfreq (word frequency from SUBTEXT-NL or LEXICON)
        - tree (ALPINO constituency tree)
        
    Future choices: tree from morphosyntactic_nl parser, relying on Alpino but using 
    a different end hook.
    """
    if annot_type.lower() == 'gpt2':
        suffix = '_GPT2.csv'
        altsuffix = '-gpt2.csv'
    elif annot_type.lower() == 'dep':
        suffix = '-annot.csv'
        altsuffix = '_annot.csv'
    elif annot_type.lower() == 'stan':
        suffix = '-stanza_annot.csv'
        altsuffix = ''
    elif annot_type.lower() in ['wf', 'wordfreq']:
        suffix = '_freq.csv'
        altsuffix = '-wf.csv'
    elif annot_type.lower() == 'tree':
        suffix = '_syntfeats.csv'
        altsuffix = '-syntfeats.csv'
        # For now annotations are experimental, and only in my home folder
        # This one is based on my Alpino parse of original transcript
        #annot_dir = '/home/lacnsg/hugwei/Documents/Scripts/AudioBook_Experiment/syntfeats_old/'
        # This one is based on Sophie's Alpino parse of her revisited transcript
        #annot_dir = '/home/lacnsg/hugwei/Documents/Scripts/AudioBook_Experiment/syntfeats_SophieTranscript/'
        # This one is based on my Alpino parse of Cas transcript ("New transcript")
        #annot_dir = '/home/lacnsg/hugwei/Documents/Scripts/AudioBook_Experiment/syntfeats_new/'
        # This is from morphosyntacticparser-nl, built on top of Alpino but giving quite different output
        # Main advantage is that I get a consistent output, every sentence is parsed, every word get a value
        # /!\ THIS WAS ALSO APPLIED ON NEW TRANSCRIPTS (Cas) !!
        #annot_dir = '/home/lacnsg/hugwei/Documents/Scripts/AudioBook_Experiment/syntfeats_morpho/'
    else:
        raise KeyError("Annotation type must be one of: GPT2, dep, stan, tree or wf (or wordfreq)")
    if not os.path.exists(os.path.join(annot_dir, '_'.join([storyname, transcript]).rstrip('_') + suffix)):
        # Try alternatve naming
        if os.path.exists(os.path.join(annot_dir, '_'.join([storyname, transcript]).rstrip('_') + altsuffix)):
            suffix = altsuffix
        else:
            raise FileNotFoundError(f"File {'_'.join([storyname, transcript]) + suffix} does not exist")
    filename = os.path.join(annot_dir, '_'.join([storyname, transcript]).rstrip('_') + suffix)
    
    wordlvl = pd.read_csv(filename, index_col=0) #if (annot_type not in ['wf', 'wordfreq']) else pd.read_csv(filename)
    if wordlvl.index.name == 'word':
        wordlvl = wordlvl.reset_index()
    wordlvl.columns = wordlvl.columns.str.lower()
    if 'text' in wordlvl.columns: # For dep or others..
        wordlvl = wordlvl.rename(columns={'text': 'token'})
    if 'word' in wordlvl.columns: # For wf?
        wordlvl = wordlvl.rename(columns={'word': 'token'})
        wordlvl = wordlvl[wordlvl.token.str.contains('\w')].reset_index() 
    if 'type' in wordlvl.columns: # fix to swap head and type for dependency parse (stanza is ok, but the other seems swapped)
        if wordlvl.type.dtype == int:
            head_pos = wordlvl.type
            wordlvl.deprel = wordlvl.loc[:, 'head']
            wordlvl.loc[:, 'head'] = head_pos
            wordlvl = wordlvl.drop('type', axis=1)
    # let's remove EOS
    wordlvl = wordlvl[~wordlvl.token.str.contains('</s>')].reset_index(drop=True)
    # And SOS
    wordlvl = wordlvl[~wordlvl.token.str.contains('<s>')].reset_index(drop=True)
    # Drop useless columns
    if 'level_0' in wordlvl.columns:
        wordlvl = wordlvl.drop(['level_0'], axis=1)
    
    if use:
        if isinstance(use, str): use = [use]
        return wordlvl.loc[:, (use+['token'] if 'token' not in use else use)]
    else:
        return wordlvl
    
def get_textgrid_df(storyname, annot_dir='/project/3027007.01/Stimuli/Annotations',
                    transcript='normalized'):
    """
    Reads the forced-alignment data as written in CSV file format into a pandas
    dataframe.
    """
    filename = os.path.join(annot_dir, '_'.join([storyname, transcript]) + '_timed.csv')
    return pd.read_csv(filename, index_col=0)

def get_wordlevel_aligned(storyname, use=(), annot='GPT2',
                          tg_dir='/project/3027007.01/Stimuli/Annotations',
                          annot_dir='/project/3027007.01/Stimuli/Annotations',
                          transcripts='normalized'):
    """
    Returns a DataFrame with both word onsets and corresponding word feature 
    values.
    
    Parameters
    ----------
    use : tuple of str
        See `get_wordlevel_df`
    annot : str
        'GPT2', 'dep', 'stan', 'wf' or 'tree'
    tg_dir : str | Path-like
        Abosulte or relative path to directory containing time-aligned data.
    annot_dir : str | Path-like
        Abosulte or relative path to directory containing word level annotations 
        data.
    transcripts : str | dict
        If ``dict``, must contain a key for "wordlevel" and "textgrid", meaning
        the basename is different for each file.
        
    Note:
    -----
    The data will be basically formed of:
        
        - time onsets: tg_dir/storyname+transcripts['textgrid']_timed.csv
        - Annotations: annot_dir/storyname+transcripts['wordlevel']_suffix.csv
    
    With ``suffix`` depending on which :param:`annot` has been chosen.
        
    Returns
    -------
    pd.DataFrame
    """
    if isinstance(transcripts, dict):
        transcript_wf = transcripts['wordlevel']
        transcript_tg = transcripts['textgrid']
    else:
        transcript_wf, transcript_tg = transcripts, transcripts
    wf = get_wordlevel_df(storyname, use, annot_type=annot, annot_dir=annot_dir, transcript=transcript_wf) # word-level features
    tg = get_textgrid_df(storyname, annot_dir=tg_dir, transcript=transcript_tg) # textgrid data
    
    # Dealing with contractions in french (only for constituency tree):
    if annot.lower() == 'tree':
        tmp = wf.copy()
        contractions = {'à les': 'aux', 'à le': 'au', 'de le': 'du', 'de les': 'des'}
        nfix = 0
        for k, t in wf.token.iteritems():
            if t in ['le', 'les']:
                if (wf.token[k-1] == 'à'):
                    # Merge both feature values
                    tmp.loc[k-1, ['close', 'depth', 'open']] += tmp.loc[k, ['close', 'depth', 'open']]
                    # delete the extra line
                    tmp.loc[k, ['close', 'depth', 'open']] = np.nan
                    # And convert back to the contraction:
                    tmp.loc[k-1, 'token'] = contractions[' '.join([tmp.loc[k-1, 'token'], tmp.loc[k, 'token']])]
                    #print(f"At row {k}: {wl.loc[k-1, 'token']} +  {wl.loc[k, 'token']} -> {test.loc[k-1, 'token']}")
                    nfix += 1
                 # This needs a further check to see if POS of subsequent word is VERB or NOUN
                if wf.token[k-1] in ['de']:
                    if get_pos(wf.token[k+1]) != 'VERB':
                        # Merge both feature values
                        tmp.loc[k-1, ['close', 'depth', 'open']] += tmp.loc[k, ['close', 'depth', 'open']]
                        # delete the extra line
                        tmp.loc[k, ['close', 'depth', 'open']] = np.nan
                        # And convert back to the contraction:
                        tmp.loc[k-1, 'token'] = contractions[' '.join([tmp.loc[k-1, 'token'], tmp.loc[k, 'token']])]
                        #print(f"At row {k}: {wl.loc[k-1, 'token']} +  {wl.loc[k, 'token']} -> {test.loc[k-1, 'token']}")
                        nfix += 1

        tmp.dropna(inplace=True)
        tmp.reset_index(inplace=True)
        wf = tmp
        print(f"Fixed {nfix} contractions of the kind 'à le(s)' -> 'au(x)'")
    
    # Aligning word features (need to check for matching words)
    if (annot.lower() != 'gpt2') or (any([fr_story in storyname for fr_story in ['EAU', 'ANGE', 'BALL']])): # GPT2 features are aligned but we also force align on all french stories
        print("Aligning dataframes...\n")
        # For french stories, fix bad character encoding
        for c in wf.columns:
            if (pd.api.types.is_string_dtype(wf[c])):
                if 'ANGE' in storyname:
                    wf[c] = wf[c].str.replace('Ãĥ','').str.replace('Ã¢ÂĢÂĶ', '"').str.replace('ÃĤ','').str.replace('Ã¢ÂĢÂĻ',"'").str.replace('ÃħÂĵ', "Å?").str.replace('Â','Ã').apply(repair_char)
                else:
                    wf[c] = wf[c].str.replace('âĢĻ', "'").str.replace('âĢĶ', '"').str.replace("Â«", '«').str.replace('Â»', '»').apply(repair_char)
                
        # lower case
        wf.token = wf.token.str.lower()
        tg.word = tg.word.str.lower()
        words = tg.word.unique()
        
        # Indices of "token" appearing also in forced-aligned "words"
        indices = [k for (k, d) in enumerate(wf.token) if (d in words)]
        # drop all other words from word features
        wf = wf.iloc[indices]
        
        # gradually loop through matching pairs:
        k_wf = 0
        for idx, row in tqdm(tg.iterrows(), total=len(tg), file=sys.stdout):
            # match
            if row.word == wf.token.iloc[k_wf]:
                # add word features to text-grid dataframe at matching indices
                tg.loc[idx, wf.columns.to_list()] = wf.iloc[k_wf]
                k_wf += 1
                continue
                
            # Check forward word to skip on compound words (need more than one word ahead to account for possible punctuation)
            for ahead in range(1, 5):
                if (k_wf<len(wf)-ahead) and (row.word == wf.token.iloc[k_wf+ahead]):
                    # add word features to text-grid dataframe at matching indices
                    tg.loc[idx, wf.columns.to_list()] = wf.iloc[k_wf+ahead]
                    k_wf += ahead + 1
                    break
                
        # Now remove compound words, containing "-" in final tg:
        if tg.word.str.count('-').ravel().sum() > 0:
            counts = tg.word.str.count('-').ravel()
            pos_dash = np.argwhere(counts)
            for k, p in enumerate(pos_dash):
                tg = tg.drop(p).reset_index(drop=True)
        return tg
    else:
        # This work too for "tree" annotations, morphosyntactic ones, on new Transcripts (Cas version).
        # Remove punctuation in case...
        wf = wf.loc[wf['token'].str.contains(f"[^{string.punctuation}]")].reset_index(drop=True)
        # Replace weird encoding of wrong apostrophe for Cas transcripts...
        wf.token = wf.token.str.replace('Ã¢ÂĢÂ', "’")
        # Deal with compoud words!!!!! By removing them entirely.
        counts = tg.word.str.count('-').ravel()
        pos_dash = np.argwhere(counts)
        only_counts = counts[pos_dash]
        for k, p in enumerate(pos_dash):
            tg = tg.drop(p).reset_index(drop=True)
            to_drop = []
            for c in range(only_counts[k][0]+1):
                if k==0:
                    to_drop.append(p[0]+c)
                else:
                    shift = only_counts[:k-1].sum() - k
                    to_drop.append(p[0]+c+shift)
            wf = wf.drop(to_drop)
        wf = wf.reset_index(drop=True)
        return tg.join(wf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Text re-generation")
    parser.add_argument("-o", "--outdir", help="Output directory (created if does not exist)", default="clean_text")
    parser.add_argument("--eos", help="EOS token to be used, if unset will not use any")
    parser.add_argument("--tg", help="Only create textgrid dataframed", action="store_true")
    parser.add_argument("--dir", help="Input directory (for textgrid to csv)", default=os.path.join(STIM_PATH, 'Alignments'))
    
    args = parser.parse_args()
    
    TEXTGRID_FILES = glob.glob(os.path.join(args.dir, '*TextGrid'))

    os.makedirs(args.outdir, exist_ok=True)
    for fname in TEXTGRID_FILES:
        print("Processing ", os.path.basename(fname))
        tgdata = tg.TextGrid.fromFile(fname)
        wordlevel = tgdata.getFirst('ORT-MAU')
        words = [t.mark for t in wordlevel if t.mark != ''] # remove silences
    
        if args.tg:
            df = pd.DataFrame([(w.mark, w.minTime, w.maxTime) for w in wordlevel if w.mark != ""],
                              columns=['word', 'onset', 'offset'])
            df.to_csv(os.path.basename(fname).split('.')[0] + '_timed.csv')
        else:
            # Create a text with sentence line by line. Will use capitalisation of
            # word to detect new sentences (as punctuation has been removed)
            text = generate_text(words, add_eos=args.eos is not None, eos=args.eos)
        
            with open(os.path.join(args.outdir, os.path.basename(fname).split('.')[0] + '_cleaned.txt'), "w") as f:
                f.write(text)
