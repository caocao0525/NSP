#!/usr/bin/env python
# coding: utf-8

# # Utilities
# Various functions to process the initial data

# In[3]:


#### To convert the file into .py
#!jupyter nbconvert --to script css_utility.ipynb


# In[2]:


import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from motif_utils import seq2kmer
from motif_utils import kmer2seq
from scipy.stats import norm
import random
import collections
import operator
import itertools
import pickle
import seaborn as sns
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook


# ## Index
# 
# * **[1. Gene and Genome file preprocessing](#1.-Gene-and-Genome-file-preprocessing)**
#     * [1-1. Gene file separation by chromosome](#1-1.-Gene-file-separation-by-chromosome)
#     * [1-2. Genome statistics](#1-2.-Genome-statistics)
# * **[2. Chromatin state preprocessing](#2.-Chromatin-state-preprocessing)**
#     * [2-1. Chromatin state file info](#2-1.-Chromatin-state-file-info)
#     * [2-2. Prerequisite dictionaries](#2-2.-Prerequisite-dictionaries)
#         * [2-2-1. Function to convert RGB into decimal RGB](#2-2-1.-Function-to-convert-RGB-into-decimal-RGB)
#     * [2-3. Generate CSS .bed to dataframe](#2-3.-Generate-CSS-.bed-to-dataframe)
#         * [2-3-1. Individual dataframe analysis](#2-3-1.-Individual-dataframe-analysis)
#     * [2-4. CSS string generation from dataframe](#2-4.-CSS-string-generation-from-dataframe)
#         * [2-4-1. Real length CSS](#2-4-1.-Real-length-CSS)
#         * [2-4-2. Unit-length CSS](#2-4-2.-Unit-length-CSS)
#     * [2-5. Chromatin State Statistics](#2-5.-Chromatin-State-Statistics)
# * **[3. Cutting the telomere: where to cut?](#3.-Cutting-the-telomere:-where-to-cut?)**
#     * [3-1. Quiescent state distribution](#3-1.-Quiescent-state-distribution)
#     * [3-2. Cut the telomere region on CSS and save the file](#3-2.-Cut-the-telomere-region-on-CSS-and-save-the-file) -> **pretrain data are saved**
#     * [3-3. Cut the chromatin states : genic/non-genic area](#3-3.-Cut-the-chromatin-states-:-genic-or-non-genic-area)
#         * [3-3-1. Genic area](#3-3-1.-Genic-area)
#         * [3-3-2. Non-genic area (intergenic region)](#3-3-2.-Non-genic-area-(intergenic-region))
#         * [3-3-3. Genic or Non-genic raw-length CSS to unit-length CSS](#3-3-3.-Genic-or-Non-genic-raw-length-CSS-to-unit-length-CSS)
#         * [3-3-4. Cut the unit-length css into trainable size and kmerize it](#3-3-4.-Cut-the-unit-length-css-into-trainable-size-and-kmerize-it) 
#         * [3-3-5. Fine-tuning data: Dataframe version](#3-3-5.-Fine-tuning-data:-Dataframe-version)
#         * [3-3-6. Fine-tuning data: save files as .tsv](#3-3-6.-Fine-tuning-data:-save-files-as-.tsv) ->**fine-tuning data are saved**
#     * [3-4. Count the number of 15th states in genic and non-genic region](#3-4.-Count-the-number-of-15th-states-in-genic-and-non-genic-region) 
# * **[4. CSS Pattern analysis](#4.-CSS-Pattern-analysis)**
# * **[5. Training result analysis](#5.-Training-result-analysis)**

# **Frequently used functions**

# In[3]:


def flatLst(lst):
    flatten_lst=[elm for sublst in lst for elm in sublst]
    return flatten_lst


# In[4]:


def file_list_maker(path, files):
    all_files=[]
    for file in files:
        file_path=os.path.join(path,file)
        all_files.append(file_path)
    return all_files


# In[75]:


def colored_css_str_as_is(sub_str):   # convert space into space
    col_str=""
    for letter in sub_str:
        if letter==" ":
            col_str+=" "
        else:                
            for state in list(state_col_255_dict.keys()):
                if letter==state:
                    r=state_col_255_dict[letter][0]
                    g=state_col_255_dict[letter][1]
                    b=state_col_255_dict[letter][2]
                    col_letter="\033[38;2;{};{};{}m{}\033[38;2;255;255;255m".format(r,g,b,letter)
                    col_str+=col_letter
    return print("\033[1m"+col_str+"\033[0;0m") 


# ## 1. Gene and Genome file preprocessing
# Handling the human gene location file and the reference human genome file *hg19*

# **Gene file info**
# * This file includes the information of the location of genes on the human genome.
# * Name: `RefSeq.WholeGene.bed`
# * Location: (local linux DLBOX2 ->) `../database/RefSeq/` (server ->) `euphonium:/work/Database/UCSC/hg19/` 
# * Structure: 
#     * tab-delimited
#     * columns: `{0:"chromosome",1:"TxStart",2:"TxEnd",3:"name",4:"unk0",5:'strand', 6:'cdsStart', 7:'cdsEnd',8:"unk1",9:"exonCount",10:"unk2",11:"unk3"}`
# <br>
# 
# **Genome file info**
# 
# * This file is the human reference genome file.
# * Name: `genome.fa`
# * Location: (local linux DLBOX2, macpro ->) `../database/hg19/` (server ->) `/work/Database/UCSC/hg19/`
# * Chromosome-wise file location: (local linux DLBOX2, macpro ->) `../database/hg19/genome_per_chr/`
# * Structure:
#     * `>` delimiter per chromosome (e.g. `>chr1`)
#     * The file is separated chromosome-wise, using following command lines
#         > (1) `sed 's/>//g' genome.fa > genome_mod.fa` : find `>` and remove it then save as `genome.fa`<br>
#         > (2) `awk '$1 ~/^chr/{close(name);name=$1;next}{print $1>name}' genome_mod.fa` : find string starting `chr` form `genome_mod.fa` and save the 1st field (=the base string) as reading the file. 

# In[5]:


# load the file from local
whole_gene_file='../database/RefSeq/RefSeq.WholeGene.bed'


# ### 1-1. Gene file separation by chromosome
# #### Function: `WhGene2GLChr`
# * **Description**: Generate the chromosome-wise list of dataframe of gene location
# <br>
# * **Input**: `whole_gene_file`
# * **Output**: `g_df_chr_lst` A list of chromosome-wise Dataframes, each of which contains `chromosome` (chromosome number), `TxStart`, `TxEnd`, and `name` (gene name). Note that `chrM` is removed in the process. 
# 
# * This fuction is used in the function `compGene2css` [jump](#compGene2css) which generates **`css_gene_lst_all`**, the list of list that contains the chromatin states for genic region per chromosome.

# In[6]:


# function for preprocess the whole gene data and produce chromosome-wise gene lists
# each element is dataframe

def whGene2GLChr(whole_gene_file='../database/RefSeq/RefSeq.WholeGene.bed'):
    print("Extracting the gene file ...")
    g_fn=whole_gene_file
    g_df_raw=pd.read_csv(g_fn, sep='\t', lineterminator='\n', header=None, low_memory=False)
    g_df_int=g_df_raw.rename(columns={0:"chromosome",1:"TxStart",2:"TxEnd",3:"name",4:"unk0",
                                  5:'strand', 6:'cdsStart', 7:'cdsEnd',8:"unk1",9:"exonCount",
                                  10:"unk2",11:"unk3"})
    g_df=g_df_int[["chromosome","TxStart","TxEnd","name"]]
    
    # Remove other than regular chromosomes
    chr_lst=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10',
             'chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19',
             'chr20','chr21','chr22','chrX','chrY']
    g_df=g_df.loc[g_df["chromosome"].isin(chr_lst)]
    
    # Create a list of chromosome-wise dataframe 
    g_df_chr_lst=[]
    for num in range(len(chr_lst)):
        chr_num=chr_lst[num]
        g_chr_df='g_'+chr_num
        locals()[g_chr_df]=g_df[g_df["chromosome"]==chr_num]
        g_chr_df=locals()[g_chr_df]
        g_chr_df=g_chr_df.sort_values("TxStart")
        g_df_chr_lst.append(g_chr_df)
    print("Done!")
    
    return g_df_chr_lst


# ### 1-2. Genome statistics
# 
# * Prerequisite file: chromosome-wise separated reference genome file.

# In[7]:


# prerequisite file load
chr_path='../database/hg19/genome_per_chr/'
chr_list=[os.path.join(chr_path, file) for file in sorted(os.listdir(chr_path))]
chr1=chr_list[0]


# #### Function `chrNdist`
# 
# * **Description**: Generate the index list and dataframe ('start' and 'end' location) of 'N' base in genome file. <br> 'N' indicates that it can be *any* base (See [reference](https://iubmb.qmul.ac.uk/misc/naseq.html))
# * **Input**: Chromosome-wise separated genome
# * **Output**: Two elements (`all_n_index` (list) and  `n_dist_df`(dataframe)). <br> `all_n_index` is just a list of all the indices where 'N's are located, while `n_dist_df` accomodates 'start', 'end', and 'count' as columns.
# * **Note** that the 'N' here stands for 50 bases. (resolution=50 bases)

# In[8]:


def chrNdist(chr_file=chr1):
    """
    input: divided genome by chromosome (without any index, only genome)
    output: dataframe of [start, end] position of "N" in the genome sequence
    """
    with open(chr_file) as infile:
        all_n_line="N"*50    # python reads text line by 50 characters
        all_n_index=[]
        all_n_start=[1]
        all_n_end=[]

        for i, line in enumerate(infile):
            if all_n_line in line:
                all_n_index.append(i)    # all_n_index is a list of N

        for i, num in enumerate(all_n_index):   
            if i==0:        
                pre_num=num
            elif num !=pre_num+1:
                all_n_start.append(num)
            pre_num=num   
        for i, num in enumerate(all_n_index):   
            if i==0:        
                pre_num=num
            elif num !=pre_num+1:
                all_n_end.append(pre_num+1)
            pre_num=num
        all_n_end.append(all_n_index[-1]+1)

        assert len(all_n_start)==len(all_n_end)
        
        n_dist_df=pd.DataFrame({"start":all_n_start,"end":all_n_end, 
                                "count":[e-s+1 for s,e in zip(all_n_start,all_n_end)]},
                               columns=["start","end","count"])
        ######## uncomment this block if you want to draw the histogram!
#         fig=plt.figure(figsize=(8,4))
#         plt.hist(all_n_index, 50, facecolor='teal', alpha=0.75)
#         plt.xlabel("Position")
#         plt.ylabel("number of 'N' lines")
#         plt.show()    
        return all_n_index, n_dist_df


# #### Function: `all_chr_Ndist `
# 
# * **Description**
#     * Draw a histogram of 'N' distiribution chromosome-wise.
#     * Generate a list of chromosome-wise list of the index for 'N' location (still, resolution = 50 bases)
# * **Input**: The reference genome file path `'../database/hg19/genome_per_chr/'`
# * **Option**: Normalization (default=`True`)
# 
# * **Output**
#     * A list of chromosome-wise list of 'N' location on genome.
#     * `all_chr_n_index_norm` (if normalization ON) 
#     * `all_chr_n_index` (if normalization OFF)
# <!-- ![](./desc_img/all_chr_Ndist.png) -->
# 
# <img src="./desc_img/all_chr_Ndist.png" width="500" height="250" />

# In[9]:


def all_chr_Ndist(ref_genome_path='../database/hg19/genome_per_chr/', normalization=True):
    
    """
    input: ref_genome_path='../database/hg19/genome_per_chr/'
    output: all_chr_n_index_norm (normalization ON) / all_chr_n_index (normalization OFF)
    option: normalization (all chromosome length= 0 to 1 for drawing a dist. graph)
    """
    
    path=ref_genome_path
    chr_list=[(file, os.path.join(path, file)) for file in sorted(os.listdir(path)) if "chrM" not in file] # remove chrM
    
    fig=plt.figure(figsize=(12,6))
    ax = plt.subplot(111)
    
    all_chr_n_index=[] # list of list (raw data)
    all_chr_n_index_norm=[] # list of list (normalized data)
    
    for i, (chr_no, chr_path) in enumerate(chr_list):
        all_n_index, n_dist_df=chrNdist(chr_path)
        # save the raw data
        all_chr_n_index.append(all_n_index)
        
        ########### normalization here ###########
        all_n_index_norm=[elm/all_n_index[-1] for elm in all_n_index]
        ##########################################
        
        grad_color=plt.cm.terrain(i*10)
        ax.hist(all_n_index_norm, 50, color=grad_color, histtype="step", label=chr_no)
        all_chr_n_index_norm.append(all_n_index_norm)
        
    ### show only the normalized disribution
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # Shrink current axis's height by 20% on the bottom
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Normalized Position")
    plt.ylabel("number of 'N' lines")
    plt.grid(b=None)

    plt.show()  
    
    if normalization:
        return all_chr_n_index_norm 
    else:
        return all_chr_n_index


# # 2. Chromatin state preprocessing
# **[back to index](#Index)**
# 
# Chromatin state file (`.bed` file) preprocessing to further analysis

# ## 2-1. Chromatin state file info
# 
# * This files are the chromatin state-annotated (15 different states, per 200 bps) genomes of 127 different cells.
# * Location: (local linux DLBOX2, macpro ->) `/database/bed/unzipped`  (server ->) `euph:/work/ChIP-seq/ROADMAP/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/final/*_15_coreMarks_dense.bed`
# * Structure: tab-delimited, 4 columns (chromosome numner, start, end, and state number)

# In[10]:


# create a pickle for a cell-wise dataframe
def total_df2pickle(total_df_list):
    for num, df_cell in enumerate(tqdm.notebook.tqdm(total_df_list)):
        path="../database/cell_pickle/"
        if num+1 < 10:
            file_name=path+"df_cell"+"00"+str(num+1)+".pkl"
            df_cell_pickled=df_cell.to_pickle(file_name)
        elif num+1 < 100:
            file_name=path+"df_cell"+"0"+str(num+1)+".pkl"
            df_cell_pickled=df_cell.to_pickle(file_name)
        else:
            file_name=path+"df_cell"+str(num+1)+".pkl"
            df_cell_pickled=df_cell.to_pickle(file_name)

path='../database/bed/unzipped/'
bed_files=os.listdir(path)

pickle_path='../database/cell_pickle'
pickle_files=os.listdir(pickle_path)
            
all_files=file_list_maker(path, bed_files)
all_cell_pickles=file_list_maker(pickle_path, pickle_files)


# In[11]:


all_files[0]


# In[12]:


all_cell_pickles[0]


# ## 2-2. Prerequisite dictionaries

# In[13]:


state_dict={1:"A", 2:"B", 3:"C", 4:"D", 5:"E",6:"F",7:"G",8:"H" ,
                9:"I" ,10:"J",11:"K", 12:"L", 13:"M", 14:"N", 15:"O"}


# In[14]:


css_name=['TssA','TssAFlnk','TxFlnk','Tx','TxWk','EnhG','Enh','ZNF/Rpts',
          'Het','TssBiv','BivFlnk','EnhBiv','ReprPC','ReprPcWk','Quies']


# In[15]:


css_dict=dict(zip(list(state_dict.values()), css_name))  # css_dict={"A":"TssA", "B":"TssAFlnk", ... }


# In[16]:


# color dict update using the info from https://egg2.wustl.edu/roadmap/web_portal/chr_state_learning.html
# 18th May 2022
css_color_dict={'TssA':(255,0,0), # Red
                'TssAFlnk': (255,69,0), # OrangeRed
                'TxFlnk': (50,205,50), # LimeGreen
                'Tx': (0,128,0), # Green
                'TxWk': (0,100,0), # DarkGreen
                'EnhG': (194,225,5), # GreenYellow 
                'Enh': (255,255,0),# Yellow
                'ZNF/Rpts': (102,205,170), # Medium Aquamarine
                'Het': (138,145,208), # PaleTurquoise
                'TssBiv': (205,92,92), # IndianRed
                'BivFlnk': (233,150,122), # DarkSalmon
                'EnhBiv': (189,183,107), # DarkKhaki
                'ReprPC': (128,128,128), # Silver
                'ReprPCWk': (192,192,192), # Gainsboro
                'Quies': (240, 240, 240)}  # White -> bright gray 


# ### 2-2-1. Function to convert RGB into decimal RGB

# In[17]:


def colors2color_dec(css_color_dict):
    colors=list(css_color_dict.values())
    color_dec_list=[]
    for color in colors:
        color_dec=tuple(rgb_elm/255 for rgb_elm in color)
        color_dec_list.append(color_dec)        
    return color_dec_list


# In[18]:


state_col_dict=dict(zip(list(state_dict.values()),colors2color_dec(css_color_dict)))


# In[19]:


state_col_255_dict=dict(zip(list(state_dict.values()),list(css_color_dict.values())))


# In[20]:


css_name_col_dict=dict(zip(css_name,state_col_dict.values()))


# ## 2-3. Generate CSS .bed to dataframe

# In[21]:


# create dataframe from bed file
# bed file here means: EXXX_15_coreMarks_stateno.bed

def bed2df_as_is(filename):    
    
    """Create dataframe from the .bed file, as is.
    Dataframe contains following columns:
    chromosome |  start |  end  | state """
    
    df_raw=pd.read_csv(filename, sep='\t', lineterminator='\n', header=None, low_memory=False)
    df=df_raw.rename(columns={0:"chromosome",1:"start",2:"end",3:"state"})
    df=df[:-1]
    df["start"]=pd.to_numeric(df["start"])
    df["end"]=pd.to_numeric(df["end"])
    
    return df


# In[22]:


def bed2df_expanded(filename):
    
    """Create an expanded dataframe from the .bed file.
    Dataframe contains following columns:
    chromosome |  start |  end  | state | length | unit | state_seq | state_seq_full"""
   
    df_raw=pd.read_csv(filename, sep='\t', lineterminator='\n', header=None, low_memory=False)
    df=df_raw.rename(columns={0:"chromosome",1:"start",2:"end",3:"state"})
    df=df[:-1]
    df["start"]=pd.to_numeric(df["start"])
    df["end"]=pd.to_numeric(df["end"])
    df["state"]=pd.to_numeric(df["state"])
    df["length"]=df["end"]-df["start"]
    df["unit"]=(df["length"]/200).astype(int)  # chromatin state is annotated every 200 bp (18th May 2022)
               
    df["state_seq"]=df["state"].map(state_dict)
    df["state_seq_full"]=df["unit"]*df["state_seq"]
    
    return df 


# In[23]:


def total_df_maker(all_files):
    
    """Create a list of dataframe from a list of bed files.
    This function utilizes the function named 'bed2df_expanded.'"""
    
    total_df=[]
    for filename in all_files:
        df=bed2df_expanded(filename)
        total_df.append(df)
    return total_df


# ### 2-3-1. Individual dataframe analysis
# 
# * Functions for analyzing an individual dataframe
# * CSS here refers Chromatin state sequence

# In[24]:


def numchr(df):
    assert "chromosome" in df.columns, "Check your df has the column named 'chromosome'"
    return df["chromosome"].nunique()    


# In[25]:


# create a large piece of string of the whole state_seq_full 
# CSS: chromatin-state sequence

def df2css_allchr(df):
    
    """Create a large piece of string of the whole state_seq_full 
    This function generates a string from the entire chromosomes"""
    
    state_seq_full_list=df["state_seq_full"].tolist()
    state_seq_full_to_str=''.join([elm for elm in state_seq_full_list ])
    return state_seq_full_to_str


# #### Create CSS chromosome-wise

# In[26]:


# first, learn where one chromosome ends in the df
# this is just a prerequisite function for df2css_chr

def df2chr_index(df):
    
    """Create a list of smaller piece of string of the state_seq_full per chromosome
    This function generates a list of chromatin state sequence strings chromosome-wise"""
    
    total_row=len(df)
    chr_len=[]
    chr_check=[]
    chr_index=[]

    for i in range(total_row):
        if (df["start"].iloc[i]==0) & (i >0):
            chr_len.append(df["end"].iloc[i-1]) # chr_len stores the end position of each chromosome
            chr_check.append(df["start"].iloc[i]) # for assertion : later check chr_check are all zero
            chr_index.append(i-1) # the index (row number)

    end_len=df["end"].iloc[-1] # add the final end position
    end_index=total_row-1 # add the final end index (row number)
 
    chr_len.append(end_len)
    chr_index.append(end_index)

    assert len(chr_len)==df["chromosome"].nunique() #assert the length of the list corresponds to no. of chromosome
    assert len(chr_index)==df["chromosome"].nunique()
    
    return chr_index


# #### Create df cut by each chromosome

# In[27]:


def df2chr_df(df):
   
    """Create a list of dataframes, each of which containing 
    the the whole expanded type of dataframe per chromosome"""
    
    start=0
    df_chr_list=[]
    chr_index=df2chr_index(df)
    
    for index in chr_index:
        df_chr=df[start:index+1] # note that python [i:j] means from i to j-1
        chr_name=df["chromosome"].iloc[start] # string, such as chr1, chr2, ...
        df_name='df_'+chr_name  # the chromosome-wise data stored like df_chr1, df_chr2, ...
        locals()[df_name]=df_chr # make a string into a variable name
        df_chr_list.append(df_chr)
        start=index+1
    
    return df_chr_list   # elm is the df of each chromosome


# #### Create CSS chromosome-wise, string only

# In[28]:


# create a list of dataframes, each of which contains the name of chromosome and chromosome-wise string of state_seq_full
# This is prerequisite function for df2css_chr_string

def df2css_chr(df):
   
    """Create a list of dataframes, each of which containing 
    the chromosome name and the state_seq_full per chromosome (2 columns)"""
    
    start=0
    df2col_chr_list=[]
    chr_index=df2chr_index(df)
    
    for index in chr_index:
        df_chr=df[["chromosome","state_seq_full"]][start:index+1] # note that python [i:j] means from i to j-1
        chr_name=df["chromosome"].iloc[start] # string, such as chr1, chr2, ...
        df2col_name='df2col_'+chr_name  # the chromosome-wise data stored like df2col_chr1, df2col_chr2, ...
        locals()[df2col_name]=df_chr # make a string into a variable name
        df2col_chr_list.append(df_chr)
        start=index+1
    
    return df2col_chr_list    


# In[29]:


def df2css_chr_str(df):
    
    """Create a list of strings which is the state_seq_full, all-connected per chromosome"""
    
    chr_index=df2chr_index(df)  
    chr_index_num=len(chr_index) 

    df2col_chr_list=df2css_chr(df)  # contains a list of df: chromosome name, state_seq_full (2-column datafame)
    chr_css_list=[]

    for num in range(chr_index_num): 
        css_full_list=df2col_chr_list[num]["state_seq_full"].tolist()  # extract the state_seq_full only and make it a list
        css_full_to_str=''.join([elm for elm in css_full_list]) # make it a long string of all-connected state_seq_full (chromosome-wise)
        chr_css_list.append(css_full_to_str)
    return chr_css_list


# ## 2-4. CSS string generation from dataframe

# ### 2-4-1. Real length CSS
# 
# #### Function: `df2longcss`
# * make a long string of the css (not using unit, but the **real** length)
# * ChrM is removed
# * chromosome-wise list
# * real length

# In[30]:


# make a long string of the css (not using unit, but the real length)

def df2longcss(df):
    df_lst_chr=df2chr_df(df)
    # remove the microchondria DNA from df_lst_chr
    if df_lst_chr[-3]["chromosome"].iloc[0]=="chrM":
        del df_lst_chr[-3]
        assert df_lst_chr[-3]["chromosome"].iloc[0]=="chr22"
    else:   
        assert df_lst_chr[-3]["chromosome"].iloc[0]=="chr22"
    
    all_css=[]
    for i in range(len(df_lst_chr)):
        df_chr=df_lst_chr[i]
        css_chr=''
        for j in range(len(df_chr)):
            css_chr+=df_chr["length"].iloc[j]*df_chr["state_seq"].iloc[j]
        all_css.append(css_chr)  
    return all_css


# ### 2-4-2. Unit-length CSS

# #### Function: `df2unitcss`
# 
# * make a unit-length string of the css (not the real length, but **200-bp resolution unit**)
# * ChrM is removed
# * chromosome-wise list
# * unit length (chromatin is annotated per 200 bp)

# In[31]:


# make a long string of the css (unit length, not the real length)

def df2unitcss(df):
    df_lst_chr=df2chr_df(df)
    # remove the microchondria DNA from df_lst_chr
    if df_lst_chr[-3]["chromosome"].iloc[0]=="chrM":
        del df_lst_chr[-3]
        assert df_lst_chr[-3]["chromosome"].iloc[0]=="chr22"
    else:   
        assert df_lst_chr[-3]["chromosome"].iloc[0]=="chr22"
    
    all_unit_css=[]
    for i in range(len(df_lst_chr)):
        df_chr=df_lst_chr[i]
        css_chr=''
        for j in range(len(df_chr)):
            css_chr+=df_chr["unit"].iloc[j]*df_chr["state_seq"].iloc[j]
        all_unit_css.append(css_chr)  
    return all_unit_css


# ## 2-5. Chromatin State Statistics
# 
# 
# #### Function: `prop_data2df`
# 
# * With 15th state (including 15ths state)
# * `'../database/conserv_overlap/'` contains the emission of the state (occupation of the state on the genome) 
# * State distribution on genome across all the cell types
# * Mostly for visualization
#     <img src="./desc_img/prop_data2df.png" width="500">

# In[32]:


def prop_data2df(path='../database/conserv_overlap/'):
    file_list=[os.path.join(path, file) for file in os.listdir(path)]
    
    temp_df=pd.read_csv(file_list[0],sep='\t', lineterminator='\n')
    init_col=pd.DataFrame(temp_df["state (Emission order)"])
    init_col=init_col.rename(columns={"state (Emission order)":"state"})
    for file in file_list:
        file_name=file.split('/')[3]
        sample_name=file_name.split('_')[0]

        prop_data=pd.read_csv(file, sep='\t', lineterminator='\n')
        prop=prop_data["Genome %"]
        temp_df=pd.concat([init_col,prop], axis=1)
        temp_df=temp_df.rename(columns={"Genome %":str(sample_name)})
        init_col=temp_dfx
    
    # show the result df (first col=state, other col=samples)
    temp_df.drop(temp_df.tail(1).index, inplace=True) # remove the last row (100%)
    
    # transposed and trimmed df (col+1=state no. row=samples)
    trans_df=temp_df.T
    trans_df.drop(trans_df.head(1).index, inplace=True)
    trans_df.columns=temp_df["state"].to_list()
    
    state_list=temp_df["state"].to_list()
    
    ################### create a plot for genome proportion across cell types
    fig=plt.figure(figsize=(9,5))
    ax=fig.add_subplot(111)
    for i in range(len(state_list)):
        state=list(css_color_dict.keys())[i]
        state_as_colname=list(trans_df.columns)[i]

        color=tuple([elm/255 for elm in css_color_dict[state]])

        bp=ax.boxplot(trans_df.iloc[:,i],widths=0.65,positions = [i+1], notch=True,patch_artist=True, 
                     boxprops=dict(facecolor=color, color="gray"),whiskerprops=dict(color="gray", linewidth=2),
                     medianprops=dict(color=color, linewidth=2),
                     capprops=dict(color="gray", linewidth=2),
                     flierprops=dict(markeredgecolor=color, markeredgewidth=1.5))
    plt.xticks(list(range(1,16)),list(trans_df.columns))
    plt.xlabel("Chromatin state")
    plt.ylabel("Genome [%]\n across Different Cell Types")
    fig.autofmt_xdate(rotation=45)
    plt.show()
    ###################
    
    return temp_df, trans_df


# In[33]:


# temp_df, trans_df=prop_data2df(path='../database/conserv_overlap/')


# # 3. Cutting the telomere: where to cut?
# **[back to index](#Index)**

# ## 3-1. Quiescent state distribution
# How Quiescent states are distributed on the whole genome?
# 
# #### Function: `UnitCSS_Q_Dist`
# 
# * Input: df, chromosome number
# * Output: `q_index` index of genome (not normalized) where Quiescent states are found.

# In[34]:


# index list for O state in unit-length css sequence:
def UnitCSS_Q_Dist(df, chr_no=1):
    all_unit_css=df2unitcss(df)
    chr_unit_css=all_unit_css[chr_no]
    q_index=[]
    for i,state in enumerate(chr_unit_css):
        if state=="O":
            q_index.append(i)
    ######## uncomment this block if you want to draw the histogram!
#     fig=plt.figure(figsize=(8,4))
#     plt.hist(q_index, 30, histtype="step", color='orange')
# #     sns.histplot(q_index, kde=False, color='orange', bins=30, element="step", fill=False)

#     plt.xlabel("Position")
#     plt.ylabel("number of 'O' state")
#     plt.show()
    return q_index


# #### Function: `all_chr_UnitCSS_Q_Dist(df, normalization=True)`
# 
# * Input: df, normalization (T/F, default=T)
# * Output: list of list, the element of which is a list contains the position index of the Q state in a chromosome
#     * Normalization True: `all_chr_q_index_norm`
#     * Normalization False: `all_chr_q_index`
# * Graph (distribution histogram)
# <img src="./desc_img/all_chr_UnitCSS_Q_Dist.png" width="400" height="150">

# In[35]:


def all_chr_UnitCSS_Q_Dist(df,normalization=True):
    
    """
    input: df (the dataframe acquired by bed2df_expanded function for a chromatin state bed file)
    output: all_chr_q_index_norm (normalization ON) / all_chr_q_index (normalization OFF)
    option: normalization (all chromosome length= 0 to 1 for drawing a dist. graph)
    """
    
    all_unit_css=df2unitcss(df)  # a list of unit-css of df sample, chromosome wise
       
    fig=plt.figure(figsize=(12,6))
    ax = plt.subplot(111)
    all_chr_q_index=[] # list of list (raw data)
    all_chr_q_index_norm=[] # list of list (normalized data)
    
    for i in range(len(all_unit_css)):
        q_index=UnitCSS_Q_Dist(df, chr_no=i)
        all_chr_q_index.append(q_index)
        
        ########### normalization here ###########
        q_index_norm=[elm/q_index[-1] for elm in q_index]
        ##########################################
        all_chr_q_index_norm.append(q_index_norm)
        if i <=21:
            chr_name="chr"+str(i+1)
        elif i==23:
            chr_name="chrX"
        else:
            chr_name="chrY"

        grad_color=plt.cm.coolwarm(i*10)
#         ax.hist(q_index_norm, 100, color=grad_color, ec='white', alpha=0.5, label=chr_no)
        ax.hist(q_index_norm, 50, color=grad_color, histtype="step", label=chr_name)

    ### show only the normalized disribution
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # Shrink current axis's height by 20% on the bottom
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Normalized Position")
    plt.ylabel("number of 'O' state")
    plt.grid(b=None)
    
    if normalization:
        return all_chr_q_index_norm
    else:
        return all_chr_q_index


# ## 3-2. Cut the telomere region on CSS and save the file

# ### 3-2-1. Random cut
# **Pretrain data was generated by this function, and chromosome-wise data are saved at `database/wo_telo`**

# #### Function: `chr_cssWOtelo_ranCUT_Kmer`
# 
# * Cut the list of CSS into trainable size and save after trimming the telometer region 
#     >1. Cut the telomere (50 units=10000 bp)
#     >2. Select the sample `df` and chromosome number `chr_no` to take
#     >3. Determine the range of random cut of the string (e.g. `100` to `2000`)
#     >4. Determine the `k` for making kmer
#     
# * Input: df,chr_no,num1=5,num2=510, k=3, weight_rn=False, v_name="v1.01"
# > weight_rn `True`: 50% cut into 510, 50% cut randomly between 5 and 510 <br>
# > weight_rn `False` : 100% cut randomly between 5 and 510
# * Output file name: `k_wo_telo_v1.01.txt`, v1 stands for version 1 (considering telomere length) 
# * Usage: `chr_cssWOtelo_ranCUT_Kmer(df,1,100,200,6)`
# 
# * Version control (e.g. v1.01)
#      * v1: version 1, telomere length set at 50 units.
#      * .01: not weighted random, from 5 to 510 
# 
# >*Output message* <br>
# >unit-length css of chr1 cut randomly(weighted range:5-510) for 3mer was saved at '../database/wo_telo/'

# In[36]:


###############################
#    Chromosome-wise save     #
###############################

# randomly cut the string 

def chr_cssWOtelo_ranCUT_Kmer(df,chr_no,num1=5,num2=510, k=3, weight_rn=False, v_name="v1.01"):
    """
    Usage: chr_cssWOtelo_ranCUT_Kmer(df,chr_no,num1,num2, weight_rn, k, v_name)
    
    - df: expanded version of 1 sample bed file
    - chr_no: no. of chromosome
    - num1: cut range start
    - num2: cut range end
    - weight_rn: 
      if True: random with weighted, 50% of chance to be num2, 50% random between num1 and num2
      if False: random between num1 and num2
    - k: kmer
    - v_name: version name to be used as a file name 
      (Conventionally, 01 was used for weighted_rn False, 02 for True
       v1 just stands for telomere is set to be 50 unit)
    
    output: randomly cut w15 css for one chromosome unit-length css
    """
    all_unit_css=df2unitcss(df)
    ch1_unit_css=all_unit_css[chr_no]
    ch1_unit_css_wotelo=ch1_unit_css[50:-50] #cut the telomere

    splitted=[]
    prev=0

    ori_lst=[elm for elm in range(num1,num2+1)]   # list of num between num1 and num2
    sin_lst=[num2]*len(ori_lst)   # list of all num2 (length is the same of ori_lst)
    tot_lst=ori_lst+sin_lst
    
    while True:
        
        if weight_rn:
            n=random.choice(tot_lst)

        else:
            n=random.randint(num1,num2)
        
        splitted.append(ch1_unit_css_wotelo[prev:prev+n])
        prev=prev+n
        if prev >= len(ch1_unit_css_wotelo)-1:
            break
   
    ch1_unit_css_wotelo_kmer=[seq2kmer(item, k) for item in splitted]
    
      
    path='../database/wo_telo/'
    fn_base="chr"+str(chr_no)+"_"+str(k)+"_wo_telo_"+v_name   # version 1.01_pre (Oct. 2022) : telo 50 unit, rn 200-1000
                                                              # version 1.01 (Oct. 2022) : telo 50, rn 5 - 510
    ext=".txt"
          
    fn=path+fn_base+ext  # file name

    with open(fn,"w") as save_file:
        save_file.write("\n".join(ch1_unit_css_wotelo_kmer))
          
    return print("unit-length css of chr{} cut randomly(weighted range:{}-{}) for {}mer was saved at {}".format(chr_no, num1, num2, k,fn))


# #### Function: `cell_cssWOtelo_ranCUT_Kmer`
# 
# * Conduct the same work but now cell-wise, not chromosome-wise

# In[1]:


###############################
#    Cell-wise save     #
###############################

# randomly cut the string 

def cell_cssWOtelo_ranCUT_Kmer(all_file_path=all_files, cell_num=0, num1=5,num2=510, k=4, weight_rn=False, v_name="v1.01"):
    """
    Usage: chr_cssWOtelo_ranCUT_Kmer(df,chr_no,num1,num2, weight_rn, k, v_name)
    
    - all_file_path: the list of all_files (see css_utility)
    - cell_num: the number of cell in the list of all_files
    - num1: cut range start
    - num2: cut range end
    - weight_rn: 
      if True: random with weighted, 50% of chance to be num2, 50% random between num1 and num2
      if False: random between num1 and num2
    - k: kmer
    
    output: randomly cut w15 css for one chromosome unit-length css
    """
    target_cell=all_file_path[cell_num]
    cell_id=target_cell.split("/")[-1].split("_")[1]
    assert type(cell_id)==str,"Check the all_file path"
    
    df=bed2df_expanded(target_cell)    
    all_unit_css=df2unitcss(df)
    all_chr=len(all_unit_css)
    
    all_chr_unit_css_wotelo_kmer=[]
    
    for chr_no in range(all_chr):        
    
        chr_unit_css=all_unit_css[chr_no]
        chr_unit_css_wotelo=chr_unit_css[50:-50] #cut the telomere

        splitted=[]
        prev=0

        ori_lst=[elm for elm in range(num1,num2+1)]   # list of num between num1 and num2
        sin_lst=[num2]*len(ori_lst)   # list of all num2 (length is the same of ori_lst)
        tot_lst=ori_lst+sin_lst

        while True:

            if weight_rn:
                n=random.choice(tot_lst)

            else:
                n=random.randint(num1,num2)

            splitted.append(chr_unit_css_wotelo[prev:prev+n])
            prev=prev+n
            if prev >= len(chr_unit_css_wotelo)-1:
                break

        chr_unit_css_wotelo_kmer=[seq2kmer(item, k) for item in splitted]
        all_chr_unit_css_wotelo_kmer.append(chr_unit_css_wotelo_kmer)
        
    all_unit_css_wotelo_kmer=flatLst(all_chr_unit_css_wotelo_kmer)
    
    path='../database/wo_telo/'
    fn_base="cell"+cell_id+"_"+str(k)+"_wo_telo_"+v_name   # version 1.01_pre (Oct. 2022) : telo 50 unit, rn 200-1000
                                                              # version 1.01 (Oct. 2022) : telo 50, rn 5 - 510
    ext=".txt"
          
    fn=path+fn_base+ext  # file name

    with open(fn,"w") as save_file:
        save_file.write("\n".join(all_unit_css_wotelo_kmer))
    
    
    return print("unit-length css of cell ID {} cut randomly(weighted range:{}-{}) for {}mer was saved at {}".format(cell_id, num1, num2, k,fn))


# ### 3-2-1. Kmerized data visualization

# #### Function: `dataLengCompo`
# * Input: data path, k (of kmer), color, bins, dna or not (default=false)
# 

# In[37]:


def dataLengCompo(path, k, color="teal", bins=15, dna=False):
    """
    Create a histogram of data length (elements before k-merization in the training dataset list)
    """
    file_name=path
    with open(file_name) as f:
        len_lst=[]
        for line_no, line in enumerate(f):
            if dna:
                line_len=int((len(line)-1)/k)+(k-1)  # -1 comes from the space only between DNA sequence kmer
                if line_len!=0:
                    len_lst.append(line_len)               
            else:
                line_len=int(len(line)/(k+1))+(k-1)  # reduced 
#                 line_len=int(len(line)/(k+1))*k # +1 comes from the space after the kmers
                len_lst.append(line_len)
                
    fig=plt.figure(figsize=(6,4))
    
    s=sns.histplot(len_lst, kde=False, color=color, log_scale=True, bins=bins, element="step", fill=False)
    sns.set_style("whitegrid")
    plt.xlabel("Length of each element in training dataset", fontsize=12)
    plt.xlim([1,10000])
    plt.show()
    return  


# ## 3-3. Cut the chromatin states : genic or non-genic area

# ### 3-3-1. Genic area
# #### Function: `compGene2css`
# 
# * Input: whole_gene_file, df
# * Output: `css_gene_lst_all` list of list that css for genic region per chromosome (which can be utilized very frequently after this)
# * The output is pickled as `"../database/temp_files/css_gene_lst_all"`

# In[38]:


def compGene2css(whole_gene_file,df):   # note that the result is also overlapped css... 
    """
    Input: Reference gene file, df (CSS)
    Output: list of chromosome-wise list that contains the css at genic area only.
    """
    g_lst_chr=whGene2GLChr(whole_gene_file) # list of gene table df per chromosome
    css_lst_chr=df2longcss(df) # list of long css per chromosome
    total_chr=len(g_lst_chr)
    
    css_gene_lst_all=[]
    for i in tqdm_notebook(range(total_chr)):
        css=css_lst_chr[i]   # long css of i-th chromosome
        gene_df=g_lst_chr[i] # gene df of i-th chromosome
        
        css_gene_lst_chr=[]
        for j in range(len(gene_df)):
            g_start=gene_df["TxStart"].iloc[j]-1  # python counts form 0
            g_end=gene_df["TxEnd"].iloc[j]+1      # python excludes the end
            
            css_gene=css[g_start:g_end]           # cut the gene area only
            css_gene_lst_chr.append(css_gene)     # store in the list
          
        css_gene_lst_all.append(css_gene_lst_chr)  # list of list
    
    assert len(css_gene_lst_all)==total_chr
    return css_gene_lst_all


# #### Function: `countGeneCss`
# * How many css data strips are in the Non-genic (intergenic) region?
# * How long each css data strips are in the Non-genic (intergenic) region?
# * Input: `css_gene_lst_all`, the result list from `compGene2css(whole_gene_file,df)`. (Also pickled at `"../database/temp_files/css_gene_lst_all"`
# * Output: Two lists (`g_css_cnt_all` and `g_css_len_all`) and their distribution histogram
# 
# <img src="./desc_img/countGeneCss.png" width="600" height="300">

# In[39]:


def countGeneCss(css_gene_lst_all):
    g_css_cnt_all=[]
    g_css_len_all=[]
    tot_chr=len(css_gene_lst_all)
    for chr_no in range(tot_chr):
        g_chr_lst=css_gene_lst_all[chr_no]
        g_css_cnt_all.append(len(g_chr_lst))
        g_css_len_chr=[]
        for i in range(len(g_chr_lst)):
            g_css_len=len(g_chr_lst[i])
            g_css_len_chr.append(g_css_len)  # to let it iterate for chr!
        g_css_len_all.append(g_css_len_chr)
    g_css_len_all=flatLst(g_css_len_all) 
    
    g_css_len_all=list(filter(lambda elm: elm!=0, g_css_len_all))  # remove 0s
        
    # visualization for ng_css_cnt_all (no. of data strips per chromosome)
    fig,(ax1, ax2)=plt.subplots(1,2,figsize=(12,4), sharey=False)
    ax1=sns.histplot(g_css_cnt_all, bins=12, color="cadetblue", element="step", fill=False, ax=ax1)
    ax1.set_xlabel("Count of data strip on Genic region", fontsize=13)
    ax1.set_ylabel("Count", fontsize=13)
    ax1.grid(b=None)
    ax1.xaxis.grid(None)
    ax1.yaxis.grid()
    
    # visualization for ng_css_cnt_all (no. of data strips per chromosome)
    ax2=sns.histplot(g_css_len_all, bins=15, log_scale=True, color="crimson", element="step", fill=False, ax=ax2)
    ax2.set_xlabel("Length of CSS on Genic region", fontsize=13)
    ax2.set_ylabel("Count", fontsize=13)
    ax2.grid(b=None)
    plt.grid(False)
            
    return g_css_cnt_all,g_css_len_all


# ### 3-3-2. Non-genic area (intergenic region)
# 
# * The problem in evaluating the intergenic region is that the positions of genes are frequently duplicated. Therefore, the gene table shares lots of same start and end position.
#     1. First, we need to take a look how many genes are duplicated at the start and end position.
#     2. Second, gene table has been collapsed to remove the overlaps.

# #### Function: `count_samePos` (To take a look how many genes are overlapped)
# * Input: `whole_gene_file='../database/RefSeq/RefSeq.WholeGene.bed'`
# * Output: 2 dataframes (`df_cnt` and `df_pro`) and visualization for them in violin plot
#     * `df_cnt` : Chromosome-wise list of the count of the duplicated gene Start and End position on genome
#     * `df_pro` : Chromosome-wise list of the proportion of the duplicated gene Start and End position on genome (per gene)    
#     
# <img src="./desc_img/gene_dup_start_end_vis.png" width="500" height="200">

# In[40]:


# function to visualize how many genes are sharing the start and end position on genome

def count_samePos(whole_gene_file='../database/RefSeq/RefSeq.WholeGene.bed'):
    g_df_chr_lst=whGene2GLChr(whole_gene_file)
    cnt_same_start_all=[]
    pro_same_start_all=[]
    cnt_same_end_all=[]
    pro_same_end_all=[]
    tot_chr_no=len(g_df_chr_lst)
    
    ########### count the same start position ###########
    def count_sameStart(g_df_chr_lst,chr_no):
        cnt_same_start=0
        tot_start=len(g_df_chr_lst[chr_no])
        for i in range(len(g_df_chr_lst[chr_no])):
            chr1=g_df_chr_lst[chr_no]["TxStart"]
            if i==0:
                continue
            elif chr1.iloc[i]==chr1.iloc[i-1]:
                cnt_same_start+=1  # how many same start in rows
            else:
                continue
        prop_same_start=cnt_same_start/tot_start
        return cnt_same_start, prop_same_start
    
    ########### count the same end position ############
    def count_sameEnd(g_df_chr_lst,chr_no):
        cnt_same_end=0
        tot_end=len(g_df_chr_lst[chr_no])
        for i in range(len(g_df_chr_lst[chr_no])):
            chr1=g_df_chr_lst[chr_no]["TxEnd"]       
            if i==0:
                continue
            elif chr1.iloc[i]==chr1.iloc[i-1]:
                cnt_same_end+=1  # how many same start in rows
            else:
                continue
        prop_same_end=cnt_same_end/tot_end
        return cnt_same_end, prop_same_end
    ####################################################
    
    for chr_no in tqdm_notebook(range(tot_chr_no)):
        cnt_same_start, prop_same_start = count_sameStart(g_df_chr_lst,chr_no)
        cnt_same_end, prop_same_end = count_sameEnd(g_df_chr_lst,chr_no)
        
        cnt_same_start_all.append(cnt_same_start)
        pro_same_start_all.append(prop_same_start)
        cnt_same_end_all.append(cnt_same_end)
        pro_same_end_all.append(prop_same_end)
        
    dict_cnt={"cnt_same_start":cnt_same_start_all, "cnt_same_end":cnt_same_end_all}
    dict_pro={"pro_same_start":pro_same_start_all, "pro_same_end":pro_same_end_all}
    df_cnt=pd.DataFrame(dict_cnt)
    df_pro=pd.DataFrame(dict_pro)
    
    ###### Visualization ######
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5), sharey=False)
    ax1=sns.violinplot(data=df_cnt, palette="pastel", linewidth=0.7, saturation=0.5, ax=ax1)
    ax1.set_ylabel("Count", fontsize=15)
    ax2=sns.violinplot(data=df_pro, palette="husl", linewidth=0.7, saturation=0.5, ax=ax2)
    ax2.set_ylim([0.2,0.8])
    ax2.set_ylabel("Proportion", fontsize=15)
    plt.show()

    return df_cnt, df_pro


# #### Function: `removeOverlapDF` and `gene_removeDupl`
# 
# * Main function: `gene_removeDupl`
# * `removeOverlapDF`: function used inside the main function.
# * To acquire final collapsed gene table, run `gene_removeDupl`

# In[41]:


def removeOverlapDF(test_df):    
    new_lst=[]
    for i in range(len(test_df)):
        start=test_df["TxStart"].iloc[i]
        end=test_df["TxEnd"].iloc[i]

        exist_pair=(start,end)

        if i==0:
            new_pair=exist_pair
            new_lst.append(new_pair)        
        else:
            start_pre=test_df["TxStart"].iloc[i-1]
            end_pre=test_df["TxEnd"].iloc[i-1]

            # first, concatenate all the shared start
            if start==start_pre:
                new_end=max(end, end_pre)
                new_pair=(start, new_end)
            # second, concatenate all the shared end
            elif end==end_pre:
                new_start=min(start, start_pre)
                new_pair=(new_start, end)
            else:    
                new_pair=exist_pair

        new_lst.append(new_pair) 
    new_lst=list(dict.fromkeys(new_lst))
    
    mod_lst=[[start, end] for (start, end) in new_lst] # as a list element

    for j, elm in enumerate(mod_lst):
        start, end = elm[0], elm[1]

        if j==0:
            continue
        else:
            start_pre=mod_lst[j-1][0]
            end_pre=mod_lst[j-1][1]

            if end_pre>=end:
                mod_lst[j][0]=mod_lst[j-1][0]  # if end_pre is larger than end, replace start as start_pre
                mod_lst[j][1]=mod_lst[j-1][1]  # if end_pre is larger than end, replace end as end_pre

            elif start <=end_pre:
                mod_lst[j][0]=mod_lst[j-1][0]  # current start=start_pre
                mod_lst[j-1][1]=max(mod_lst[j][1],mod_lst[j-1][1])  # end_pre = end

            else:
                continue
           
    mod_lst=[tuple(elm) for elm in mod_lst]
    fin_lst=list(dict.fromkeys(mod_lst))
    gene_collapsed_df=pd.DataFrame(fin_lst, columns=["TxStart", "TxEnd"])
 
    return gene_collapsed_df


# In[42]:


def gene_removeDupl(whole_gene_file='../database/RefSeq/RefSeq.WholeGene.bed'):
    g_df_chr_lst=whGene2GLChr(whole_gene_file)
    new_gene_lst_all=[]
    for chr_no in range(len(g_df_chr_lst)):
        gene_df=g_df_chr_lst[chr_no]
        gene_collapsed_df=removeOverlapDF(gene_df)
        new_gene_lst_all.append(gene_collapsed_df)
    return new_gene_lst_all # list of chromosome-wise dataframe for collapsed gene table


# #### Function: `compNonGene2css`
# * This function extracts the css on the non-genic (intergenic) area of the genome.
# * The function `gene_removeDupl` was used here, for extracting the non-genic region index.
# * Input: `whole_gene_file` and `df` (from the css bed file)
# * Output: `css_Ngene_lst_all` The CSS on the non-genic region

# In[43]:


def compNonGene2css(whole_gene_file,df):
    """
    Input: Reference gene file, df (CSS)
    Output: list of chromosome-wise list that contains the css at "non-genic" area only.
    """
    
    print("Extracting the CSS on the intergenic region ...")

    ########### new fancy gene table without overlap ###########
    new_gene_lst_all=gene_removeDupl(whole_gene_file)
    ############################################################
    
    css_lst_chr=df2longcss(df) # list of long css per chromosome
    total_chr=len(new_gene_lst_all)
    
    css_Ngene_lst_all=[]
        
    for i in tqdm_notebook(range(total_chr)):
        css=css_lst_chr[i]   # long css of i-th chromosome
        gene_df=new_gene_lst_all[i] # gene df of i-th chromosome
        
        assert gene_df["TxStart"].iloc[0]>=1, "Gene starts from the very first location at {}-th chromosome.".format(i)
        assert gene_df["TxEnd"].iloc[-1]<=len(css), "Gene ends at the very last location at {}-th chromosome.".format(i)  
                
        css_Ngene_lst_chr=[]        
        for j in range(len(gene_df)):
            if j==0:
                ng_start=1 # to avoid any "zero" causing problem 
                ng_end=gene_df["TxStart"].iloc[j]
#                 print("j: {} | ng_start: {} - ng_end: {} ".format(j, ng_start, ng_end)) # for checking
            elif j==len(gene_df)-1: 
                ng_start=gene_df["TxEnd"].iloc[j]
                ng_end=len(css)
#                 print("j: {} | ng_start: {} - ng_end: {} ".format(j, ng_start, ng_end)) # for checking
            else:
                ng_start=gene_df["TxEnd"].iloc[j-1]
                ng_end=gene_df["TxStart"].iloc[j]
#                 print("j: {} | ng_start: {} - ng_end: {} ".format(j, ng_start, ng_end)) # for checking 
        
            css_Ngene=css[ng_start:ng_end]
            css_Ngene_lst_chr.append(css_Ngene)
        
        css_Ngene_lst_all.append(css_Ngene_lst_chr) 
        
    assert len(css_Ngene_lst_all)==total_chr
    print("Done!")
    
    return css_Ngene_lst_all


# #### Function: `countNgeneCss`
# * How many css data strips are in the Non-genic (intergenic) region?
# * How long each css data strips are in the Non-genic (intergenic) region?
# * Input: `css_Ngene_lst_all`, the result list from `compNonGene2css(whole_gene_file,df)`. (Also pickled at `"../database/temp_files/css_Ngene_lst_all"`
# * Output: Two lists (`ng_css_cnt_all` and `ng_css_len_all`) and their distribution histogram
# 
# <img src="./desc_img/countNgeneCss.png" width="600" height="300">

# In[44]:


def countNgeneCss(css_Ngene_lst_all):
    ng_css_cnt_all=[]
    ng_css_len_all=[]
    tot_chr=len(css_Ngene_lst_all)
    for chr_no in range(tot_chr):
        ng_chr_lst=css_Ngene_lst_all[chr_no]
        ng_css_cnt_all.append(len(ng_chr_lst))
        ng_css_len_chr=[]
        for i in range(len(ng_chr_lst)):
            ng_css_len=len(ng_chr_lst[i])
            ng_css_len_chr.append(ng_css_len)  # to let it iterate for chr!
        ng_css_len_all.append(ng_css_len_chr)
    ng_css_len_all=flatLst(ng_css_len_all) 
    
    ng_css_len_all=list(filter(lambda elm: elm!=0, ng_css_len_all))  # remove 0s
        
    # visualization for ng_css_cnt_all (no. of data strips per chromosome)
    fig,(ax1, ax2)=plt.subplots(1,2,figsize=(12,4), sharey=False)
    ax1=sns.histplot(ng_css_cnt_all, bins=12, color="navy", element="step", fill=False, ax=ax1)
    ax1.set_xlabel("Count of data strip on Intergenic region", fontsize=13)
    ax1.set_ylabel("Count", fontsize=13)
    ax1.grid(b=None)
    ax1.xaxis.grid(None)
    ax1.yaxis.grid()
    
    # visualization for ng_css_cnt_all (no. of data strips per chromosome)
    ax2=sns.histplot(ng_css_len_all, bins=15, log_scale=True, color="maroon", element="step", fill=False, ax=ax2)
    ax2.set_xlabel("Length of CSS on Intergenic region", fontsize=13)
    ax2.set_ylabel("Count", fontsize=13)
    ax2.grid(b=None)
    plt.grid(False)
            
    return ng_css_cnt_all,ng_css_len_all


# ### 3-3-3. Genic or Non-genic raw-length CSS to unit-length CSS
# 
# * For the genic and intergenic region, the css is the raw length, not the unit length. To keep the same training data condition, the data should be formed as unit length (200-bp).
# * So, the purpose is to convert `css_Ngene_lst_all` and `css_gene_lst_all` into the unit-length version of them.
# * To do this job, 2 functions are required : `long2unitCSS` and `Convert2unitCSS_main`

# #### Function (preliminary) : `long2unitCSS` (included in the main function)
# 
# * As the **preliminary** function, `long2unitCSS`, investigates 
#     1. The sequence of the state (letter) appears -> as a list of string
#     2. How many times the state appears -> as a list of list (numbers)   
# 
# * Input: `long_css_lst` which is a list of string.
# 
# * Output
#     1. `let_str_lst_all`: The list of string that only shows the sequence of the css 
#     2. `unit_cnt_lst_all`: The list of list of unit-length of each state in the list `let_str_lst_all`

# In[45]:


# the idea is to separate, count, combine
def long2unitCSS(long_css_lst, unit=200):
    """
    * description *
    long_css is the result of the function "df2longcss" (real length css), 
    and this function aims to convert it into the result of the function "df2unitcss",
    which is shortest possible version of the css.
    Why? because pre-train data for ChromBERT is done by unit-length, 
    and the genic/intergenic css is acquired as a long-css
    
    Input: long_css_lst (type=list) acquired by df2longcss(df) and the unit length bp (default=200 bp)
    Output: let_str_lst_all (list of unit state) and unit_cnt_lst_all (list of list)
    """
    assert type(long_css_lst)==list, "Check the input type: it should be a list, but now it's {}".format(type(long_css_lst))
    assert type(long_css_lst[0])==str, "Check the type of input element: it should be a string, but it's {}".format(type(long_css_lst[0]))
    let_str_lst_all=[]
    unit_cnt_lst_all=[]
    for elm in long_css_lst:
        unit_str=''
        unit_cnt_lst=[]
        unit_cnt=0
        for i, let_str in enumerate(elm):
            if i==0:     # handling the first letter
                unit_str+=let_str
                unit_cnt=1
            elif i==len(elm)-1:    # handling the final letter
                unit_cnt+=1
                unit_cnt_lst.append(int(unit_cnt/unit)) 
            elif let_str==elm[i-1]:
                unit_cnt+=1      
            elif (let_str!=elm[i-1] and i!=len(elm)-1):
                unit_str+=let_str            
                unit_cnt+=1
                unit_cnt_lst.append(int(unit_cnt/unit))  
                unit_cnt=1
            else:
                continue
        let_str_lst_all.append(unit_str)
        unit_cnt_lst_all.append(unit_cnt_lst)
    return let_str_lst_all, unit_cnt_lst_all


# #### Function (main): `Convert2unitCSS_main`
# * Input: `css_gene_lst_all` or `css_Ngene_lst_all`, the raw-length css on genic and non-genic regions, and the unit (default=200, as the css are annotated per )
# * Output: `css_unit_lst_all`, the list of chromosome-wise list of unit-length css.

# In[46]:


def Convert2unitCSS_main(css_lst_all, unit=200): # should be either css_gene_lst_all or css_Ngene_lst_all
    """
    Input: css_gene_lst_all or css_Ngene_lst_all, the list of chromosome-wise list of the css in genic, intergenic regions.
    Output: css_gene_unit_lst_all or css_Ngene_unit_lst_all
    """
    print("Converting css from the raw length into unit-length ... ")
    css_unit_lst_all=[]
    for chr_no in tqdm_notebook(range(len(css_lst_all))):
        css_chr_lst=css_lst_all[chr_no]
        css_chr_unit_lst=[]
        let_str_lst_all, unit_cnt_lst_all=long2unitCSS(css_chr_lst, unit=unit)
        unit_css_lst=['']*len(let_str_lst_all)
        for i, let_str in enumerate(let_str_lst_all):
            for j in range(len(let_str)-1):
                unit_css_lst[i]+=let_str[j]*unit_cnt_lst_all[i][j] # only unit will be multiplied!
        unit_css_lst=[css for css in unit_css_lst if css!='']  # remove the empty element
        css_unit_lst_all.append(unit_css_lst)
    print("Done!")
    return css_unit_lst_all


# Now following files are saved at : `../database/temp_files/` 
# * `css_gene_unit_lst_all` : The unit-length css on the genic area
# * `css_Ngene_unit_lst_all`: The unit-length css on the intergenic area

# ### 3-3-4. Cut the unit-length css into trainable size and kmerize it
# 
# #### Function: `chr_css_CUT_Kmer`
# * Input: Unit-length css list of chromosome-wise list (e.g. `css_gene_unit_lst_all` or `css_Ngene_unit_lst_all`)
# * Output: 
#     1. `splitted` : List of strings before kmerization (to visualize later)
#     2. `kmerized_unit_css` :  List of strings after kmerization (to use as a trinable data)
#  
# * Usage (e.g. Generate a 3-mer traning data from 2nd chromosome in intergenic area)
# > `splitted, kmerized_unit_css=chr_css_CUT_Kmer(css_Ngene_unit_lst_all, 2, 510, 3)`
# * And the data can be stored like 
# > `with open("../database/fine_tune/genic_and_intergenic/3mer/chr2_Ngene.txt", "w") as f:         f.write("\n".join(kmerized_unit_css))`
#        
# * The reason why the above code for saving is not included is because it takes too much time.. dunno why

# In[47]:


# Cut the unit-length string (input: unit-css, not df)
def chr_css_CUT_Kmer(unit_css, chr_no, cut_thres, k):
    """    
    Prepare kmer dataset for unit_css, as is if length<=510, else cut it to be length>510   
    Usage: chr_css_CUT_Kmer(unit_css, chr_no, cut_thres, k)
    
    - unit_css: list of chromosome-wise list of unit-length css (e.g. css_gene_unit_lst_all)
    - chr_no: no. of chromosome
    - cut_thres: length of split, default=510
    - k: kmer
    
    Output: 1. splitted (before kmerization) 2. kmerized_unit_css (after kmerization) 
    """    
    chr_unit_css=unit_css[chr_no]   # designated chromosome no.    
    splitted=[] # bucket for the all the splitted strings   
    cnt_short, cnt_long=0,0
    for css_elm in chr_unit_css:
        if len(css_elm) <=cut_thres:
            splitted.append(css_elm)
            cnt_short+=1
        else:
            cnt_long+=1
            prev=0
            while True:
                splitted.append(css_elm[prev:prev+cut_thres])
                prev+=cut_thres
                if prev>=len(css_elm)-1:
                    break                   
    kmerized_unit_css=[seq2kmer(item, k) for item in splitted]
    long_pro=cnt_long/(cnt_long+cnt_short)
    
    return splitted, kmerized_unit_css


# #### Function: `saveCUTs_all`
# 
# * Simply save the file created from the above fucntion: k-merized genic and intergenic unit-length css
# * 3mer, 4mer files are already stored at `../database/fine_tune/genic_and_intergenic/`
# * Usage
# > `saveCUTs_all(css_gene_unit_lst_all, 510, 3, gene=True)`
# > saves the css on the genic region after 3-merization.

# In[48]:


def saveCUTs_all(unit_css, cut_thres, k, gene=True):
    for chr_no in range(len(unit_css)):        
        _, kmerized=chr_css_CUT_Kmer(unit_css, chr_no, cut_thres, k)
        chr_num=str(chr_no+1)
        if gene:
            g='gene'
        else:
            g='Ngene'
   
        path="../database/fine_tune/genic_and_intergenic/"
        kmer=str(k)+'mer/'
        folder=g+"/"
        name="chr"+chr_num+"_"+g+".txt"
        f_name=path+kmer+folder+name
        
        with open(f_name, "w") as f:
            f.write("\n".join(kmerized))
    return #print("{}merized files for {} are saved at {}.".format(k,unit_css,path+kmer))


# 

# ### 3-3-5. Fine-tuning data: Dataframe version

# #### Function: `prepFT_gNg`
# * Create a dataframe version of dataset, accommodating the same number of genic and non-genic region unit css.
# * Input: `path` (for the specific task), `k`, `sampling_no` (number of chromosome you want to pick as a random no.)
# * Output: `df_g_ng_all` the dataframe containing same amount of genic/non-genic css strips

# In[49]:


# preparing the dataframe-version for generating train and dev dataset
def prepFT_gNg(path="../database/fine_tune/genic_and_intergenic/", k=4, sampling_no=10):
    dir_k=path+str(k)+"mer/"
    
    dir_g=dir_k+"gene/"
    dir_ng=dir_k+"Ngene/"
    g_files=os.listdir(dir_g)
    ng_files=os.listdir(dir_ng)
    all_g_files=file_list_maker(dir_g,g_files)
    all_ng_files=file_list_maker(dir_ng,ng_files)
    
    g_len_all,ng_len_all=[],[]
    df_ng_all,df_g_all=[],[]
    
    ### for Ngene data
    for chr_ng in all_ng_files:
        df_ng=pd.read_csv(chr_ng, header=None, names=["sequence"], sep="\n")
        df_ng["label"]=0        
        ng_len=len(df_ng)  # only for checking length
        ng_len_all.append(ng_len)  # only for checking length
        
        df_ng_all.append(df_ng) 
    df_ng_concat=pd.concat(df_ng_all)  # for ng, concatenate all the list
    
    ### for gene data
    sample=random.sample([i for i, elm in enumerate(all_g_files)], sampling_no)
    print("Sampled chromosome for genic region: {}".format(sample))
    for i, chr_g in enumerate(all_g_files):
        df_g=pd.read_csv(chr_g, header=None, names=["sequence"], sep="\n")
        df_g["label"]=1
        g_len=len(df_g)  # only for checking length
        g_len_all.append(g_len)  # only for checking length
        
        if i in sample:   # sampling 
            df_g_all.append(df_g)
        else:
            continue
    df_g_concat=pd.concat(df_g_all)
    
    ### for the length adjustment ###
    if len(df_g_concat)>len(df_ng_concat):
        df_g_concat=df_g_concat[:len(df_ng_concat)] 
    elif len(df_g_concat)<len(df_ng_concat):
        df_ng_concat=df_ng_concat[:len(df_g_concat)]
    assert len(df_g_concat)==len(df_ng_concat)
    
    df_g_ng_all=pd.concat([df_ng_concat,df_g_concat]).sample(frac=1).reset_index(drop=True)  # shuffling    
    
    ### for visualization purpose ###
#     fig, ax = plt.subplots(1,1,figsize=(6,4))
#     ax=sns.histplot(g_len_all, color="teal", element="step", bins=10, fill=False) #cumulative=True
#     ax=sns.histplot(ng_len_all, color="orange", element="step", bins=4, fill=False)
#     plt.title("Cumulative plot of genic/intergenic data size", fontsize=13)
#     ax.set_xlabel("Length of data", fontsize=13)
#     ax.legend(["genic","intergenic"])
#     plt.show()   
        
    return df_g_ng_all


# ### 3-3-6. Fine-tuning data: save files as .tsv

# #### Function: `saveTF_gNg`
# * Fine-tuning files for classifying genic and intergenic area already are saved at `"../database/fine_tune/genic_and_intergenic/"` (4mer only)
# * Input: `df_g_ng_all` (Result from the function `prepFT_gNg`), `path`, `k`, `len_train`, `len_dev`
# * Output: Files are saved at "`path/kmer/`" folder

# In[50]:


def saveTF_gNg(df_g_ng_all, path="../database/fine_tune/genic_and_intergenic/",k=4,len_train=30000,len_dev=1000):
    dir_k=path+str(k)+"mer/"
    df_g_ng_train=df_g_ng_all[:len_train]
    df_g_ng_dev=df_g_ng_all[len_train:len_train+len_dev]    
    
    train_name=dir_k+"train.tsv"
    dev_name=dir_k+"dev.tsv"
    
    df_g_ng_train.to_csv(train_name, sep="\t", index=False)
    df_g_ng_dev.to_csv(dev_name, sep="\t", index=False)
    
    return print("train.tsv and dev.tsv Files are saved at '{}'.". format(dir_k))


# ## 3-4. Count the number of 15th states in genic and non-genic region

# #### Function: `QnonQforCell`
# 
# * Calculate the numbers of genes that contain/ not contain 15th state (Quiescent) for all 127 cells
# * Caution: it takes tremendous of time. Just use pickled output at `"../database/temp_files/"`
# * Input: cell file list, whole gene file
# * Output: `q_cnt_lst` (The number of gene that contains 15th state) / `not_q_cnt_lst` (genes do not have 15th state)
# * Note that you need to flatten it when use

# In[51]:


# for cell-wise count : how many 15th-including genes are there per cell

# caution: takes tremendous of time!
# better make it for a single cell?
# No, it was required, and the result files are pickled at ./temp_files

def QnonQforCell(all_files=all_files,whole_gene_file=whole_gene_file):
    total_cells=len(all_files)
    
    q_cnt_lst=[]
    not_q_cnt_lst=[]
#     for i in range(total_cells):
    for i in tqdm_notebook(range(total_cells)):
        cell_path=all_files[i]
        df=bed2df_expanded(cell_path)
        css_gene_lst_all=compGene2css(whole_gene_file,df)
        
        q_cnt=0
        not_q_cnt=0
        for j in range(len(css_gene_lst_all)):
            css_gene_lst=css_gene_lst_all[j]
            for k in range(len(css_gene_lst)):
                css_gene=css_gene_lst[k]
                if "O" in css_gene:
                    q_cnt+=1
                else:
                    not_q_cnt+=1
        q_cnt_lst.append(q_cnt)
        not_q_cnt_lst.append(not_q_cnt)
    return q_cnt_lst, not_q_cnt_lst


# #### Function: `QnonQforChr`
# * Similar to `QnonQforCell`, but it is a flatten version

# In[52]:


# for chromosome-wise list of list -> flatten list

def QnonQforChr(all_files=all_files,whole_gene_file=whole_gene_file):
#     import itertools
    total_cells=len(all_files)
    
    q_cnt_lst_all=[]
    not_q_cnt_lst_all=[]
#     for i in range(total_cells):
    for i in tqdm_notebook(range(total_cells)):
        cell_path=all_files[i]
        df=bed2df_expanded(cell_path)
        css_gene_lst_all=compGene2css(whole_gene_file,df)
        
        q_cnt_lst=[]
        not_q_cnt_lst=[]
        for j in range(len(css_gene_lst_all)):
            css_gene_lst=css_gene_lst_all[j]
            
            q_cnt=0
            not_q_cnt=0
            for k in range(len(css_gene_lst)):
                css_gene=css_gene_lst[k]
                if "O" in css_gene:
                    q_cnt+=1
                else:
                    not_q_cnt+=1
                    
            q_cnt_lst.append(q_cnt)
            not_q_cnt_lst.append(not_q_cnt)        
        q_cnt_lst_all.append(q_cnt_lst)
        not_q_cnt_lst_all.append(not_q_cnt_lst)

#     flatten the list of list and make it into list
    q_cnt_lst_all=list(itertools.chain.from_iterable(q_cnt_lst_all))
    not_q_cnt_lst_all=list(itertools.chain.from_iterable(not_q_cnt_lst))
        
    return q_cnt_lst_all, not_q_cnt_lst_all


# #### Function: `QnonQforCellHistT1`
# 
# * Input: `q_cnt_lst`, `not_q_cnt_lst` (they are pickled at `"../database/temp_files/"`)
# * How to load the pickled data
#     > `with open("path", "rb") as f:`  <br>
#     > `data=pickle.load(f)`
# * Output: Histogram of the numbers of gene per cell that contains/ don't contain 15th state in the all cell types
# <br><br>
# 
# <img src="./desc_img/qnonq_hist1.png" width="400" height="150">

# In[53]:


# draw a histogram type1 (group by data)
def QnonQforCellHistT1(q_cnt_lst, not_q_cnt_lst, bin_size=20):
    """Run this after executing QnonQforCell"""
    data_w=q_cnt_lst
    data_wo=not_q_cnt_lst

    mu_w, std_w=norm.fit(data_w)
    mu_wo, std_wo=norm.fit(data_wo)

    fig=plt.figure(figsize=(8,4))
    ax=fig.add_subplot(1,1,1)
    ax.hist(data_w, bins=bin_size, alpha=0.3, color="k")
    ax.hist(data_wo, bins=bin_size, alpha=0.5, color="r")

    title='Number of Genic region with/without Quiescent state'
    
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("No. of Genes", fontsize=15)
    plt.xticks(fontsize=12)
    ax.set_ylabel("Counts", fontsize=15)
    ax.legend(["With Q", "Without Q"])
    plt.yticks(fontsize=12)


# #### Function: `QnonQforCellHistT2`
# 
# * Input: `q_cnt_lst`, `not_q_cnt_lst` and `bin_size` (they are pickled at `"../database/temp_files/"`)
# * How to load the pickled data
#     > `with open("path", "rb") as f:`  <br>
#     > `data=pickle.load(f)`
# * Output: Histogram of the numbers of gene per cell that contains/ don't contain 15th state in the all cell types that is grouped by bin (Well, I don't know why I wrote this code..)
# <br><br>
# 
# <img src="./desc_img/qnonq_hist2.png" width="400" height="150">

# In[54]:


# draw a histogram type2 (group by bin)
def QnonQforCellHistT2(q_cnt_lst, not_q_cnt_lst,bin_size):
    """Run this after executing QnonQforCell"""
    data_w=q_cnt_lst
    data_wo=not_q_cnt_lst

    mu_w, std_w=norm.fit(data_w)
    mu_wo, std_wo=norm.fit(data_wo)

    fig=plt.figure(figsize=(8,4))
    ax=fig.add_subplot(1,1,1)
    ax.hist([data_w,data_wo], bins=bin_size, alpha=0.5, color=["teal","orange"], label=["with Quiescent state","without Quiescent state"])
    
    ax.legend(loc="upper left")

    title='Number of Genic region with/without Quiescent state'
    plt.title(title)
    plt.legend()
    plt.xlabel("No. of Genes")
    plt.ylabel("Counts")
    plt.show()


# #### Fuction: `QnonQforCellSwarmp`
# * Create a dataframe of two lists (below) and draw a dual swarmp graph in a single figure.
# * Input: `q_cnt_lst` and `not_q_cnt_lst` (find them pickled at `"../database/temp_files/"`)
# * Output: `q_cnt_data` (dataframe of the two lists) and the graph
# 
# <img src="./desc_img/qnonq_swarmp.png" width="400" height="150">

# In[55]:


def QnonQforCellSwarmp(q_cnt_lst, not_q_cnt_lst):
    q_cnt_data=pd.DataFrame({"q_cnt":q_cnt_lst, "not_q_cnt":not_q_cnt_lst}) # create a dataframe
    fig=plt.figure(figsize=(6,4))
    sns.swarmplot(data=q_cnt_data, palette="bone")
    plt.grid(b=None)
    plt.ylabel("Counts", fontsize=12)
    plt.show()
    return q_cnt_data


# #### Function: `cntQinGene`
# * This function generates three lists
#     1. 15th state-including gene count
#     2. 15th state-including gene length
#     3. Proportion of 15th state in the 15th state-including gene
# * Input: `css_gene_lst_all` (pickled at `"../database/temp_files/"`, note that it's 2.8Gb)
# * Output: 
#     1. `cnt_o_lst`: 15th state-including gene count
#     2. `gene_len_lst`: 15th state-including gene length
#     3. `pro_o_lst`: Proportion of 15th state in the 15th state-including gene

# In[3]:


# generate three lists: 15th state-including gene count, gene length, proportion of 15th state per gene
def cntQinGene(css_gene_lst_all):
    """run this after executing compGene2css(whole_gene_file,df)
       [Input]
       css_gene_lst_all : list of css list of each chromosome
       [Output]
       cnt_o_lst : list of Quiescent state counts list per chromosome
       gene_len_lst : list of gene length (in terms of chromatin state Anno.200bps) list per chromosome
       pro_o_lst : list of proportion of Quiescent state per gene list per chromosome
    """
    cnt_o_lst=[]
    gene_len_lst=[]
    pro_o_lst=[]
    for i in range(len(css_gene_lst_all)):
        css_gene_lst=css_gene_lst_all[i]
        
        cnt_o_chr=[]
        gene_len_chr=[]
        pro_o_chr=[]
        for j in range(len(css_gene_lst)):
            css_gene=css_gene_lst[j]
            cnt_o=css_gene.count("O")
            gene_len=len(css_gene)
            pro_o=cnt_o/gene_len
            
            cnt_o_chr.append(cnt_o)
            gene_len_chr.append(gene_len)
            pro_o_chr.append(pro_o)
            
        cnt_o_lst.append(cnt_o_chr)
        gene_len_lst.append(gene_len_chr)
        pro_o_lst.append(pro_o_chr)
        
    return cnt_o_lst, gene_len_lst, pro_o_lst


# #### Function: `cntQinGeneVis1`
# * For visualization of the result of the function `cntQinGene`.
# * Input: `cnt_o_lst`, `gene_len_lst`, and `pro_o_lst` (result of `cntQinGene`), for more info, see that function.
# * Output: Letter-value histogram of `cnt_o_lst` and `gene_len_lst`, and violin plot for `pro_o_lst`
# <img src="./desc_img/cntQinGeneVis1.png" width="500" height="150">

# In[4]:


def cntQinGeneVis1(cnt_o_lst, gene_len_lst, pro_o_lst):
    """
    Input: cnt_o_lst, gene_len_lst, pro_o_lst (the result lists of cntQinGene(css_gene_lst_all). To load the css_gene_lst_all, find the file at "../database/temp_files/")
    Output: Dataframe of those 3 lists, and the visualization of the above lists using histogram
    """
    def flatLst(lst):
        flatten_lst=[elm for sublst in lst for elm in sublst]
        return flatten_lst

    cnt_o_lst_flat=flatLst(cnt_o_lst)
    gene_len_lst_flat=flatLst(gene_len_lst)
    pro_o_lst_flat=flatLst(pro_o_lst)
    
    three_df=pd.DataFrame({"gene_len":gene_len_lst_flat, "cnt_o":cnt_o_lst_flat, "o_proportion":pro_o_lst_flat})
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5), sharey=False)
    ax1=sns.boxenplot(data=three_df[["gene_len","cnt_o"]],palette="viridis", width=0.6, linewidth=0.01, scale="linear", ax=ax1)
    ax1.set_ylabel("Count", fontsize=15)
    
    ax2=sns.violinplot(data=three_df[["o_proportion"]], linewidth=0.6, inner="box", width=0.6, color="lightgray", ax=ax2)  
    ax2.set_ylabel("Proportion", fontsize=15)
    ax2.grid()
    plt.grid(b=None)
    
    return three_df   


# # 4. CSS Pattern analysis
# **[back to index](#Index)**

# ## 4-1. For 15th-including data
# 
# * Target data: CSS dataset with 15th state included
# * Starting data is acquired from `all_unit_css=df2unitcss(df)` [Jump](#Unit-length-css)
# * `all_unit_css` is a list, the element of which is chromosome-wise all-connected **unit-length** (per 200 bp) CSS
# > `len(all_unit_css)` = 24 <br>
# > `len(all_unit_css[0])` =1246253
# <!-- * Start from the process [3-2. Cut the telomere region on CSS and save the file](#3-2.-Cut-the-telomere-region-on-CSS-and-save-the-file) -->

# In[58]:


## but it must be a distribution where 15th states covers almost of the entire area. 
## So I stopped here, because basic statistics are known from 4-2. For 15th-less data


# In[ ]:





# In[ ]:





# ## 4-2. For 15th-less data
# 
# Now the dataframe has been transformed into a list of string all connected css, chromosome-wise.<br>
# The variable of the above list is now called chr_css_list.<br>
# Following functions will analyze the statistics of the each strings.

# In[59]:


def css_list2count(df, chr_css_list):
    
    """Input: chr_css_list acquired by df2css_chr_str(df), 
    which is a list of string all connected css, chromosome-wise.
    Output: a dataframe (col: chromosome, row:letter)"""
    
    state_alphabets=list(state_dict.values())
    chr_names=list(df["chromosome"].unique())
    count_all=pd.DataFrame(columns=chr_names, index=state_alphabets)  # create an empty dataframe 
    
    for num, _ in enumerate(chr_css_list):   # for each chromosome..
        chr_css=chr_css_list[num]
        chr_name=chr_names[num]

        for letter in state_alphabets:   # count the number of A, B, C, D ... in the string
            count_all.loc[letter][chr_name]=chr_css.count(letter)
    
    return count_all


# In[60]:


def draw_count_barplot_incl15(count_all, chr_no):
    
    """ Draw a bar plot (chromatin state vs. count) per chromosome
    input(1) table of 'count_all' which is created by the function css_list2count(df, chr_css_list) 
    input(2) chromosome name in string, e.g.) 'chr1', 'chr2', ... 
    output: bar plot of the all chromatin state count (including 15th state)"""

    count_all_renamed=count_all.rename(index=css_dict)
    color_dec=colors2color_dec(css_color_dict)
    count_all_renamed.loc[:,chr_no].plot.bar(rot=45, color=color_dec)
    ax0=ax0.set_ylabel("Counts", fontsize=14)


# In[61]:


def draw_count_barplot_wo15(count_all, chr_no):
    
    """ Draw a bar plot (chromatin state vs. count) per chromosome
    input(1) table of 'count_all' which is created by the function css_list2count(df, chr_css_list) 
    input(2) chromosome name in string, e.g.) 'chr1', 'chr2', ... 
    output: bar plot of the all chromatin state count except for 15th state"""

    count_all_renamed=count_all.rename(index=css_dict)
    color_dec=colors2color_dec(css_color_dict)
    ax0=count_all_renamed.loc[:,chr_no][:-1].plot.bar(rot=45, color=color_dec)
    ax0.set_ylabel("Counts", fontsize=14)  


# In[62]:


def colored_css_str(sub_str):
    col_str=""
    for letter in sub_str:
        for state in list(state_col_255_dict.keys()):
            if letter==state:
                r=state_col_255_dict[letter][0]
                g=state_col_255_dict[letter][1]
                b=state_col_255_dict[letter][2]
                col_letter="\033[38;2;{};{};{}m{}\033[38;2;255;255;255m".format(r,g,b,letter)
                col_str+=col_letter
    return print("\033[1m"+col_str+"\033[0;0m") 


# **Frequently used function!** <br>
# To convert any string into colored string according to the color palette for CSS.

# In[63]:


def colored_css_str_as_is(sub_str):   # convert space into space
    col_str=""
    for letter in sub_str:
        if letter==" ":
            col_str+=" "
        else:                
            for state in list(state_col_255_dict.keys()):
                if letter==state:
                    r=state_col_255_dict[letter][0]
                    g=state_col_255_dict[letter][1]
                    b=state_col_255_dict[letter][2]
                    col_letter="\033[38;2;{};{};{}m{}\033[38;2;255;255;255m".format(r,g,b,letter)
                    col_str+=col_letter
    return print("\033[1m"+col_str+"\033[0;0m") 


# #### css pattern analysis without 15th state (state **O**)
# 
# 1. create a list of a css without 15th state, the element of which is connected (df2inbetweeen_lst)
# 2. create a whole list of css without 15th state, using a all-chromosome df (df2wo15list)
# 3. calculate the length of each element of the generated list, and analyze the statistics

# In[64]:


def df2inbetweeen_lst(df):
    lst=[]
    df_wo_o=df[df["state"]!=15]   #remove the 15th state from the css
    css_df=df_wo_o["state_seq_full"]
    str_elm=css_df.iloc[0]  # the very first elm
    for i in range(1, len(css_df)):
        # check the index first
        cid=css_df.index[i] #init=1
        pid=css_df.index[i-1] # init=0
        ssf=css_df
        if (cid-pid)!=1: # if the index is separated (not a succeeding numbers)
            lst.append(str_elm)
            str_elm=ssf.iloc[i]
        else:            # if encountered a consecutive index
            str_elm+=ssf.iloc[i] # attach the next str to the previous str
            if i==len(css_df)-1:   # treat the final line
                lst.append(str_elm)
    return lst


# In[65]:


def df2wo15list(df):
    total_lst=[]
    df_chr_list=df2chr_df(df)   # a list, elm of which is the df of each chromosome
    for df_chr in df_chr_list:   # for each chromosome, create a grand list by adding up the whole
        lst_chr=df2inbetweeen_lst(df_chr)
        total_lst+=lst_chr
    return total_lst   # total_lst here consists of the connected-patterns betweeen 15th state


# In[66]:


def css_elm_stat(total_lst):# graph of the length distribution 
    len_lst=[]              # total_lst here consists of the connected-patterns betweeen 15th state
    for elm in total_lst:
        assert type(elm)==str, "element type is not string"
        len_lst.append(len(elm))
    print("total count: ", len(total_lst))
    print("max length: ", max(len_lst))
    print("min length: ", min(len_lst))
    print("average length: ",np.mean(len_lst))
    fig =plt.figure(figsize=(6,4))
    plt.hist(len_lst, bins=20, log=True, color="teal", edgecolor="white")
    plt.xlabel("length of chromatin state pattern", fontsize=14)
    plt.ylabel("Count", fontsize=14)


# In[67]:


def lst2let_compose(total_lst):# graph of the number of letter composed for a pattern
    letter_cnt=[]              # total_lst here consists of the connected-patterns betweeen 15th state
    for word in total_lst:
        chk_let=word[0]
        num_let=1
        for let in word:
            if let!=chk_let:
                num_let+=1
                chk_let=let
        letter_cnt.append(num_let)
    print("total count: ", len(letter_cnt))
    print("max composition: ", max(letter_cnt))
    print("min composition: ", min(letter_cnt))
    print("average composition: ", np.mean(letter_cnt))
    fig =plt.figure(figsize=(6,4))
    plt.hist(letter_cnt, bins=20, log=True, color="orange", edgecolor="white")
    plt.xlabel("number of state in a composition", fontsize=14)
    plt.ylabel("Count", fontsize=14)


# In[68]:


def custom_colorlist(data_dict):
    
    """ 
    INPUT: solo chromatin state data in dict such as 
           data_dict={'I': 114, 'A': 23, 'N': 119, 'G': 33, 'E': 131, 'H': 1}
    OUTPUT: customized colormap according to ROADMAP (type=list)
    """
    state_list=list(data_dict.keys())
    colormap_list=[]
    assert type(state_list[0])==str
    for state in state_list:
        if css_dict[state] in css_name_col_dict.keys():
            color_rgb=css_name_col_dict[css_dict[state]]
            colormap_list.append(color_rgb)
    return colormap_list


# In[69]:


def lst2solo_compose(total_lst):# graph of a solo pattern frequency
    
    """INPUT: the entire list of in-between pattern w.o. 15th state (total_lst)
       OUTPUT: the most/least frequent solo pattern and the frequency graph
    """
    
    letter_cnt=[]
    for word in total_lst:
        chk_let=word[0]
        num_let=1
        for let in word:
            if let!=chk_let:
                num_let+=1
                chk_let=let
        letter_cnt.append(num_let)
    css_lst_dict=dict(zip(total_lst, letter_cnt))
    
    lst_for_solo=[]                   # prepare to make a solo pattern list
    for pattern, num in list(css_lst_dict.items()): # as a tuple element (key, val)
        if num==1:
            lst_for_solo.append(pattern[0])
    solo_counter=collections.Counter(lst_for_solo)
    solo_data_dict=dict(solo_counter) # ditionary of solo pattern and the frequency
    solo_data_dict=dict(sorted(solo_data_dict.items(), reverse=True, key=lambda item: item[1]))
    my_color=custom_colorlist(solo_data_dict)  # create a customized colormap using solo data
    
    for pattern, num in solo_data_dict.items():
        if num is max(solo_data_dict.values()):
            max_state=pattern
            max_num=num
        elif num is min(solo_data_dict.values()):
            min_state=pattern
            min_num=num

    print("frequency of solo pattern: ", len(lst_for_solo))
    print("the most frequent solo pattern: ", css_dict[max_state], " for ", max_num, " times appeared." )
    print("the least frequent solo pattern: ", css_dict[min_state], " for ", min_num, " times appeared." )
    
    x=[css_dict[state] for state in solo_data_dict.keys()]
    y=solo_data_dict.values()
    
    fig =plt.figure(figsize=(6,4))
    plt.bar(x,y, color=my_color)
    plt.xlabel("solo pattern", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)


# #### make a kmer and save as a sample

# In[70]:


def total_lst2kmer(total_lst,k):
    total_kmer_lst=[]
    for elm in total_lst:
        elm2kmer=seq2kmer(elm, k)
        if len(elm2kmer) >0:   # remove the short pattern... will be fine?
            total_kmer_lst.append(elm2kmer)
    return total_kmer_lst


# In[71]:


# total_kmer_lst=total_lst2kmer(total_lst,6)


# In[72]:


# file_name02="../database/test_data/6_tr01.txt"
# with open(file_name02,"w") as g:
#     g.write("\n".join(total_kmer_lst))
# g.close()


# In[ ]:





# In[ ]:





# # 5. Training result analysis
# **[back to index](#Index)**

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




