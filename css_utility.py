#!/usr/bin/env python
# coding: utf-8

# # Utilities
# Various functions to process the initial data

# In[37]:


# ### To convert the file into .py
# !jupyter nbconvert --to script css_utility.ipynb


# In[56]:


import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from motif_utils import seq2kmer
from motif_utils import kmer2seq
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import random
import collections
import operator
import itertools
import pickle
import seaborn as sns
from tqdm import tqdm
from tqdm.notebook import tqdm
from tqdm.notebook import tqdm_notebook
import glob
from wordcloud import WordCloud
import stylecloud
from collections import Counter


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
# * **[3. Cutting the chromatin state (Dataset Preparation)](#3.-Cutting-the-chromatin-state-(Dataset-Preparation))**
#     * [3-1. Quiescent state distribution](#3-1.-Quiescent-state-distribution)
#     * [3-2. Cut the telomere region on CSS and save the file](#3-2.-Cut-the-telomere-region-on-CSS-and-save-the-file) <font color="royalblue">-> **pretrain data are saved**</font>
#     * [3-3. Cut the chromatin states : genic/non-genic area](#3-3.-Cut-the-chromatin-states-:-genic-or-non-genic-area)
#         * [3-3-1. Genic area](#3-3-1.-Genic-area)
#         * [3-3-2. Non-genic area (intergenic region)](#3-3-2.-Non-genic-area-(intergenic-region))
#         * [3-3-3. Genic or Non-genic raw-length CSS to unit-length CSS](#3-3-3.-Genic-or-Non-genic-raw-length-CSS-to-unit-length-CSS)
#             * [3-3-3-0. Small code modifications](#3-3-3-0.-Small-code-modifications)
#             * [3-3-3-1. CSS for 57 Epigenomes Genic regions are saved.](#3-3-3-1.-CSS-for-57-Epigenomes-Genic-regions-are-saved.)
#         * [3-3-4. Cut the unit-length css into trainable size and kmerize it](#3-3-4.-Cut-the-unit-length-css-into-trainable-size-and-kmerize-it) <font color="royalblue">-> **pretrain data are saved**</font>
#         * [3-3-5. Fine-tuning data: Dataframe version](#3-3-5.-Fine-tuning-data:-Dataframe-version)
#         * [3-3-6. Fine-tuning data: save files as .tsv](#3-3-6.-Fine-tuning-data:-save-files-as-.tsv) <font color="orange"> -> **fine-tuning data are saved** </font>
#     * [3-4. Count the number of 15th states in genic and non-genic region](#3-4.-Count-the-number-of-15th-states-in-genic-and-non-genic-region)         
#     * [3-5. Complexity of CSS in genic area](#3-5.-Complexity-of-CSS-in-genic-area)
#         * [3-5-1. Create a matrix to show the statistics](#3-5-1.-Create-a-matrix-to-show-the-statistics)
#         * [3-5-2. Extract the complex and less complex css on gene](#3-5-2.-Extract-the-complex-and-less-complex-css-on-gene)
#             * [3-5-2-1. CSS for 57 Epigenomes Complex and Less Complex Genic regions are saved.](#3-5-2-1.-CSS-for-57-Epigenomes-Complex-and-Less-Complex-Genic-regions-are-saved.)
#         * [3-5-3. Cut into Kmer and save](#3-5-3.-Cut-into-Kmer-and-save) <font color="royalblue">-> **pretrain data are saved**</font>
#         * [3-5-4. Show the composition for each case](#3-5-4.-Show-the-composition-for-each-case)
#         * [3-5-5. Prepare and save Fine-tuning for Complex gene CSS and others](#3-5-5.-Prepare-and-save-Fine-tuning-for-Complex-gene-CSS-and-others) <font color="orange"> -> **fine-tuning data are saved**</font>
#     * [3-6. Gene expression classification](#3-6.-Gene-expression-classification)
#         * [3-6-1. Gene expression file into the list of dataframe](#3-6-1.-Gene-expression-file-into-the-list-of-dataframe)
#         * [3-6-2. Matching to CSS](#3-6-2.-Matching-to-CSS)
#             * [3-6-2-1. CSS for various gene expression cases are saved.](#3-6-2-1.-CSS-for-various-gene-expression-cases-are-saved.)
#         * [3-6-3. Cut into Kmer and save](#3-6-3.-Cut-into-Kmer-and-save)<font color="royalblue">-> **pretrain data are saved**</font>
#         * [3-6-4. Fine-tuning data](#3-6-4.-Fine-tuning-data) <font color="orange"> -> **fine-tuning data are saved** </font>
#     * [3-7. Promoter classification](#3-7.-Promoter-classification)
#     * [3-8. Enhancer classification](#3-8.-Enhancer-classification)
# * **[4. CSS Pattern analysis](#4.-CSS-Pattern-analysis)**
# * **[5. Training result analysis](#5.-Training-result-analysis)**
#     * [5-1. Evaluation](#5-1.-Evaluation)
#         * [5-1-2. Pretrain evaluation](#5-1-2.-Pretrain-evaluation)
#         * [5-1-3. Fine tuning evaluation](#5-1-3.-Fine-tuning-evaluation)
#     * [5-2. Motif](#5-2.-Motif)

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


# create a pickle for a cell-wise dataframe (should be modified to correct the cell ID)
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

pickle_path='../database/cell_pickle/'
pickle_files=os.listdir(pickle_path)
            
all_files=file_list_maker(path, bed_files)
all_cell_pickles=file_list_maker(pickle_path, pickle_files)


# In[11]:


all_files[0]


# In[12]:


all_cell_pickles[0]


# #### Updated the pickled df to match the cell ID
# * The following function has been conducted and no need to run
# * Output path is `../database/roadmap/df_pickled/`

# In[5]:


path_unzipped="../database/bed/unzipped/" ## unzipped bed file (chromatin state annotation file for ROADMAP)
unzipped_epi=sorted(os.listdir(path_unzipped))
unzipped_epi_files=[os.path.join(path_unzipped,file) for file in unzipped_epi]

def unzipped_to_df(unzipped_epi_files, output_path="../database/roadmap/df_pickled/"):
    for file in unzipped_epi_files:
        cell_id=file.split("/")[-1][:4]
        output_name=output_path+cell_id+"_df_pickled.pkl"
        df=bed2df_expanded(file)
        df.to_pickle(output_name)
    return print("done!")
# unzipped_to_df(unzipped_epi_files, output_path="../database/roadmap/df_pickled/")


# * The following function has been conducted and no need to run
# * Input path: (df_pickled_path=) `../database/roadmap/df_pickled/`
# * Output path: `../database/roadmap/css_pickled/`

# In[19]:


def pickled_df2unit_css(df_pickled_path, output_path="../database/roadmap/css_unit_pickled/",verbose=True):
    
    def load_pickled_df(df_pickled_file):
        with open(df_pickled_file, "rb") as f:
            df = pickle.load(f)
        unit_css = df2unitcss(df)
        return unit_css   
        df_pickled_files = [os.path.join(df_pickled_path, df) for df in sorted(os.listdir(df_pickled_path))]      
    
    for file in df_pickled_files:
        cell_id = file.split("/")[-1][:4]
        output_name = output_path + cell_id + "_css_pickled.pkl"           
        unit_css=load_pickled_df(file)
        with open(output_name, 'wb') as g:
            pickle.dump(unit_css, g)          
        if verbose:
            print(cell_id+" is done")

    return print("All done!")
# pickled_df2unit_css(df_pickled_path,output_path="../database/roadmap/css_pickled/")


# In[ ]:





# ## 2-2. Prerequisite dictionaries

# In[10]:


state_dict={1:"A", 2:"B", 3:"C", 4:"D", 5:"E",6:"F",7:"G",8:"H" ,
                9:"I" ,10:"J",11:"K", 12:"L", 13:"M", 14:"N", 15:"O"}


# In[11]:


css_name=['TssA','TssAFlnk','TxFlnk','Tx','TxWk','EnhG','Enh','ZNF/Rpts',
          'Het','TssBiv','BivFlnk','EnhBiv','ReprPC','ReprPcWk','Quies']


# In[12]:


css_dict=dict(zip(list(state_dict.values()), css_name))  # css_dict={"A":"TssA", "B":"TssAFlnk", ... }


# In[13]:


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

# In[14]:


def colors2color_dec(css_color_dict):
    colors=list(css_color_dict.values())
    color_dec_list=[]
    for color in colors:
        color_dec=tuple(rgb_elm/255 for rgb_elm in color)
        color_dec_list.append(color_dec)        
    return color_dec_list


# **scale 0 to 1**

# In[15]:


state_col_dict=dict(zip(list(state_dict.values()),colors2color_dec(css_color_dict)))


# **scale 0 to 255**

# In[16]:


state_col_255_dict=dict(zip(list(state_dict.values()),list(css_color_dict.values())))


# **hexacode**

# In[17]:


hexa_state_col_dict={letter: "#{:02x}{:02x}{:02x}".format(*rgb) for letter,rgb in state_col_255_dict.items()}


# **name instead of alphabets**

# In[18]:


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

# In[33]:


# make a long string of the css (not using unit, but the real length)

# def df2longcss(df):
#     df_lst_chr=df2chr_df(df)
#     # remove the microchondria DNA from df_lst_chr
#     if df_lst_chr[-3]["chromosome"].iloc[0]=="chrM":
#         del df_lst_chr[-3]
#         assert df_lst_chr[-3]["chromosome"].iloc[0]=="chr22"
#     else:   
#         assert df_lst_chr[-3]["chromosome"].iloc[0]=="chr22"
    
#     all_css=[]
#     for i in range(len(df_lst_chr)):
#         df_chr=df_lst_chr[i]
#         css_chr=''
#         for j in range(len(df_chr)):
#             css_chr+=df_chr["length"].iloc[j]*df_chr["state_seq"].iloc[j]
#         all_css.append(css_chr)  
#     return all_css


# In[53]:


# make a long string of the css (not using unit, but the real length)
# modified 4.July 2023, to support the case where ChromosomeM is not at -3, but -2

def df2longcss(df):
    df_lst_chr=df2chr_df(df)
    # remove the microchondria DNA from df_lst_chr
    if df_lst_chr[-3]["chromosome"].iloc[0]=="chrM":
        del df_lst_chr[-3]
#         assert df_lst_chr[-3]["chromosome"].iloc[0]=="chr22"
    elif df_lst_chr[-2]["chromosome"].iloc[0]=="chrM":
        del df_lst_chr[-2]
#         assert df_lst_chr[-3]["chromosome"].iloc[0]=="chr22"
    
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

# In[58]:


# make a long string of the css (unit length, not the real length)

def df2unitcss(df):
    df_lst_chr=df2chr_df(df)
    # remove the microchondria DNA from df_lst_chr
    if df_lst_chr[-3]["chromosome"].iloc[0]=="chrM":
        del df_lst_chr[-3]
#         assert df_lst_chr[-3]["chromosome"].iloc[0]=="chr22"
#     else:   
#         assert df_lst_chr[-3]["chromosome"].iloc[0]=="chr22"
    
    all_unit_css=[]
    for i in range(len(df_lst_chr)):
        df_chr=df_lst_chr[i]
        css_chr=''
        for j in range(len(df_chr)):
            css_chr+=df_chr["unit"].iloc[j]*df_chr["state_seq"].iloc[j]
        all_unit_css.append(css_chr)  
    return all_unit_css


# #### These are new functions (corrected)

# In[43]:


def shorten_string(s, factor):
    # This regular expression matches groups of the same character.
    pattern = re.compile(r'(.)\1*')

    # This function will be used to replace each match.
    def replacer(match):
        # The group that was matched.
        group = match.group()

        # Calculate the new length, rounding as necessary.
        new_length = round(len(group) / factor)

        # Return the character repeated the new number of times.
        return group[0] * new_length

    # Use re.sub to replace each match in the string.
    return pattern.sub(replacer, s)


# In[44]:


def Convert2unitCSS_main_new(css_lst_all, unit=200):# should be either css_gene_lst_all or css_Ngene_lst_all
    """
    Input: css_gene_lst_all or css_Ngene_lst_all, the list of chromosome-wise list of the css in genic, intergenic regions.
    Output: css_gene_unit_lst_all or css_Ngene_unit_lst_all
    """
    reduced_all=[]
    for i in range(len(css_lst_all)):
        reduced_chr=[]
        for j in range(len(css_lst_all[i])):
            reduced=shorten_string(css_lst_all[i][j], unit)
            reduced_chr.append(reduced)
        reduced_all.append(reduced_chr)
    return reduced_all


# In[ ]:





# In[ ]:





# In[ ]:





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


# # 3. Cutting the chromatin state (Dataset Preparation)
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

# In[41]:


# def shorten_string(s, factor):
#     # This regular expression matches groups of the same character.
#     pattern = re.compile(r'(.)\1*')

#     # This function will be used to replace each match.
#     def replacer(match):
#         # The group that was matched.
#         group = match.group()

#         # Calculate the new length, rounding as necessary.
#         new_length = round(len(group) / factor)

#         # Return the character repeated the new number of times.
#         return group[0] * new_length

#     # Use re.sub to replace each match in the string.
#     return pattern.sub(replacer, s)


# In[42]:


# def Convert2unitCSS_main_new(css_lst_all, unit=200):# should be either css_gene_lst_all or css_Ngene_lst_all
#     """
#     Input: css_gene_lst_all or css_Ngene_lst_all, the list of chromosome-wise list of the css in genic, intergenic regions.
#     Output: css_gene_unit_lst_all or css_Ngene_unit_lst_all
#     """
#     reduced_all=[]
#     for i in range(len(css_lst_all)):
#         reduced_chr=[]
#         for j in range(len(css_lst_all[i])):
#             reduced=shorten_string(css_lst_all[i][j], unit)
#             reduced_chr.append(reduced)
#         reduced_all.append(reduced_chr)
#     return reduced_all


# In[38]:


# def compGene2css(whole_gene_file,df):   # note that the result is also overlapped css... >>rewrite it with gene_removeDupl!
#     """
#     Input: Reference gene file, df (CSS)
#     Output: list of chromosome-wise list that contains the css at genic area only.
#     """
#     g_lst_chr=whGene2GLChr(whole_gene_file) # list of gene table df per chromosome
#     css_lst_chr=df2longcss(df) # list of long css per chromosome
#     total_chr=len(g_lst_chr)
    
#     css_gene_lst_all=[]
#     for i in tqdm_notebook(range(total_chr)):
#         css=css_lst_chr[i]   # long css of i-th chromosome
#         gene_df=g_lst_chr[i] # gene df of i-th chromosome
        
#         css_gene_lst_chr=[]
#         for j in range(len(gene_df)):
#             g_start=gene_df["TxStart"].iloc[j]-1  # python counts form 0
#             g_end=gene_df["TxEnd"].iloc[j]+1      # python excludes the end
            
#             css_gene=css[g_start:g_end]           # cut the gene area only
#             css_gene_lst_chr.append(css_gene)     # store in the list
          
#         css_gene_lst_all.append(css_gene_lst_chr)  # list of list
    
#     assert len(css_gene_lst_all)==total_chr
#     return css_gene_lst_all


# In[7]:


def compGene2css(whole_gene_file,df):   # fixed June. 29. 2023
    """
    Input: Reference gene file, df (CSS)
    Output: list of chromosome-wise list that contains the css at genic area only.
    """
#     g_lst_chr=whGene2GLChr(whole_gene_file) # list of gene table df per chromosome
    
    ########### new fancy gene table without overlap ###########
#     g_lst_chr=gene_removeDupl(whole_gene_file) #### fixed June. 29. 2023
    g_df_chr_lst=whGene2GLChr(whole_gene_file) #### fixed June. 29. 2023
    g_lst_chr=merge_intervals(g_df_chr_lst) #### fixed June. 29. 2023
    ############################################################
    
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


# In[ ]:





# #### Function: `pickled_df2gene_unit_css`
# 
# * This function is already executed and no need to rerun.
# * This function saves the all genic region in ROADMAP data in unit css 
# * Input: `df_pickled_path="../database/roadmap/df_pickled/"`, `output_path="../database/roadmap/"`
# * The output is pickled under the output path

# In[25]:


def pickled_df2gene_unit_css(df_pickled_path="../database/roadmap/df_pickled/", output_path="../database/roadmap/", verbose=True):
    """
    Save unit CSS into Genic, for the entire 127 epigenomes
    """
    df_pickled_files = [os.path.join(df_pickled_path, df) for df in sorted(os.listdir(df_pickled_path))]
    
    def load_pickled_df(df_pickled_file):
        with open(df_pickled_file, "rb") as f:
            df = pickle.load(f)
        return df
    
    for file in df_pickled_files:
        cell_id = file.split("/")[-1][:4]          

        gene_output_name = output_path +"gene_css_unit_pickled/"+ cell_id + "_gene_css_pickled.pkl"
        df=load_pickled_df(file)

        css_gene_lst_all=compGene2css(whole_gene_file,df)
        css_gene_unit_lst_all=Convert2unitCSS_main(css_gene_lst_all, unit=200)

        with open(gene_output_name, 'wb') as g:
            pickle.dump(css_gene_unit_lst_all, g)

        if verbose:
            print(cell_id+" is done")

    return print("All done!")


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


# In[ ]:





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


# In[ ]:





# In[ ]:


#### Merging the gene table #### modified June. 29. 2023

def merge_intervals(df_list):
    merged_list = []  # List to hold merged DataFrames

    for df in df_list:
        # Sort by 'TxStart'
        df = df.sort_values(by='TxStart')

        # Initialize an empty list to store the merged intervals
        merged = []

        # Iterate through the rows in the DataFrame
        for _, row in df.iterrows():
            # If the list of merged intervals is empty, or the current interval does not overlap with the previous one,
            # append it to the list
            if not merged or merged[-1]['TxEnd'] < row['TxStart']:
                merged.append({'TxStart': row['TxStart'], 'TxEnd': row['TxEnd']})  # Only keep 'TxStart' and 'TxEnd'
            else:
                # Otherwise, there is an overlap, so we merge the current and previous intervals
                merged[-1]['TxEnd'] = max(merged[-1]['TxEnd'], row['TxEnd'])

        # Convert the merged intervals back into a DataFrame and append it to the list
        merged_list.append(pd.DataFrame(merged))

    return merged_list  # a list of DF, containing only TxStart and TxEnd


# In[ ]:





# #### Function: `compNonGene2css`
# * This function extracts the css on the non-genic (intergenic) area of the genome.
# * The function `gene_removeDupl` was used here, for extracting the non-genic region index.
# * Input: `whole_gene_file` and `df` (from the css bed file)
# * Output: `css_Ngene_lst_all` The CSS on the non-genic region

# In[20]:


##### fixed June 29. 2023
def compNonGene2css(whole_gene_file,df): 
    """
    Input: Reference gene file, df (CSS)
    Output: list of chromosome-wise list that contains the css at "non-genic" area only.
    """
    
    print("Extracting the CSS on the intergenic region ...")

    ########### new fancy gene table without overlap ###########
#     new_gene_lst_all=gene_removeDupl(whole_gene_file) ##### fixed June 29. 2023
    g_df_chr_lst=whGene2GLChr(whole_gene_file)  ##### fixed June 29. 2023
    new_gene_lst_all=merge_intervals(g_df_chr_lst) ##### fixed June 29. 2023
    ############################################################
    
    css_lst_chr=df2longcss(df) # list of long css per chromosome
    total_chr=len(new_gene_lst_all)
    
    css_Ngene_lst_all=[]
        
    for i in tqdm_notebook(range(total_chr)):
        css=css_lst_chr[i]   # long css of i-th chromosome
        gene_df=new_gene_lst_all[i] # gene df of i-th chromosome
        
#         assert gene_df["TxStart"].iloc[0]>=1, "Gene starts from the very first location at {}-th chromosome.".format(i)
#         assert gene_df["TxEnd"].iloc[-1]<=len(css), "Gene ends at the very last location at {}-th chromosome.".format(i)  
        ### asertion was removed because it produces an error when trying to apply to cells without Y chr.        
    
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


################## this is obsolete, use Convert2unitCSS_main_new instead

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


# def Convert2unitCSS_main(css_lst_all, unit=200): # should be either css_gene_lst_all or css_Ngene_lst_all
#     """
#     Input: css_gene_lst_all or css_Ngene_lst_all, the list of chromosome-wise list of the css in genic, intergenic regions.
#     Output: css_gene_unit_lst_all or css_Ngene_unit_lst_all
#     """
#     print("Converting css from the raw length into unit-length ... ")
#     css_unit_lst_all=[]
#     for chr_no in tqdm_notebook(range(len(css_lst_all))):
#         css_chr_lst=css_lst_all[chr_no]
#         css_chr_unit_lst=[]
#         let_str_lst_all, unit_cnt_lst_all=long2unitCSS(css_chr_lst, unit=unit)
#         unit_css_lst=['']*len(let_str_lst_all)
#         for i, let_str in enumerate(let_str_lst_all):
#             for j in range(len(let_str)-1):
#                 unit_css_lst[i]+=let_str[j]*unit_cnt_lst_all[i][j] # only unit will be multiplied!
#         unit_css_lst=[css for css in unit_css_lst if css!='']  # remove the empty element
#         css_unit_lst_all.append(unit_css_lst)
#     print("Done!")
#     return css_unit_lst_all


# In[48]:


################## this is obsolete, use Convert2unitCSS_main_new instead

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
#             for j in range(len(let_str)-1):   ###fixed 29. June. 2023, not yet good to go
            for j in range(len(let_str)): ###fixed 29. June. 2023
                unit_css_lst[i]+=let_str[j]*unit_cnt_lst_all[i][j] # only unit will be multiplied!
        unit_css_lst=[css for css in unit_css_lst if css!='']  # remove the empty element
        css_unit_lst_all.append(unit_css_lst)
    print("Done!")
    return css_unit_lst_all


# Now following files are saved at : `../database/temp_files/` 
# * `css_gene_unit_lst_all` : The unit-length css on the genic area
# * `css_Ngene_unit_lst_all`: The unit-length css on the intergenic area

# In[ ]:





# ### 3-3-3-0. Small code modifications

# ### Genic anc Intergenic re-saved by following functions

# The series of following functions were written to manage the discrepancies in chromosome numbers created by th e previous function, which is caused by some cells without Y chromosme (female cells). All functions to save the genic css and intergenic css were conducted and stored, thus no need to redo (so far).

# In[38]:


def shorten_string(s, factor):
    # This regular expression matches groups of the same character.
    pattern = re.compile(r'(.)\1*')

    # This function will be used to replace each match.
    def replacer(match):
        # The group that was matched.
        group = match.group()

        # Calculate the new length, rounding as necessary.
        new_length = round(len(group) / factor)

        # Return the character repeated the new number of times.
        return group[0] * new_length

    # Use re.sub to replace each match in the string.
    return pattern.sub(replacer, s)


# In[40]:


def Convert2unitCSS_main_new(css_lst_all, unit=200):# should be either css_gene_lst_all or css_Ngene_lst_all
    """
    Input: css_gene_lst_all or css_Ngene_lst_all, the list of chromosome-wise list of the css in genic, intergenic regions.
    Output: css_gene_unit_lst_all or css_Ngene_unit_lst_all
    """
    reduced_all=[]
    for i in range(len(css_Ngene_lst_all)):
        reduced_chr=[]
        for j in range(len(css_Ngene_lst_all[i])):
            reduced=shorten_string(css_Ngene_lst_all[i][j], unit)
            reduced_chr.append(reduced)
        reduced_all.append(reduced_chr)
    return css_unit_lst_all


# In[49]:


##### fixed Jul 6. 2023
def compGene2css_work(whole_gene_file,df): 
    """
    Input: Reference gene file, df (CSS)
    Output: list of chromosome-wise list that contains the css at "genic" area only.
    """
    
    print("Extracting the CSS on the genic region ...")

    ########### new fancy gene table without overlap ###########
#     new_gene_lst_all=gene_removeDupl(whole_gene_file) ##### fixed June 29. 2023
    g_df_chr_lst=whGene2GLChr(whole_gene_file)  ##### fixed June 29. 2023
    new_gene_lst_all=merge_intervals(g_df_chr_lst) ##### fixed June 29. 2023
    ############################################################
    
    #### Remove chrM ###########################################
    contains_chrM = df['chromosome'].str.contains('chrM').any()  #check whether it contains M
    if contains_chrM:
        df= df[~df['chromosome'].str.contains('chrM')]
    
    contains_chrY = df['chromosome'].str.contains('chrY').any()
    
    ##### if the target file does not contain Y, remove Y in the gene list file
    if not contains_chrY:
        new_gene_lst_all=new_gene_lst_all[:-1] ## the final element is for Y
    ############################################################
    
    assert len(df["chromosome"].unique())==len(new_gene_lst_all)
        
    css_lst_chr=df2longcss(df) # list of long css per chromosome
    total_chr=len(new_gene_lst_all)
    
    css_gene_lst_all=[]
    for i in tqdm_notebook(range(total_chr)):
        css=css_lst_chr[i]   # long css of i-th chromosome
        gene_df=new_gene_lst_all[i] # gene df of i-th chromosome
        
        css_gene_lst_chr=[]
        for j in range(len(gene_df)):
            g_start=gene_df["TxStart"].iloc[j]-1  # python counts form 0
            g_end=gene_df["TxEnd"].iloc[j]+1      # python excludes the end
            
            css_gene=css[g_start:g_end]           # cut the gene area only
            css_gene_lst_chr.append(css_gene)     # store in the list
          
        css_gene_lst_all.append(css_gene_lst_chr)  # list of list
    
    assert len(css_gene_lst_all)==total_chr
    print("Done!")
    return css_gene_lst_all  ## long version css


# In[50]:


##### fixed June 29. 2023
def compNonGene2css_work(whole_gene_file,df): 
    """
    Input: Reference gene file, df (CSS)
    Output: list of chromosome-wise list that contains the css at "non-genic" area only.
    """
    
    print("Extracting the CSS on the intergenic region ...")

    ########### new fancy gene table without overlap ###########
#     new_gene_lst_all=gene_removeDupl(whole_gene_file) ##### fixed June 29. 2023
    g_df_chr_lst=whGene2GLChr(whole_gene_file)  ##### fixed June 29. 2023
    new_gene_lst_all=merge_intervals(g_df_chr_lst) ##### fixed June 29. 2023
    ############################################################
    
    #### Remove chrM ###########################################
    contains_chrM = df['chromosome'].str.contains('chrM').any()  #check whether it contains M
    if contains_chrM:
        df= df[~df['chromosome'].str.contains('chrM')]
    
    contains_chrY = df['chromosome'].str.contains('chrY').any()
    
    ##### if the target file does not contain Y, remove Y in the gene list file
    if not contains_chrY:
        new_gene_lst_all=new_gene_lst_all[:-1] ## the final element is for Y
    ############################################################
    
    assert len(df["chromosome"].unique())==len(new_gene_lst_all)
        
    css_lst_chr=df2longcss(df) # list of long css per chromosome
    total_chr=len(new_gene_lst_all)
    
    css_Ngene_lst_all=[]
        
    for i in tqdm_notebook(range(total_chr)):
        css=css_lst_chr[i]   # long css of i-th chromosome
        gene_df=new_gene_lst_all[i] # gene df of i-th chromosome
        
#         assert gene_df["TxStart"].iloc[0]>=1, "Gene starts from the very first location at {}-th chromosome.".format(i)
#         assert gene_df["TxEnd"].iloc[-1]<=len(css), "Gene ends at the very last location at {}-th chromosome.".format(i)  
        ### asertion was removed because it produces an error when trying to apply to cells without Y chr.        
    
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
    
    return css_Ngene_lst_all   ## long version css


# In[51]:


# now working on here
def pickled_df2gene_unit_css_new(df_pickled_path="../database/roadmap/df_pickled/", output_path="../database/roadmap/", verbose=True):
    """
    Save unit CSS for genic, for the entire 127 epigenomes
    """
    df_pickled_files = [os.path.join(df_pickled_path, df) for df in sorted(os.listdir(df_pickled_path))]
    
    def load_pickled_df(df_pickled_file):
        with open(df_pickled_file, "rb") as f:
            df = pickle.load(f)
        return df
    
    for file in df_pickled_files:
        cell_id = file.split("/")[-1][:4]  

        gene_output_name = output_path +"gene_css_unit_pickled/"+ cell_id + "_gene_css_pickled.pkl"
        df=load_pickled_df(file)

        css_gene_lst_all=compGene2css_work(whole_gene_file,df)  # use existing one for genic regions
        css_gene_unit_lst_all=Convert2unitCSS_main_new(css_gene_lst_all, unit=200)

        with open(gene_output_name, 'wb') as g:
            pickle.dump(css_gene_unit_lst_all, g)

        if verbose:
            print(cell_id+" is done")

    return print("All done!")


# In[52]:


def pickled_df2Ngene_unit_css_new(df_pickled_path="../database/roadmap/df_pickled/", output_path="../database/roadmap/", verbose=True):
    """
    Save unit CSS for Intergenic, for the entire 127 epigenomes
    """
    df_pickled_files = [os.path.join(df_pickled_path, df) for df in sorted(os.listdir(df_pickled_path))]
    
    def load_pickled_df(df_pickled_file):
        with open(df_pickled_file, "rb") as f:
            df = pickle.load(f)
        return df
    
    for file in df_pickled_files:
        cell_id = file.split("/")[-1][:4]  
        
#         if int(cell_id[1:])>115:  # temp

        Ngene_output_name = output_path +"Ngene_css_unit_pickled/"+ cell_id + "_Ngene_css_pickled.pkl"
        df=load_pickled_df(file)

        css_Ngene_lst_all=compNonGene2css_work(whole_gene_file,df)
        css_Ngene_unit_lst_all=Convert2unitCSS_main_new(css_Ngene_lst_all, unit=200)

        with open(Ngene_output_name, 'wb') as g:
            pickle.dump(css_Ngene_unit_lst_all, g)

        if verbose:
            print(cell_id+" is done")

    return print("All done!")


# ### Just to save entire long-version css per chromosome, except for chrM 

# `save_longcss` was conducted and the files are saved at `"/data1/chromatin_state/database_backup/roadmap_long_css/"`

# In[54]:


def removeChrM(df):
    #### Remove chrM ###########################################
    contains_chrM = df['chromosome'].str.contains('chrM').any()  #check whether it contains M
    if contains_chrM:
        df= df[~df['chromosome'].str.contains('chrM')]
    return df


# In[55]:


def save_longcss(df_pickled_path, output_path="/data1/chromatin_state/database_backup/roadmap_long_css/", verbose=True):
    file_lst = [os.path.join(df_pickled_path,file) for file in sorted(os.listdir(df_pickled_path))]
    counter = 0  # Add a counter
    if verbose:
        print("output path = ", output_path)
    for file in file_lst:
        cell_id = file.split("/")[-1][:4]
        with open(file,"rb") as f:
            df = pickle.load(f)
        df = removeChrM(df)
        long_css = df2longcss(df)        
        output_file_name = cell_id + "_longcss_woChrM.pkl"
        with open(os.path.join(output_path, output_file_name),"wb") as g:  # Use os.path.join
            pickle.dump(long_css, g)
        counter += 1  # Increment the counter
        if verbose and counter % 10 == 0:  # If counter is divisible by 10
            print(f"{counter} files have been saved.")
    print("All saved.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 3-3-3-1. CSS for 57 Epigenomes Genic regions are saved.

# #### Function:` extGenic_byCell`
# * Input: output path
# * This function cut CSS of each cell type by Genic area, and reduce it as unit length
# * Output: function has been already executed, and pickled at `../database/temp_files/whole_gene_unit/`
#     * The saved file names are like `E003_css_gene_unit_lst_all.pkl`
# * **Note** that it takes up to 10 hours to complete if you use macbook pro.

# In[2]:


# Save the whole gene area of the 57 epigenomes, in CSS unit sequences (total no. 56, because no E000 for CSS)
# Following function has been already executed, and pickled at "../database/temp_files/whole_gene_unit/"

def extGenic_byCell(output_path="../database/temp_files/whole_gene_unit/", verbose=True):
    """
    Extract the genic area CSS from the designated 57 epigenome in EG.name.txt
    and save them at "../database/temp_files/whole_gene_unit/"
    """
    # note that EG.name.txt contains E000 (which is not in CSS bed file)
    bed_file_path="../database/bed/unzipped/"
    epi_name_path="../database/bed/gene_expression/EG.name.txt"

    epi_name_df=pd.read_csv(epi_name_path, names=["epi_num","epi_name"], sep="\t", header=None, index_col=False)
    epi_name_df=epi_name_df.dropna()
    epi_num=epi_name_df["epi_num"].dropna().to_list() # number, 0th field
    epi_name=epi_name_df["epi_name"].dropna().to_list() # name, 1st field
    bed_file_lst=sorted(os.listdir(bed_file_path))
    
    # list comprehension for extract the bed files that corresponds to the target epigenome
    epi_target_tuple=[(num, bed_file) for num in epi_num for bed_file in bed_file_lst if num in bed_file]
    epi_target=[tup[1] for tup in epi_target_tuple]
    path="../database/bed/unzipped/"
    
#     print(epi_name_df)
    for epi in epi_target:
        cell_type=epi_name_df.loc[epi_name_df["epi_num"]==epi[:4],"epi_name"].values[0]
        if verbose: 
            print("{}: {} is now processed ...".format(epi, cell_type))
        
        df_epi=bed2df_expanded(path+epi)  # create df of the css for the cell
        css_epi_gene_lst_all=compGene2css(whole_gene_file,df_epi) # list of the css on the genic region
        css_epi_gene_unit_lst_all=Convert2unitCSS_main(css_epi_gene_lst_all,unit=200) # make css to unit length 
        # note that the above list is chromosome-wise list
        
        # total number of genes        
        print("Total number of genes: {}".format(len(flatLst(css_epi_gene_unit_lst_all))))
        
        # pickle it!
        epi_gene_css_name=output_path+epi[:4]+"_css_gene_unit_lst_all.pkl"
        with open(epi_gene_css_name, "wb") as f:
            pickle.dump(css_epi_gene_unit_lst_all,f)

    return print("Files are pickled at {}.".format(output_path))   


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


# In[ ]:





# #### Function: `saveCUTs_all`
# 
# * Simply save the file created from the above fucntion: k-merized genic and intergenic unit-length css
# * 3mer, 4mer files are already stored at `../database/fine_tune/genic_and_intergenic/`
# * Usage
# > `saveCUTs_all(css_gene_unit_lst_all, 510, 3, gene=True)`
# > saves the css on the genic region after 3-merization.

# In[1]:


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
    return print("{}merized files for {} are saved at {}.".format(k,unit_css,path+kmer))


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


# ## 3-5. Complexity of CSS in genic area

# **[back to index](#Index)**

# ### 3-5-1. Create a matrix to show the statistics

# #### Function: `complexity_overview_mat`
# 
# * Usage: Produce a dataframe describing the complexity of the CSS pattern
# * Input: list of css (Here, `gene_css_all` which is pickled at `"../database/temp_files/"`)
# * Columns: `["length","uniq","switch","uniq_pro","switch_pro"]`
# * `length`: gene length
# * `uniq`: How many unique states are 
# * `switch`: How many times the states changed
# * `uniq_pro`: Proportion of `uniq` per gene length
# * `switch_pro`: Proportion of `switch` per gene length
# * Output: dataframe

# In[1]:


def complexity_overview_mat(chr_gene_css):
    abs_uniq_all=[]
    abs_switch_all=[]
    gene_len_all=[]
    compl_uniq_all=[]
    compl_swit_all=[]
    for num in range(len(chr_gene_css)):
        gene_css=chr_gene_css[num]
        gene_css_len=len(gene_css)
        css_uniq=len(set(gene_css)) # only the unique css (min=1, max=gene_css_len)
        
        tot_char=""
        for i, char in enumerate(gene_css):
            if i==0 or char!=gene_css[i-1]:
                tot_char+=char
            css_switch=len(tot_char) # num. of swtiching in css (min=1, max=gene_css_len)
            complexity_uniq=css_uniq/gene_css_len
            complexity_swit=css_switch/gene_css_len
        
        gene_len_all.append(gene_css_len)
        abs_uniq_all.append(css_uniq)
        abs_switch_all.append(css_switch)
        compl_uniq_all.append(complexity_uniq)
        compl_swit_all.append(complexity_swit)
        
    data=list(zip(gene_len_all,abs_uniq_all, abs_switch_all,compl_uniq_all,compl_swit_all))
    df=pd.DataFrame(data,columns=["length","uniq","switch","uniq_pro","switch_pro"])
    df=df[df["length"]>=2]  # remove when the length = 1 unit (=200 bps)
    
    return df


# ### 3-5-2. Extract the complex and less complex css on gene

# #### Function: `extract_complex_css`
# 
# * Usage: Return two lists (css on complex gene / less complex gene) 
# * Input: list of css (Here, `gene_css_all` which is pickled as `"../database/temp_files/css_gene_unit_lst_all"`)
# * Output: `comp_gene_css_all`,`less_comp_gene_css_all`
# * Above output files are stored as pickle, at `"../database/temp_files/complexity` using following commands:
# 
# `with open("../database/temp_files/complexity/thres_mean/comp", "wb") as f:
#     pickle.dump(comp_gene_css_all,f)`
#     
# `with open("../database/temp_files/complexity/thres_mean/less_comp", "wb") as g:
#     pickle.dump(less_comp_gene_css_all,g)`
# 

# In[1]:


# extract according to the complexity

def extract_complex_css(gene_css_all, thres="mean"):
    '''
    Load the file first by `pickle.load(open("../database/temp_files/css_gene_unit_lst_all","rb"))`
    This function will extract the css of gene which is defined as complex in css pattern.
    '''
    tot_gene_css=flatLst(gene_css_all) # flatten it from 24 chromosomes
    tot_gene_css=[gene_css for gene_css in tot_gene_css if len(gene_css)>=2] # length<2 removed
    
    df=complexity_overview_mat(tot_gene_css) # from the process, length<2 was removed
    # df columns=["length","uniq","switch","uniq_pro","switch_pro"]     
    assert len(tot_gene_css)==len(df), "length of tot_gene_css and df do not match"
    
    df["css"]=tot_gene_css # add new column with css (per gene)
        
    comp_gene_css_all=[]
    less_comp_gene_css_all=[]
    
    if thres=="mean":
        thres_val=np.mean(df["switch_pro"])
    
    for i, css in enumerate(tot_gene_css):
        if df["switch_pro"].iloc[i]>=thres_val:
            comp_gene_css_all.append(df["css"].iloc[i])
        else:
            less_comp_gene_css_all.append(df["css"].iloc[i])
        
    return comp_gene_css_all,less_comp_gene_css_all


# ### 3-5-2-1. CSS for 57 Epigenomes Complex and Less Complex Genic regions are saved.

# In[4]:


# Save the complex and less complex genic area of the 57 epigenomes, in CSS unit sequences
# Following function has been already executed, and pickled at "../database/temp_files/complexity/thres_mean/byCellType/"

def extCompGenic_byCell(output_path="../database/temp_files/complexity/", thres="mean", all_file=True, verbose=True, **kwargs):
    """
    This function extract CSS complex and less-complex genic region, according to the threshold.
    (1) To process all the .pkl file in ../database/temp_files/whole_gene_unit/, set 'all_file=True'.
        If you want to process only one file at a time, set e.g.) 'file=E003_css_gene_unit_lst_all.pkl'
    """
    
    css_gene_path="../database/temp_files/whole_gene_unit/"
    if thres=="mean":
        output_path_mod=output_path+"thres_"+thres+"/byCellType/"
    else:
        print("No threshold other than 'mean'.")
    
    # File list of CSS on genic region for all cell types
    files_under_folder=sorted(os.listdir(css_gene_path))
    cell_gene_css_all=[file for file in files_under_folder if file.startswith('E') and file.endswith('.pkl')]
    
    if all_file:
        if verbose: print("processing all files ...")
        for epi_css in tqdm_notebook(cell_gene_css_all):             
            epi_num=epi_css[:4] # e.g.) E003
            if verbose: print("{} is now processed ...".format(epi_num))
            file_path=css_gene_path+epi_css
            with open(file_path,"rb") as f:
                cell_gene_css=pickle.load(f)
            comp_gene_css,less_comp_gene_css=extract_complex_css(cell_gene_css, thres=thres)
            comp_name=output_path_mod+epi_num+"_comp_gene_css.pkl"
            less_name=output_path_mod+epi_num+"_less_comp_gene_css.pkl"
            with open(comp_name,"wb") as g:
                pickle.dump(comp_gene_css, g)
            with open(less_name,"wb") as h:
                pickle.dump(less_comp_gene_css, h)  
                           
    elif len(kwargs)>0:
        for file_key, file_name in kwargs.items():            
            epi_num=file_name[:4]
            file_path=css_gene_path+file_name
            if verbose: print("all_file=False, processing single case for {}.".format(epi_num))
            with open(file_path,"rb") as f:
                cell_gene_css=pickle.load(f)
            comp_gene_css,less_comp_gene_css=extract_complex_css(cell_gene_css, thres=thres)
            comp_name=output_path_mod+epi_num+"_comp_gene_css.pkl"
            less_name=output_path_mod+epi_num+"_less_comp_gene_css.pkl"
            with open(comp_name,"wb") as g:
                pickle.dump(comp_gene_css, g)
            with open(less_name,"wb") as h:
                pickle.dump(less_comp_gene_css, h)               
    else:
        raise ValueError("Set all_file=True, or desginate any file name to proceed!")
    
    return print("Results are stored at {}".format(output_path_mod))


# In[ ]:





# ### 3-5-3. Cut into Kmer and save

# #### Function: `css_CUT_Kmer` (general form of `chr_css_CUT_Kmer`)
# 
# * Usage: For any list of CSS, cut them when it is longer than `cut_thres`, and make it `k`-mer
# * Input: list of css (Here, `comp_gene_css_all` which is generated from the above fnt `extract_complex_css`)
# * Output: `splitted` (raw splitted list),`kmerized_unit_css` (k-merized form)

# In[1]:


# Cut if it is longer than 510
def css_CUT_Kmer(css, cut_thres=510, k=5):
    """ 
    A GENERAL version of `chr_css_CUT_Kmer` and updated to remove any nan in sequence
    Prepare kmer dataset for unit_css, as is if length<=510, else cut it to be length>510   
    Usage: css_CUT_Kmer(css, cut_thres, k)
    
    - css: unit-length css (e.g. comp_gene_css_all)
    - cut_thres: length of split, default=510
    - k: kmer
    
    Output: 1. splitted (before kmerization) 2. kmerized_unit_css (after kmerization) 
    """    
    splitted=[] # bucket for the all the splitted strings   
    for css_elm in css:
        if len(css_elm) <k:  # if the length of css_elm is shorter than k (cannot create k-mer)
            continue
        elif len(css_elm) <=cut_thres:
            splitted.append(css_elm)
        else:  
            prev=0
            while True:
                splitted.append(css_elm[prev:prev+cut_thres])
                prev+=cut_thres
                if prev>=len(css_elm)-1:
                    break      

    kmerized_unit_css_raw=[seq2kmer(item, k) for item in splitted] # k-merize here
    
    ### this part is updated to prevent any empty string to be generated ###
    kmerized_unit_css=[item for item in kmerized_unit_css_raw if item!=""]
    ########################################################################
    
    return splitted, kmerized_unit_css


# In[2]:


# # Cut if it is longer than 510
# def css_CUT_Kmer(css, cut_thres=510, k=5):
#     """ 
#     A GENERAL version of `chr_css_CUT_Kmer`
#     Prepare kmer dataset for unit_css, as is if length<=510, else cut it to be length>510   
#     Usage: css_CUT_Kmer(css, cut_thres, k)
    
#     - css: unit-length css (e.g. comp_gene_css_all)
#     - cut_thres: length of split, default=510
#     - k: kmer
    
#     Output: 1. splitted (before kmerization) 2. kmerized_unit_css (after kmerization) 
#     """    
#     splitted=[] # bucket for the all the splitted strings   
#     for css_elm in css:
#         if len(css_elm) <k:  # if the length of css_elm is shorter than k (cannot create k-mer)
#             continue
#         elif len(css_elm) <=cut_thres:
#             splitted.append(css_elm)
#         else:  
#             prev=0
#             while True:
#                 splitted.append(css_elm[prev:prev+cut_thres])
#                 prev+=cut_thres
#                 if prev>=len(css_elm)-1:
#                     break      
            
#     kmerized_unit_css=[seq2kmer(item, k) for item in splitted] # k-merize here
    
#     return splitted, kmerized_unit_css


# #### Function: `save_as_txt` 
# 
# * Usage: simply save the list as txt file, under the path, with the designated file name.
# * Input: list of css (Here, `comp_gene_css_all` which is generated from the fnt `extract_complex_css`)
# * Remarks: This file includes the above function `css_CUT_Kmer`
# * Output: None, just displaying that it is saved.

# In[6]:


def save_as_txt(css, path="../database/wo_telo/", filename="complex_gene_all", cut_thres=510, k=5):
    
    _, kmerized_unit_css=css_CUT_Kmer(css, cut_thres, k)
    
    full_path=path+filename+"_"+str(k)+".txt"
    with open(full_path,"w") as save_file:
        save_file.write("\n".join(kmerized_unit_css))
    return print("{} is saved at {}".format(filename, path))  


# ### 3-5-4. Show the composition for each case

# #### Function: `css_composition_piechart` 
# 
# * Usage: After running `css_CUT_Kmer`, show the composition of CSS in either complex or less complex genic area
# * Input: splitted_lst can be the first production of the function `css_CUT_Kmer`
# * complexity: `True`=splitted (produced from `comp_gene_css_all`, `False`=less_splitted (produced from `less_comp_gene_css_all`) 
# * show_pct: threshold to show the percentage in pie chart (default=5)
# * Output: None, just displaying the pie chart.

# In[1]:


def css_composition_piechart(splitted_lst, complexity=True, show_pct=5):
    """
    Usage: css_composition_piechart(splitted_lst, complexity=True, show_pct=5)
    Input: splitted_lst can be the first production of the function "css_CUT_Kmer"
    complexity: True=splitted (produced from comp_gene_css_all, False=less_splitted (produced from less_comp_gene_css_all)
    show_pct: threshold to show the percentage in pie chart
    """
    state_count = {chr(i): 0 for i in range(ord('A'), ord('O')+1)}
    for elm in splitted_lst:
        for state in elm:
            if state in state_count:
                state_count[state] += 1  # create a dictionary, value of which is the no. of state appeared overall
    total = sum(state_count.values())
    sizes = [i/sum(state_count.values())*100 for i in state_count.values()] # percentage of occupation
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.pie(state_count.values(),colors=[state_col_dict[label] for label in state_count.keys()], autopct=lambda p: '{:.2f}%'.format(p) if p > show_pct else '')

    if complexity:
        title="Complex gene CS composition,"+" total:"+" "+str(total)
    else:
        title="Less complex gene CS composition,"+" total:"+" "+str(total)
    
    for t in ax.texts:
        t.set_color("white")
        t.set_fontsize(20)

    ax.set_title(title,fontsize=20)
    plt.show()


# ### 3-5-5. Prepare and save Fine-tuning for Complex gene CSS and others

# #### Function: `prep_and_saveTF_CompNcomp` 
# 
# * Usage: Prerpare and save the fine-tuning data for **complex** and **less complex gene css**
# * Input files are loaded inside the function, which are pickled at `"../database/temp_files/complexity/thres_mean/"`
# * Output: None, just displaying the report that the file is saved.

# In[34]:


# now for compG and nonCompG (the function covers from prepration to save)
def prep_and_saveTF_CompNcomp(condition="thres_mean", cut_thres=510, k=5, save_path="CompG_and_lessCompG",len_tr=20000, len_dev=1000):
    """
    prepare fine tuning data for [the complex gene css / less complex gene css]
    """
    print("* Project name: ", save_path)
    print("* condition: ", condition)
    print("* Cut threshold length: ", cut_thres)
    print("* k-merization: ", k)
    print("* train: dev = {} : {}".format(len_tr,len_dev))
    
    comp_path="../database/temp_files/complexity/"+condition+"/comp"
    comp=pickle.load(open(comp_path, "rb"))
    less_comp_path="../database/temp_files/complexity/"+condition+"/less_comp"
    less_comp=pickle.load(open(less_comp_path, "rb"))
    
    # kmerization
    _, comp_kmerized=css_CUT_Kmer(comp, cut_thres, k)
    _, less_comp_kmerized=css_CUT_Kmer(less_comp, cut_thres, k)
    
    # make it dataframe
    df_comp=pd.DataFrame(comp_kmerized, columns=["sequence"])
    df_comp["label"]=1
    df_less_comp=pd.DataFrame(less_comp_kmerized, columns=["sequence"])
    df_less_comp["label"]=0
    
    # make them have the same length
    if len(df_comp)>len(df_less_comp):
        df_comp=df_comp[:len(df_less_comp)] 
    elif len(df_comp)<len(df_less_comp):
        df_less_comp=df_less_comp[:len(df_comp)]
    assert len(df_comp)==len(df_less_comp), "Check the data length."
    
    # shuffling ...
    df_comp_all=pd.concat([df_comp,df_less_comp]).sample(frac=1).reset_index(drop=True)  

    # cutting into train and dev
    assert len(df_comp_all)> len_tr+len_dev, "Not enough data length."
    df_comp_train=df_comp_all[:len_tr]
    df_comp_dev=df_comp_all[len_tr:len_tr+len_dev]    
  
    path="../database/fine_tune/"+save_path+"/"+str(k)+"mer/"
    train_name=path+"train.tsv"
    dev_name=path+"dev.tsv"
    
    df_comp_train.to_csv(train_name, sep="\t", index=False)
    df_comp_dev.to_csv(dev_name, sep="\t", index=False)

    return print("Fine-tuning data for {} are {}merized and saved at {}.".format(save_path,k,path))


# #### Function: `prep_and_saveTF_CompNgene` 
# 
# * Usage: Prerpare and save the fine-tuning data for **complex** and **None gene css**
# * Input files are loaded inside the function, which are pickled at `"../database/temp_files/complexity/thres_mean/"` for complex gene, and at `"../database/temp_files/css_Ngene_unit_lst_all"` for intergenic area (a.k.a. Ngene)
# * Output: None, just displaying the report that the file is saved.

# In[4]:


# now,  for compG and non gene (the function covers from prepration to save)
def prep_and_saveTF_CompNgene(condition="thres_mean", cut_thres=510, k=5, save_path="CompG_and_intergenic",len_tr=20000, len_dev=1000):
    """
    prepare fine tuning data for [the complex gene css / none gene css]
    """
    print("* Project name: ", save_path)
    print("* condition: ", condition)
    print("* Cut threshold length: ", cut_thres)
    print("* k-merization: ", k)
    print("* train: dev = {} : {}".format(len_tr,len_dev))
    
    comp_path="../database/temp_files/complexity/"+condition+"/comp"
    comp=pickle.load(open(comp_path, "rb"))
    Ngene_path="../database/temp_files/css_Ngene_unit_lst_all"
    Ngene=pickle.load(open(Ngene_path, "rb"))
    #flatten
    Ngene=flatLst(Ngene)
    
    # kmerization
    _, comp_kmerized=css_CUT_Kmer(comp, cut_thres, k)
    _, Ngene_kmerized=css_CUT_Kmer(Ngene, cut_thres, k)
    
    # make it dataframe
    df_comp=pd.DataFrame(comp_kmerized, columns=["sequence"])
    df_comp["label"]=1
    df_Ngene=pd.DataFrame(Ngene_kmerized, columns=["sequence"])
    df_Ngene["label"]=0
    
    # make them have the same length
    if len(df_comp)>len(df_Ngene):
        df_comp=df_comp[:len(df_Ngene)] 
    elif len(df_comp)<len(df_Ngene):
        df_Ngene=df_Ngene[:len(df_comp)]
    assert len(df_comp)==len(df_Ngene), "Check the data length."
    
    # shuffling ...
    df_compNgene=pd.concat([df_comp,df_Ngene]).sample(frac=1).reset_index(drop=True)  

    # cutting into train and dev
    assert len(df_compNgene)> len_tr+len_dev, "Not enough data length."
    df_compNgene_train=df_compNgene[:len_tr]
    df_compNgene_dev=df_compNgene[len_tr:len_tr+len_dev]    
  
    path="../database/fine_tune/"+save_path+"/"+str(k)+"mer/"
    train_name=path+"train.tsv"
    dev_name=path+"dev.tsv"
    
    df_compNgene_train.to_csv(train_name, sep="\t", index=False)
    df_compNgene_dev.to_csv(dev_name, sep="\t", index=False)

    return print("Fine-tuning data for {} are {}merized and saved at {}.".format(save_path,k,path))


# ## 3-6. Gene expression classification

# For more difficult tasks, gene expression can be one of the criteria to prepare fine tuning data. First, using the gene expression level from RNA-seq, highly expressed 

# ### 3-6-1. Gene expression file into the list of dataframe

# #### Function: `Gexp_Gene2GLChr`
# 
# * This function only checks a single file.
# * Usage: After the gene expression files such as `gene_highlyexpressed.refFlat` are acquired by `/database/bed/gene_expression/classifygenes_ROADMAP_RPKM.py`, apply this function to obtain the list of dataframe per chromosome contains the transcription start and end indices.
# * Input: gene expression (high/low/not) file
# * Output: a chromosome-wise list of dataframe containing `TxStart` and `TxEnd`

# In[14]:


# function for preprocess the whole gene data and produce chromosome-wise gene lists
# each element is dataframe

### this function is not essential, but just to check by create df from .refFlat
def Gexp_Gene2GLChr(exp_gene_file='../database/bed/gene_expression/E050/gene_highlyexpressed.refFlat'):
    print("Extracting the gene file ...")
    g_fn=exp_gene_file
    g_df_raw=pd.read_csv(g_fn, sep='\t', index_col=False, header=0)
    g_df=g_df_raw
    g_df=g_df.iloc[:,1:]
    g_df.rename(columns={"name":"gene_id"}, inplace=True)
    g_df.rename(columns={"#geneName":"geneName"}, inplace=True)
    g_df.rename(columns={"txStart":"TxStart"}, inplace=True) # to make it coherent to my previous codes
    g_df.rename(columns={"txEnd":"TxEnd"}, inplace=True)
#     g_df=g_df_raw.rename(columns={0:"geneName",1:"gene_id",2:"chrom",3:"strand",4:"txStart",5:"txEnd",
#                                       6:"cdsStart",7:"cdsEnd",8:"exonCount",9:"exonStart",10:"exonEnds",
#                                       11:"gene type",12:"transcript type",13:"reference transcript name",
#                                       14:"reference transcription id"})
    ## string to the list of "int", for exon start/end ##
    g_df_temp=g_df # copy for processing
    exon_start_int_lst=[]
    for i, str_lst in enumerate(g_df_temp["exonStarts"]):
        int_lst=[int(elm) for elm in str_lst.replace("[","").replace("]","").split(",")]
        assert g_df_temp["exonCount"][i]==len(int_lst) # make sure the no. element in exon st count
        exon_start_int_lst.append(int_lst)    
    g_df_temp["exonStarts"]=exon_start_int_lst

    exon_end_int_lst=[]
    for i, str_lst in enumerate(g_df_temp["exonEnds"]):
        int_lst=[int(elm) for elm in str_lst.replace("[","").replace("]","").split(",")]
        assert g_df_temp["exonCount"][i]==len(int_lst) # make sure the no. element in exon start = count
        exon_end_int_lst.append(int_lst)    
    g_df_temp["exonEnds"]=exon_end_int_lst    
    g_df=g_df_temp # and make it back the original name
        
    g_df=g_df[["geneName","gene_id","chrom","TxStart","TxEnd"]] # extract these only
    
    # Remove other than regular chromosomes
    chr_lst=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10',
             'chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19',
             'chr20','chr21','chr22','chrX','chrY']
    g_df=g_df.loc[g_df["chrom"].isin(chr_lst)]
    
    # Create a list of chromosome-wise dataframe 
    g_df_chr_lst=[]
    for num in range(len(chr_lst)):
        chr_num=chr_lst[num]
        g_chr_df='g_'+chr_num  # name it as "g_"
        locals()[g_chr_df]=g_df[g_df["chrom"]==chr_num]
        g_chr_df=locals()[g_chr_df]
        g_chr_df=g_chr_df.sort_values("TxStart")
        g_df_chr_lst.append(g_chr_df)
        
    # Remove the overlapped area (using removeOverlapDF function in css_utility.py)
    g_df_chr_collapsed_lst=[]
    for g_df_chr in g_df_chr_lst:
        g_df_chr_collapsed=removeOverlapDF(g_df_chr)
        assert len(g_df_chr)>=len(g_df_chr_collapsed)
        g_df_chr_collapsed_lst.append(g_df_chr_collapsed)
    print("Done!")
    
    return g_df_chr_collapsed_lst  # list of dataframe


# ### 3-6-2. Matching to CSS

# #### Function: `comp_expGene2css`
# 
# * Usage: modified from `compGene2css`, Use it like  `css_gene_lst_all=comp_expGene2css("../database/bed/gene_expression/gene_highlyexpressed.refFlat",df_e050)`
# * Input: 
#     * (highly/low/not) expressed gene, such as `"../database/bed/gene_expression/gene_highlyexpressed.refFlat"`
#     * df, acquired from css created by bed2df_expanded
# * Output
#     * list of chromosome-wise list that contains the css at (highly/low/not) genic area only.
# * **caution!** Do not forget to conduct `Convert2unitCSS_main(css_gene_lst_all, unit=200)`, to convert the result into 200-bps unit length

# In[1]:


def comp_expGene2css(exp_gene_file,df):   # df indicates css, created by bed2df_expanded
    """
    modified from `compGene2css`
    Input: Reference gene file, df (CSS)
    Output: list of chromosome-wise list that contains the css at (expressed) genic area only.
    """
    g_lst_chr=Gexp_Gene2GLChr(exp_gene_file)
#     g_lst_chr=whGene2GLChr(whole_gene_file) # list of gene table df per chromosome
    css_lst_chr=df2longcss(df) # list of long css per chromosome
    total_chr=len(g_lst_chr)
    
    print("Matching to the chromatin state sequence data ...")
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
    
    # remove chromosome if it is empty (e.g. chrY for female)
    css_gene_lst_all=[elm for elm in css_gene_lst_all if elm!=[]] 
            
    print("Done!")
    return css_gene_lst_all ## this is the original length! reduce it at Convert2unitCSS_main(css_lst_all, unit=200)!


# ### 3-6-2-1. CSS for various gene expression cases are saved.

# #### Function `extExpGenic_byCell_1_ver01`
# * From the css bed file for each cell, expressed genic region and highly expressed genic region `refFlat` data are saved by running the "classifygenes_ROADMAP_RPKM.py". To complete, execute `extExpGenic_byCell_2`.
# * Input: output path
# * Usage example: `extExpGenic_byCell_1_ver01(output_path="../database/temp_files/expressed/byCellType/refFlat/", all_file=False, high_only=True, verbose=True, exp=0, high_exp=10, file="E050_15_coreMarks_stateno.bed")`
# * In `ver01`, the argument `high_only` is added to produce highly_expressed case only, as the "expressed" is the same (rpkm > 0)
# * This function was executed and the result is already saved. See `../database/bed/gene_expression/byCellType/refFlat/rpkm10`

# In[13]:


def extExpGenic_byCell_1_ver01(output_path="../database/temp_files/expressed/byCellType/refFlat/", all_file=True, high_only=True, verbose=True, exp=0, high_exp=50, **kwargs):
    """
    RUN THE SECOND function 'extExpGenic_byCell_2' after running this function.
    This function extract CSS expressed genic region, mainly for "expressed" and "highly-expressed"
    (1) To process all the  ... set 'all_file=True'.
        If you want to process only one file at a time, set e.g.) all_file=False, file="E050_15_coreMarks_stateno.bed"
    (2) High_only = True will only produce the highly expressed cases (default) 
    (3) Outputs are e.g.) "E112_gene_expressed.refFlat", "E112_gene_highlyexpressed.refFlat" at output path
    """
    
    output_path_mod=output_path+"rpkm"+str(high_exp)+"/"
    
    path="../database/bed/gene_expression/"
    script="classifygenes_ROADMAP_RPKM.py"
    epi_rpkm_tsv="57epigenomes.RPKM.pc.tsv"
    gene_ref="chr.gene.refFlat"
    original_path="~/Work/chromatin_state/NSP/"
    
    save_path="./byCellType/refFlat/"+"rpkm"+str(high_exp)+"/"
    css_bed_path="../database/bed/unzipped/"

    if all_file:
        css_gene_path="../database/temp_files/whole_gene_unit/"
        # File list of CSS on genic region for all cell types
        files_under_folder=sorted(os.listdir(css_gene_path))
        cell_gene_css_all=[file for file in files_under_folder if file.startswith('E') and file.endswith('.pkl')]
        
#         all_css_bed_file=sorted(os.listdir(css_bed_path)) # all css bed file, we need to choose the target
#         # list comprehension to choose the targets (57 epigenomes)    
#         target_cell_gene_css=[css_bed for css_bed in all_css_bed_file for epi in cell_gene_css_all if css_bed[:4]==epi[:4]]
        
        if verbose: print("processing all files ...")
        for epi_css in tqdm_notebook(cell_gene_css_all):             
            epi_num=epi_css[:4] # e.g.) E003
            
            if verbose: print("{} is now processed ...".format(epi_num))
            file_path=css_bed_path+epi_css
#             df=bed2df_expanded(file_path)  # css df

            ######## Running the script at working path and come back to the original path #########
            get_ipython().run_line_magic('cd', '-q {path}')
            get_ipython().run_line_magic('run', '{script} --thre_expressed {exp} --thre_highlyexpressed {high_exp} {epi_rpkm_tsv} {epi_num} {gene_ref}')

            if not high_only:
                exp_file_name=save_path+epi_num+"_"+"gene_expressed.refFlat"
            hexp_file_name=save_path+epi_num+"_"+"gene_highlyexpressed.refFlat"
            get_ipython().run_line_magic('mv', '"gene_expressed.refFlat" {exp_file_name}')
            get_ipython().run_line_magic('mv', '"gene_highlyexpressed.refFlat" {hexp_file_name}')
            get_ipython().run_line_magic('cd', '-q {original_path}')
            ########################################################################################
                
        
    elif len(kwargs)>0:       
        for file_key, file_name in kwargs.items():            
            epi_num=file_name[:4]
            if verbose: print("all_file=False, processing single case for {}.".format(epi_num))

            file_path=css_bed_path+file_name
#             df=bed2df_expanded(file_path)  # css df for the designated file
            
            ######## Running the script at working path and come back to the original path #########
            get_ipython().run_line_magic('cd', '-q {path}')
            get_ipython().run_line_magic('run', '{script} --thre_expressed {exp} --thre_highlyexpressed {high_exp} {epi_rpkm_tsv} {epi_num} {gene_ref}')

            if not high_only:
                exp_file_name=save_path+epi_num+"_"+"gene_expressed.refFlat"
            hexp_file_name=save_path+epi_num+"_"+"gene_highlyexpressed.refFlat"
            get_ipython().run_line_magic('mv', '"gene_expressed.refFlat" {exp_file_name}')
            get_ipython().run_line_magic('mv', '"gene_highlyexpressed.refFlat" {hexp_file_name}')
            get_ipython().run_line_magic('cd', '-q {original_path}')
            ########################################################################################
            
    else:
        raise ValueError("Set all_file=True, or desginate any file name to proceed!")
    assert os.getcwd()[-3:]=="NSP", "Check the current working directory."
    
    return print("Results are stored at {}, and current working directory is : {}".format(output_path_mod, os.getcwd()))
                           


# #### Function `extExpGenic_byCell_2_ver01`
# * Expressed genic region, highly expressed genic region's css data are saved.
# * Input: output path, `high_only` for selecting whether just "expressed" will be included. `high_exp` is for designating the RPKM
# * Output: output folder name will be like `rpkm50`
# * This function was executed and the result is already saved.
# * To check the result, visit the output path.

# In[17]:


def extExpGenic_byCell_2_ver01(output_path="../database/temp_files/expressed/byCellType/",all_file=True, high_only=True, high_exp=50, verbose=True, **kwargs):
    """
    Should be executed after extExpGenic_byCell_1_ver01
    modified the previous version to make it applicalbe to highly_expressed only extraction
    with high_only=True, highly expressed gene according to the high_exp value (RPKM) are extracted.
    """
    exp_ref_path="../database/bed/gene_expression/byCellType/refFlat/"
    hexp_ref_path=exp_ref_path+"rpkm"+str(high_exp)+"/"
    
    ref_exp_file_all=sorted(os.listdir(exp_ref_path))
    ref_hexp_file_all=sorted(os.listdir(hexp_ref_path))
    
    ref_exp_all=[elm for elm in ref_exp_file_all if '_expressed' in elm and elm.startswith('E')]
    ref_hexp_all=[elm for elm in ref_hexp_file_all if 'high' in elm and elm.startswith('E')]
      
    css_gene_path="../database/temp_files/whole_gene_unit/"
    css_bed_path="../database/bed/unzipped/"
    css_bed_file_all=sorted(os.listdir(css_bed_path))    

    if all_file:
        if verbose: print("processing all files ...")
        for epi_css in tqdm_notebook(ref_hexp_all):
            epi_num=epi_css[:4]
            if verbose: print("{} is now processed ...".format(epi_num))
            # preparing df from bed
            target_bed=[elm for elm in css_bed_file_all if elm[:4]==epi_num]
            bed_path=css_bed_path+target_bed[0]
            df=bed2df_expanded(bed_path)
            # preparing ref from exp_refs
            target_hexp_ref=[elm for elm in ref_hexp_all if elm[:4]==epi_num]
            target_exp_ref=[elm for elm in ref_exp_all if elm[:4]==epi_num]
            hexp=hexp_ref_path+target_hexp_ref[0]
            exp=exp_ref_path+target_exp_ref[0]

            if not high_only:  # extract just 'expressed' case if high_only is False (default=True)
                css_exp_gene_lst=comp_expGene2css(exp,df)
                css_exp_gene_unit_lst=flatLst(Convert2unitCSS_main(css_exp_gene_lst, unit=200))
                with open(output_path+"expressed/"+epi_num+"_exp_gene_css.pkl","wb") as g:
                    pickle.dump(css_exp_gene_unit_lst,g)
                    
            css_hexp_gene_lst=comp_expGene2css(hexp,df)
            css_hexp_gene_unit_lst=flatLst(Convert2unitCSS_main(css_hexp_gene_lst, unit=200))
            with open(output_path+"rpkm"+str(high_exp)+"_highly_expressed/"+epi_num+"_highly_exp_gene_css.pkl","wb") as f:
                pickle.dump(css_hexp_gene_unit_lst,f)
            
    elif "file" in kwargs:
        file_name=kwargs["file"]
#         for file_key, file_name in kwargs.items():            
        epi_num=file_name[:4]
        if verbose: print("all_file=False, processing single case for {}.".format(epi_num))
        # preparing df from bed
        target_bed=[elm for elm in css_bed_file_all if elm[:4]==epi_num]
        bed_path=css_bed_path+target_bed[0]
        df=bed2df_expanded(bed_path)
        # preparing ref from exp_refs
        target_hexp_ref=[elm for elm in ref_hexp_all if elm[:4]==epi_num]
        target_exp_ref=[elm for elm in ref_exp_all if elm[:4]==epi_num]
        hexp=hexp_ref_path+target_hexp_ref[0]
        exp=exp_ref_path+target_exp_ref[0]
        
        if not high_only:  # extract just 'expressed' case if high_only is False (default=True)
            css_exp_gene_lst=comp_expGene2css(exp,df)
            css_exp_gene_unit_lst=flatLst(Convert2unitCSS_main(css_exp_gene_lst, unit=200))
            with open(output_path+"expressed/"+epi_num+"_exp_gene_css.pkl","wb") as g:
                pickle.dump(css_exp_gene_unit_lst,g)

        css_hexp_gene_lst=comp_expGene2css(hexp,df)
        css_hexp_gene_unit_lst=flatLst(Convert2unitCSS_main(css_hexp_gene_lst, unit=200))
        with open(output_path+"rpkm"+str(high_exp)+"_highly_expressed/"+epi_num+"_highly_exp_gene_css.pkl","wb") as f:
            pickle.dump(css_hexp_gene_unit_lst,f)

    else:
        raise ValueError("Set all_file=True, or desginate any file name to proceed!")
    return


# #### Function `extNOTexp_Genic_byCell`
# * NOT expressed genic region's css data are saved.
# * Input: output path
# * This function was executed and the result is already saved.
# * To check the result, visit the output path.

# In[8]:


def extNOTexp_Genic_byCell(output_path="../database/temp_files/expressed/byCellType/not_expressed/", all_file=True, verbose=True, **kwargs):
  # This function only compares the whole genic with expressed genic and subtract them.
  # Perhaps should be changed later?
    css_exp_path="../database/temp_files/expressed/byCellType/expressed/"
    css_whole="../database/temp_files/whole_gene_unit/"
    whole_gene_files=sorted(glob.glob(css_whole+"*.pkl"))
    exp_gene_files=sorted(glob.glob(css_exp_path+"*.pkl"))

    if all_file: 
        if verbose: print("processing all files ...")
        for gene_file in tqdm_notebook(whole_gene_files):
            pattern=r'E[0-9]+'
            epi_num=re.findall(pattern, gene_file)[0] # e.g.) 'E003'
            # take expressed gene list for the same cell type
            exp_gene_file=[file for file in exp_gene_files if epi_num in file][0]
            with open(gene_file,"rb") as f:
                whole_gene=flatLst(pickle.load(f))
            with open(exp_gene_file, "rb") as g:
                exp_gene=pickle.load(g)
            not_exp_gene_all=[]
            not_exp_gene=[gene for gene in whole_gene if gene not in exp_gene]
            not_exp_gene_all.append(not_exp_gene)
            with open(output_path+epi_num+"_not_exp_gene_css.pkl","wb") as h:
                pickle.dump(not_exp_gene,h)
    
    elif "file" in kwargs:
        exp_gene_file=kwargs["file"]    
        epi_num=exp_gene_file[:4]
        exp_gene_file_w_path=css_exp_path+exp_gene_file
        assert epi_num[0]=="E", "File name should start with 'E'. Remove any path before the file name."
        if verbose: print("all_file=False, processing single case for {}.".format(epi_num))
        
        gene_file=[elm for elm in whole_gene_files if epi_num in elm][0]        
        with open(gene_file,"rb") as f:
            whole_gene=flatLst(pickle.load(f))
        with open(exp_gene_file_w_path, "rb") as g:
            exp_gene=pickle.load(g)
        not_exp_gene=[gene for gene in whole_gene if gene not in exp_gene]
        with open(output_path+epi_num+"_not_exp_gene_css.pkl","wb") as h:
            pickle.dump(not_exp_gene,h)
    else:
        pass
    return print("files are saved at {}".format(output_path))


# ### 3-6-3. Cut into Kmer and save

# #### Function `save_kmers_ver01`
# * **Note** that this function for highly_expressed case is not useful because the pretrain is conducted for whole_cell
# * Input: output path, k for kmerization, kwargs should include "kind"
# * Usage: e.g.) `save_kmers(k=4,kind="whole_gene")`
# * Output: none, **note** that this function is already executed and `.txt` files for the pretraining have been saved. Visit the output path indicated in the function.

# In[18]:


def save_kmers_ver01(output_path="../database/pretrain/expressed/", high_exp=50, k=4,**kwargs):
    """
    "kind" for kwargs can be chosen among ("whold_gene","not_expressed","expressed", "highly_expressed")
    if "kind" is highly_expressed, RPKM value should be provided as high_exp.
    But this is not very meaningful, because pretrain is conducted with whole_gene only...
    """
    input_path="../database/temp_files/"
    epi_num_lst=pd.read_csv("../database/temp_files/whole_gene_unit/epigenome_lst.txt", header=None, names=["num"])
    epi_num=epi_num_lst["num"].tolist()
    if high_exp:
        print("The threshold for highly expressed is set as RPKM={}".format(high_exp))
    for num in tqdm_notebook(epi_num):   
        if 'kind' in kwargs:
            gene_type=kwargs["kind"]
            if gene_type=="whole_gene":
                file_name=input_path+gene_type+"_unit/"+num+"_css_gene_unit_lst_all.pkl"
            elif gene_type=="not_expressed":
                file_name=input_path+"expressed/byCellType/"+gene_type+"/"+num+"_not_exp_gene_css.pkl"
            elif gene_type=="expressed":
                file_name=input_path+"expressed/byCellType/"+gene_type+"/"+num+"_exp_gene_css.pkl"
            ### note that there is subfolder for highly expressed case
            elif gene_type=="highly_expressed":
                file_name=input_path+"expressed/byCellType/"+"rpkm"+str(high_exp)+"_"+gene_type+"/"+num+"_highly_exp_gene_css.pkl"
            else:
                pass
            with open(file_name, "rb") as f:
                target=pickle.load(f)
                ####### whole_gene is not flat list #######
                if gene_type=="whole_gene":
                    target=flatLst(target)
                ###########################################
                _, kmerized_unit_css=css_CUT_Kmer(target, k=k)
            
        if gene_type=="highly_expressed":
            output_path_mod=output_path+str(k)+"mer/"+"rpkm"+str(high_exp)+"_"+gene_type+"/"+num+"_"+gene_type+".txt"
        else:
            output_path_mod=output_path+str(k)+"mer/"+gene_type+"/"+num+"_"+gene_type+".txt"
        with open(output_path_mod,"w") as g:
            g.write("\n".join(kmerized_unit_css))           
    return 
    
    


# #### Function `save_kmers`
# * The simplest version?
# * usage: `save_kmers(k=4,kind="whole_gene")`

# In[1]:


def save_kmers(output_path="../database/pretrain/",k=4,**kwargs):
    input_path="../database/temp_files/"
    epi_num_lst=pd.read_csv("../database/temp_files/whole_gene_unit/epigenome_lst.txt", header=None, names=["num"])
    epi_num=epi_num_lst["num"].tolist()
    for num in tqdm_notebook(epi_num):   
        if 'kind' in kwargs:
            gene_type=kwargs["kind"]
            if gene_type=="whole_gene":
                file_name=input_path+gene_type+"_unit/"+num+"_css_gene_unit_lst_all.pkl"
            elif gene_type=="not_expressed":
                file_name=input_path+"expressed/byCellType/"+gene_type+"/"+num+"_not_exp_gene_css.pkl"
            elif gene_type=="expressed":
                file_name=input_path+"expressed/byCellType/"+gene_type+"/"+num+"_exp_gene_css.pkl"
            elif gene_type=="highly_expressed":
                file_name=input_path+"expressed/byCellType/"+gene_type+"/"+num+"_highly_exp_gene_css.pkl"
            else:
                pass
            with open(file_name, "rb") as f:
                target=pickle.load(f)
                ####### whole_gene is not flat list #######
                if gene_type=="whole_gene":
                    target=flatLst(target)
                ###########################################
                _, kmerized_unit_css=css_CUT_Kmer(target, k=k)
            output_path_mod=output_path+"expressed/"+str(k)+"mer/"+gene_type+"/"+num+"_"+gene_type+".txt"
            with open(output_path_mod,"w") as g:
                g.write("\n".join(kmerized_unit_css))
           
    return 


# ### 3-6-4. Fine-tuning data

# #### Function: `prep_and_saveTF_ver01`
# * Save the fine-tuning data for gene expression
# * Three different binary classifications are possible: exp vs. not exp, rpkmNN_highly exp vs. exp, rpkmNN_highly exp vs. not exp
# * Can be used with following inputs, for example:
#     <blockquote>
#     input_path="../database/temp_files/expressed/byCellType/" <br>
#     output_path="../database/fine_tune/gene_exp/4mer/Gexp_or_not" <br>
#     cl1="expressed" <br>
#     cl2="not_expressed" <br>
#     epi_num_lst=["E003","E128"] <br>
#     </blockquote>
# * This function already executed for the above conditions. See `../database/fine_tune/gene_exp/4mer`

# In[19]:


# For saving gene expression fine-tuning data
def prep_and_saveTF_ver01(input_path, output_path, cl1, cl2, epi_num_lst, cut_thres=510, k=4, len_tr=20000, len_dev=1000):
    """
    * Generallized function for preparing fine tuning data.
    * Input path will be in the temp_files
    * cl1 and cl2 refer to the name of class you want to classify in binary classification.
    * cl1 and cl2 are any of "expressed", "not_expressed", "rpkmNN_highly_expressed" (NN is number)
    * epi_num_lst should contain the name of epigenomes like "E003." If you need more, then add like ["E003", "E004", ...]
    """
    print("* Input path: ", input_path)
    print("* Binary classification for '{}' and '{}'".format(cl1, cl2))
#     ans= "yes" if incl_hexp else "no"
#     print("* Including highly expressed case: {}".format(ans))
    print("* Output path: ", output_path)
    print("* Cut threshold length: ", cut_thres)
    print("* k-merization: ", k)
    print("* train: dev = {} : {}".format(len_tr,len_dev))
    
    cl1_path=input_path+cl1+"/"
    cl2_path=input_path+cl2+"/"
    
    cl1_concat=[]
    cl2_concat=[]
    
    suffix_dict = {}
    for cl in [cl1, cl2]:
        if "highly" in cl:
#             rpkm_no=re.search(r'rpkm(\d+)',cl).group(1) # no.. this is not required. already inside the name of cl1 an cl2
            suffix_dict[cl] = "_highly_exp_gene_css.pkl"
        elif "not" in cl:
            suffix_dict[cl] = "_not_exp_gene_css.pkl"
        else:
            suffix_dict[cl] = "_exp_gene_css.pkl"
    
    for cl, concat_lst in [(cl1, cl1_concat), (cl2, cl2_concat)]:
        for epi_num in epi_num_lst:
            file_path = input_path + cl + "/" + epi_num + suffix_dict[cl]
            concat_lst.extend(pickle.load(open(file_path, "rb")))
    
    # kmerization
    _, cl1_kmerized=css_CUT_Kmer(cl1_concat, cut_thres, k)
    _, cl2_kmerized=css_CUT_Kmer(cl2_concat, cut_thres, k)
    
    # make it dataframe
    df_cl1=pd.DataFrame(cl1_kmerized, columns=["sequence"])
    df_cl1["label"]=1
    df_cl2=pd.DataFrame(cl2_kmerized, columns=["sequence"])
    df_cl2["label"]=0
    
    # make them have the same length
    if len(df_cl1)>len(df_cl2):
        df_cl1=df_cl1[:len(df_cl2)] 
    elif len(df_cl1)<len(df_cl2):
        df_cl2=df_cl2[:len(df_cl1)]
    assert len(df_cl1)==len(df_cl2), "Check the data length."
    
    # shuffling ...
    df_all=pd.concat([df_cl1,df_cl2]).sample(frac=1).reset_index(drop=True)  

    # cutting into train and dev
    assert len(df_all)> len_tr+len_dev, "Not enough data length."
    df_train=df_all[:len_tr]
    df_dev=df_all[len_tr:len_tr+len_dev]    
  
    #path="../database/fine_tune/"+save_path+"/"+str(k)+"mer/"
    
    by_tr_len=str("{:.0f}".format(len_tr/1000))
    output_path_mod=output_path+"tr_len_"+by_tr_len+"k/"
    
    # create a destination folder if it does not exist.
    if os.path.exists(output_path_mod):
        raise ValueError("Folder already exists:{}".format(output_path_mod))
    else:
        os.makedirs(output_path_mod)
    
    train_name=output_path_mod+"train.tsv"
    dev_name=output_path_mod+"dev.tsv"
    
    df_train.to_csv(train_name, sep="\t", index=False)
    df_dev.to_csv(dev_name, sep="\t", index=False)

    return print("Fine-tuning data for {} and {} (epigenome no. {}) are {}merized and saved at {}.".format(cl1, cl2, epi_num_lst, k, output_path_mod))


# ### 3-6-5. Pie chart statistics: generalized verion

# #### Funtion `css_composition_piechart_Gen`
# * Input: path for .pkl or the list of "splitted" acquired directly from the function `css_CUT_Kmer` 
#     * Either one of the path for .pkl or splitted shold be provided. 
# * Usage
#     * Create a piechar to show the composition of each state in a certain list of css.
#     * e.g.) `css_composition_piechart_Gen(load_pkl=True, pkl_path="../database/temp_files/expressed/byCellType/highly_expressed/",show_pct=5, title="highly_expressed")`
# * Output: Just a piechart for showing the composition of the css list.

# In[4]:


#Generalized version, for splitted (the result of css_CUT_Kmer) or from the pkl file saved 
# e.g.) css_composition_piechart_Gen(load_pkl=True, pkl_path="../database/temp_files/expressed/byCellType/highly_expressed/",show_pct=5, title="highly_expressed")
def css_composition_piechart_Gen(load_pkl=True, pkl_path=None, splitted=None, show_pct=5, **kwargs):
    """
    Usage: css_composition_piechart using the data from either .pkl or splitted (after running css_CUT_Kmer)
    Input: .pkl file path or2for , splitted_lst can be the first production of the function "css_CUT_Kmer"
    show_pct: threshold to show the percentage in pie chart
    """
    ### case 1. when you load .pkl which is usually stored at ../database/temp_files
    if load_pkl:
        if pkl_path is None:
            raise ValueError("Path for the folder including .pkl files is required if load_pkl is True.")
        else:
            pkl_files = sorted([f for f in os.listdir(pkl_path) if f.endswith('.pkl')])
            css_concat=[]
            for pkl_file in pkl_files:
                with open(pkl_path + pkl_file, "rb") as f:
                    css = pickle.load(f)
                    if isinstance(css[0], list):
                        css_flat = flatLst(css)
                        css_concat.extend(css_flat)
                    else:
                        css_concat.extend(css)
        splitted=css_concat

    ### case 2. when you use splitted, the first one of the results from the function css_CUT_kmer
    else:
        if splitted is None:
            raise ValueError("Splitted is required. Run the css_CUT_Kmer first.")
    
    splitted_lst=splitted
    num_elm=len(splitted_lst)
    print("total {} of fragments.".format(num_elm))
    
    state_count = {chr(i): 0 for i in range(ord('A'), ord('O')+1)}
    for elm in splitted_lst:
        for state in elm:
            if state in state_count:
                state_count[state] += 1  # create a dictionary, value of which is the no. of state appeared overall
    total = sum(state_count.values())
    sizes = [i/sum(state_count.values())*100 for i in state_count.values()] # percentage of occupation
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 6))

    ax1.pie(state_count.values(),colors=[state_col_dict[label] for label in state_count.keys()], autopct=lambda p: '{:.2f}%'.format(p) if p > show_pct else '')

    if "title" in kwargs:
        ax1.set_title(kwargs["title"], fontsize=15)
    
    for t in ax1.texts:
        t.set_color("white")
        t.set_fontsize(15)
        
    # print the list of percentages and states
    # uncomment this if you want to use text rather than picture-table
#     for state, size in zip(state_count.keys(), sizes):
#         num_states = int(round(size * total / 100))
#         print(f"{state}. {num_states} ({size:.2f}%)")

    # Hide axis
    ax2.axis('off')

    # Create table
    table = ax2.table(cellText=list(zip(state_count.keys(), [f"{size:.2f}" for size in sizes])),
                     colLabels=['State', 'Proportion'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.5, 1)
    
    plt.show()
    


# ## 3-7. Promoter classification
# **[back to index](#Index)**

# ### 3-8-1. Prmototer region by location

# #### Function `remove_chrM_and_trim_gene_file_accordingly`
# * remove the chromosome M from the all chromosome per cell, and trim the gene file
# * outputs now have the same list of chromosomes

# In[47]:


def remove_chrM_and_trim_gene_file_accordingly(whole_gene_file,df):
    
    ########### new fancy gene table without overlap ###########
    g_df_chr_lst=whGene2GLChr(whole_gene_file)  ##### fixed June 29. 2023
    new_gene_lst_all=merge_intervals(g_df_chr_lst) ##### fixed June 29. 2023
    ############################################################

    #### Remove chrM ###########################################
    contains_chrM = df['chromosome'].str.contains('chrM').any()  #check whether it contains M
    if contains_chrM:
        df= df[~df['chromosome'].str.contains('chrM')]

    contains_chrY = df['chromosome'].str.contains('chrY').any()

    ##### if the target file does not contain Y, remove Y in the gene list file
    if not contains_chrY:
        new_gene_lst_all=new_gene_lst_all[:-1] ## the final element is for Y
    ############################################################

    assert len(df["chromosome"].unique())==len(new_gene_lst_all)
    return new_gene_lst_all, df


# #### Function `ext_TSS_by_loc`
# * This function extracts the TSS regions with respect to gene location
# * Run this function for a cell
# * Output css are all reduced to unit length

# In[45]:


def ext_TSS_by_loc(whole_gene_file, df, up_num=2000, down_num=4000, gene_init=2000, unit=200):
    """
    extract TSS region by location estimation. 
    input: (1) whole_gene_file: the raw gene bed file (2) df: per cell (3) up_num: upstream (4) down_num: downstream (5) gene_init: how long the initial region would be
    output: (1) gene_start_lst_all: only gene start point per chr (2) tss_by_loc_css_unit_all: window_based (3) 
    """
    new_gene_lst_all, trimmed_df = remove_chrM_and_trim_gene_file_accordingly(whole_gene_file, df)
    
    css_lst_chr = df2longcss(trimmed_df) # list of long css per chromosome
    total_chr = len(new_gene_lst_all)
    
    gene_start_lst_all = []
    tss_by_loc_css_all = []
    tss_by_init_css_all = []
    for i in range(total_chr):
        gene_start_lst = new_gene_lst_all[i]["TxStart"]
        gene_start_lst_all.append(gene_start_lst) ### Gene start point only
        css_lst = css_lst_chr[i]
        
        tss_by_loc_css_chr = []
        tss_by_init_css_chr = []
        for j in range(len(gene_start_lst)):
            gene_start = gene_start_lst[j]
            win_start = max(0, gene_start - up_num)  # use max to prevent negative index
            win_end = min(len(css_lst), gene_start + down_num)  # use min to prevent index out of range

            tss_by_loc_css = css_lst[win_start:win_end]
            tss_by_loc_css_chr.append(tss_by_loc_css)
            
            tss_by_init_css = css_lst[gene_start:gene_start+gene_init]
            tss_by_init_css_chr.append(tss_by_init_css)
            
        tss_by_loc_css_all.append(tss_by_loc_css_chr)
        tss_by_init_css_all.append(tss_by_init_css_chr)
        
    tss_by_loc_css_unit_all=Convert2unitCSS_main_new(tss_by_loc_css_all, unit=unit)  
    tss_by_init_css_unit_all=Convert2unitCSS_main_new(tss_by_init_css_all,unit=unit)
        
    return gene_start_lst_all, tss_by_loc_css_unit_all, tss_by_init_css_unit_all   


# In[ ]:





# In[ ]:





# In[ ]:





# ## 3-8. Enhancer classification
# **[back to index](#Index)**

# ### 3-8-1. Pretrain dataset

# #### Funtion `cutKmerByCell`
# * Input: file path of a bed file like`"../database/temp_files/whole_genome/byCellType/E001_whole_css_wo_telo.txt"`)
# * Output: randomly cut from 5 to 510, without telomere, filtered to have longer than k, css list
# * Further work: save it as follows
# > `with open(output_path,"w") as save_file: `<br>
# >  <hspace> `save_file.write("\n".join(filtered_kmerized_unit_css))`

# In[2]:


def cutKmerByCell(unzipped_bed_file_path,k=4):
    df=bed2df_expanded(path)
    unit_css=df2unitcss(df)
    assert isinstance(unit_css[0], str) 
    if len(unit_css[0])>=50 and len(unit_css[-1])>50:
        unit_css[0]=unit_css[0][50:] # cut the telomere
        unit_css[-1]=unit_css[-1][:-50] # cut the telomere
        
    _, kmerized_unit_css=css_CUT_Kmer(unit_css, cut_thres=510, k=k)
    
    filtered_kmerized_unit_css=[item for item in kmerized_unit_css if len(item)>=k]
    return filtered_kmerized_unit_css


# In[ ]:





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

# ## 5-1. Evaluation

# ### 5-1-2. Pretrain evaluation

# In[4]:


def evalDFconcat(df_lst, col_name, col_rename, colormap="Set1"):
    """
    Input
    (1) df_lst: a list of target dataframes
    (2) col_name: the columns of interest
    (3) col_rename: a list of columns for the concatenated dataframes
    """
    assert type(df_lst), "Input should be a list of dataframes"
    assert type(col_rename), "col_rename should be a list"
    assert len(df_lst)==len(col_rename), "Check the length of input list"
    assert col_name in df_lst[0].columns, "'{}' is not in the column list of dataframe".format(col_name)
    df_col_lst=[]
    for num in range(len(df_lst)):       
        df_col_lst.append(df_lst[num][col_name])
    df_concat=pd.concat(df_col_lst, axis=1)
    df_concat.columns=col_rename
    
    fig=plt.figure(figsize=(6,4))
    p=sns.lineplot(data=df_concat, palette=colormap)
    p.set_ylabel(col_name, fontsize=14)
    p.set_xlabel("Iteration", fontsize=14)
#     p.set_ylim(1.0,2.8)
    p.legend(fontsize=14)
    
    return df_concat


# #### Function: `evalPre_by_folder` 
# 
# * Usage: Create a dataframe and show the result plot of pretraining 
# * Input: path of the pretraining result, basically under the folder `../database/pretrain/`
# * User input: `"all"` or a list of integer, such as `[0,1,2]`, as you can select from the list this function shows. 
# * Output: Plot of perplexity

# In[5]:


def evalPre_by_folder(path,target='all',colormap="Set1", ylim=6.5):
    """
    path: the directory you have the pretrain result (eval_results.txt)
          Multiple files can be processed.
    target: if designated as "all", it considers all the files. 
            Otherwise, a list containing the numbers of the file you want to analyze should be given.
    """
    file_list=[os.path.join(path, file) for file in sorted(os.listdir(path))]
    print("\n".join(file_list))

    target = input("Enter 'all' to process all files or a list of file numbers to process (ex. [1,2,3]): ")

    if target == 'all':
        target = 'all'
    else:
        try:
            target = ast.literal_eval(target)
            if not all(isinstance(i, int) for i in target):
                raise ValueError("Invalid input, target should be 'all' or a list of integers.")
            for i in target:
                if i > len(file_list):
                    raise ValueError("Invalid file number")
        except (ValueError, SyntaxError):
            raise ValueError("Invalid input, target should be 'all' or a list of integers in the format [1,2,3].")

    file_df_all=[]
    if target=='all':        
        for i, file in enumerate(file_list):
            f_name=re.search(r'eval_results_(.*).txt', file).group(1)
            file_df=pd.read_csv(file, header=None, names=["perplexity"])
            file_df.rename(columns={'perplexity': f_name}, inplace=True)
            file_df_all.append(file_df)
        result_df = pd.concat(file_df_all, axis=1)
        
    elif type(target)==list and type(target[0])==int:
        for i in target:
            f_name=re.search(r'eval_results_(.*).txt', file_list[i]).group(1)
            file_df=pd.read_csv(file_list[i], header=None, names=["perplexity"])
            file_df.rename(columns={'perplexity': f_name}, inplace=True)
            file_df_all.append(file_df)
        result_df = pd.concat(file_df_all, axis=1)
        
    fig=plt.figure(figsize=(6,4))
    p=sns.lineplot(data=result_df, palette=colormap)
    p.set_xlabel("Iternation", fontsize=13)
    p.set_ylim([0.5, ylim])
    
    return result_df    


# ### 5-1-3. Fine tuning evaluation

# The result of fine-tuning is provided by a `eval_result.txt` file by default, which contains acc (accuracy), auc (area under curve), mcc (Matthew's correlation coefficient), f1 score, precision, and recall. Those files can be saved with different names which contain the different experimental condition. 
# 
# The series of functions below are the unit functions for internal use, or simple use.

#  #### Function: `evalFT_df`
#  * Create dataframe from the raw file `eval_result.txt`

# In[5]:


def evalFT_df(path):
    """
    Unit function for eval_result.txt obtained after the fine tuning
    """
    df=pd.read_csv(path, index_col=False, sep="\s", header=None, engine='python')
    df.columns=["k","acc","auc","f1","mcc","precision","recall"]
    return df


# #### Function `evalFT_fig`
# * Draw af figure for a single `eval_result.txt` file at the designated path. 
# * Inputs
#     * `path` : path of the raw file
#     * `target` : any of `[acc","auc","f1","mcc","precision","recall"]` as a string, or a sub-list can be accepted
#     * `kwargs` : title can be added.
# * Output: a dataframe where the target is the column, row is the eval over iterations

# In[41]:


def evalFT_fig(path, iteration=60, target="auc", figsize=(4,2.5), colormap="Set1", **kwargs):
    """
    Unit function for drawing the figure only
    "target" should be designated, either string or a list of strings
    """
    df=evalFT_df(path)
    fig=plt.figure(figsize=figsize)
    plt.ylim([0,1])
    plt.ylabel("metrics")
    plt.xlabel("iterations")
    color_lst=sns.color_palette(colormap)
    line_lst=["-","--",":"]

    if "title" in kwargs:
        title=kwargs["title"]
        plt.title(title)
    
    df_target=pd.DataFrame()
    
    if not isinstance(target,list):
        sns.lineplot(df[target][:iteration], label=target)
        plt.legend(loc="lower right")
        auc_avg_f10=np.mean(df["auc"][:iteration].iloc[-10:])
        df_target[target]=list(df[target][:iteration])
        plt.text(2, 0.1, "final 10 AUC avg. "+str(round(auc_avg_f10,3)))
    else:    
        for i, tar in enumerate(target):
            sns.lineplot(df[tar][:iteration], label=tar, linestyle=line_lst[i], color=color_lst[i])
            plt.legend(loc="lower right")
            auc_avg_f10=np.mean(df["auc"][:iteration].iloc[-10:])
            plt.text(2, 0.1, "final 10 AUC avg. "+str(round(auc_avg_f10,3)))
            
            target_val_lst=list(df[tar][:iteration])
            df_target[tar]=target_val_lst
            
    return  df_target 


# #### Function `evalFT_overview`
# * Show the result of evaluation in a figure
# * Input:
#     * `path_all` : either one or multiple paths
#     * `target`: any of `[acc","auc","f1","mcc","precision","recall"]` as a string, or a sub-list can be accepted
# * Output: a dictionary (key: different condition, value: dataframe of evaluation over time, for designated targets)

# In[42]:


def evalFT_overview(path_all,iteration, target,colormap="Set1", show_depth=-2):
    
    if not isinstance(path_all,list):
        df_target=evalFT_fig(path_all,iteration=iteration, target=target, figsize=(4,2.5), colormap=colormap)

    else:
        keys=[]
        values=[]
        for i, path in enumerate(path_all):
            file_name_lst=os.path.splitext(path)[0].split("/")[show_depth:]
            title='   '.join(file_name_lst)
            keys.append(file_name_lst[0])
            df_target=evalFT_fig(path, iteration=iteration, target=target, colormap=colormap, title=title)
            values.append(df_target)    
        dict_df_target=dict(zip(keys,values))
        
    return dict_df_target 


# #### Function `dict_df_target_2_bar_graph`
# 
# * Create a bargraph set for different metrics
# * Run after the `evalFT_overview`

# In[43]:


def dict_df_target_2_bar_graph(dict_df_target):
    conds=dict_df_target.keys()
    dfs=dict_df_target.values()
    targets=list(list(dict_df_target.values())[0].columns) # produces a list like ['acc', 'auc', 'f1']
    
    all_mean_std={}
    for target in targets:  # "acc", "auc", or "f1"       
        dict_mean_std={}
        for cond in conds: # for "rpkm10_or_exp" or "rpkm30_or_not"
            target_val_lst=dict_df_target[cond][target]
            mean_val=round(np.mean(target_val_lst),3)
            std_val=round(np.std(target_val_lst),3)
            mean_std=(mean_val, std_val)
            dict_mean_std[cond]=mean_std
         # dictionary of key "acc", "auc", or "f1" with the value of another dict 'rpkm10_or_exp':(mean,std)
        all_mean_std[target]=dict_mean_std 
    
    ### generate a bar graph
    data=all_mean_std
    color_map = {'acc': 'cornflowerblue', 'auc': 'teal', 'f1': 'tomato'}
    for condition in data.keys():
        # Get the data for this condition
        condition_data = data[condition]

        # Separate keys and values
        keys = list(condition_data.keys())
        mean_values, std_devs = zip(*condition_data.values())

        # Define positions
        x_pos = range(len(keys))

        # Create a bar graph
        plt.figure(figsize=(5,3.5))  # create a new figure for each condition
        plt.bar(x_pos, mean_values, yerr=std_devs, align='center', alpha=0.6, 
                ecolor='gray', capsize=7, color=color_map[condition])

        # Add labels and title
        plt.ylabel(condition)
        plt.xticks(x_pos, keys, rotation=45, ha='right')
        plt.yticks(np.arange(0,1.2,0.2))
        plt.title(f'Mean Values for {condition} with Error Bars (std)')

        # Show the graph
        plt.tight_layout()
        plt.show()
    
    return all_mean_std


# #### Function: `pred_prob_overall` 
# 
# * **updated** for showing the results for separated cases (for the visualization purpose)
# * Usage: Create a dataframe for prediction result and show the result plot (confusion matrix, violin plot)
# * Input: path of the prediction result file (`pred_results.npy`) and the labeled file (`dev.tsv`)
#     * `dev_path="../database/fine_tune/CompG_and_lessCompG/4mer/dev.tsv"`
#     * `pred_path="../database/ft_result/pred/4_compless/pred_results.npy"`
# * Output: Two dataframes (`high_pred`: label 1 and its prediction ,`low_pred`: label 0 and its prediction)

# In[31]:


# def pred_prob_overall(dev_path,pred_path, color1="Blues",color2_lst=["yellowgreen","skyblue","teal","royalblue"]):
#     pred=np.load(pred_path)
#     dev=pd.read_csv(dev_path, sep="\t")
#     dev["pred"]=pred
#     dev["pred_bool"]=None
#     df=dev
    
#     assert type(color2_lst) and len(color2_lst)==4, "enter a list of 4 elements, as color names"
    
#     # confusion matrix #
#     for i in range(len(df)):
#         if df["pred"].at[i]>=0.5 :
#             df["pred_bool"].at[i]=1
#         else:
#             df["pred_bool"].at[i]=0
#     assert df["pred_bool"].isnull().any()==False, "Check the pred_bool"
#     cf_matrix=confusion_matrix(df["label"],df["pred_bool"].astype(bool))

#     group_names = ['True Neg','False Pos','False Neg','True Pos']
#     group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
#     group_percentages = ["({0:.2%})".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
#     labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
#     labels = np.asarray(labels).reshape(2,2)
    
#     # confusion matrix visualization
#     sns.heatmap(cf_matrix, annot=labels, annot_kws={'size': 16}, fmt='', cmap=color1)
#     print(classification_report(df["label"], df["pred_bool"].astype(bool)))
    
#     high_prob, low_prob=[],[]
#     label_1, label_0=[],[]
#     high_1, high_0=[],[]
#     low_1, low_0=[],[]

#     for i in range(len(df)):
#         # high_prob is defined as larger than 0.5       
#         if df["pred"].iloc[i]>=0.5:
#             high_prob.append(df["pred"].iloc[i])
#             label_1.append(df["label"].iloc[i])
#             if df["label"].iloc[i]==1:  # predition is higher than 0.5(=true), and label is 1 (=true): true positive
#                 high_1.append(df["pred"].iloc[i])
#             else:
#                 high_0.append(df["pred"].iloc[i])    
#         else:
#             low_prob.append(df["pred"].iloc[i])
#             label_0.append(df["label"].iloc[i])
#             if df["label"].iloc[i]==0: # predition is lower than 0.5(=false), and label is 0 (=false): true negative
#                 low_0.append(df["pred"].iloc[i])
#             else:
#                 low_1.append(df["pred"].iloc[i])

# #     print("false positive: {} |  false negative: {}".format(false_positive,false_negative))
#     high_pred=pd.DataFrame({'label': label_1, 'pred': high_prob})
#     low_pred=pd.DataFrame({'label': label_0, 'pred': low_prob})

#     fig=plt.figure(figsize=(8,8))
#     plt.subplots_adjust(wspace=0.5, hspace=0.5)
#     plt.subplot(2, 2, 1)
#     sns.violinplot(data=high_prob, color=color2_lst[0])
#     plt.title('High Probability', fontsize=13)
#     plt.xticks([])
#     plt.xlabel("predition >= 0.5", fontsize=13)
#     plt.ylabel("Prediction", fontsize=13)

#     plt.subplot(2, 2, 2)
#     sns.violinplot(data=low_prob, color=color2_lst[1])
#     plt.title('Low Probability', fontsize=13)
#     plt.xticks([])
#     plt.xlabel("predition < 0.5", fontsize=13)
#     plt.ylabel("Prediction", fontsize=13)
    
#     plt.subplot(2, 2, 3)
#     sns.violinplot(data=high_1, color=color2_lst[2])
#     plt.title('True positive', fontsize=13)
#     plt.xticks([])
#     plt.xlabel("For label 1", fontsize=13)
#     plt.ylabel("Prediction", fontsize=13)
    
#     plt.subplot(2, 2, 4)
#     sns.violinplot(data=low_0, color=color2_lst[3])
#     plt.title('True negative', fontsize=13)
#     plt.xticks([])
#     plt.xlabel("For label 0", fontsize=13)
#     plt.ylabel("Prediction", fontsize=13)

#     plt.show()

#     return high_pred,low_pred


# In[4]:


def pred_prob_overall(dev_path,pred_path, file_id, color1="Blues",color2_lst=["yellowgreen","skyblue","teal","royalblue"]):
    pred=np.load(pred_path)
    dev=pd.read_csv(dev_path, sep="\t")
    dev["pred"]=pred
    dev["pred_bool"]=None
    df=dev
    
    file_output_path="../database/figs/"
    
    assert type(color2_lst) and len(color2_lst)==4, "enter a list of 4 elements, as color names"
    
    # confusion matrix #
    for i in range(len(df)):
        if df["pred"].at[i]>=0.5 :
            df["pred_bool"].at[i]=1
        else:
            df["pred_bool"].at[i]=0
    assert df["pred_bool"].isnull().any()==False, "Check the pred_bool"
    cf_matrix=confusion_matrix(df["label"],df["pred_bool"].astype(bool))

    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
#     group_percentages = ["({0:.2%})".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
#     labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
#     labels = np.asarray(labels).reshape(2,2)
########## for visualize with normalization per case####################
    group_percentages = []
    for i in range(cf_matrix.shape[0]):
        for value in cf_matrix[i]:
            group_percentages.append("({0:.2%})".format(value / cf_matrix[i].sum()))

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
######################################################################
    
    # confusion matrix visualization
    sns.heatmap(cf_matrix, annot=labels, annot_kws={'size': 16}, fmt='', cmap=color1)
    
    ############### save the figure 1 ###############
    file_name1=file_output_path+file_id+'_confusion_matrix.pdf'
    plt.savefig(file_name1, format='pdf', bbox_inches='tight')
    #################################################
    print(classification_report(df["label"], df["pred_bool"].astype(bool)))
    
    high_prob, low_prob=[],[]
    label_1, label_0=[],[]
    high_1, high_0=[],[]
    low_1, low_0=[],[]

    for i in range(len(df)):
        # high_prob is defined as larger than 0.5       
        if df["pred"].iloc[i]>=0.5:
            high_prob.append(df["pred"].iloc[i])
            label_1.append(df["label"].iloc[i])
            if df["label"].iloc[i]==1:  # predition is higher than 0.5(=true), and label is 1 (=true): true positive
                high_1.append(df["pred"].iloc[i])
            else:
                high_0.append(df["pred"].iloc[i])    
        else:
            low_prob.append(df["pred"].iloc[i])
            label_0.append(df["label"].iloc[i])
            if df["label"].iloc[i]==0: # predition is lower than 0.5(=false), and label is 0 (=false): true negative
                low_0.append(df["pred"].iloc[i])
            else:
                low_1.append(df["pred"].iloc[i])

#     print("false positive: {} |  false negative: {}".format(false_positive,false_negative))
    high_pred=pd.DataFrame({'label': label_1, 'pred': high_prob})
    low_pred=pd.DataFrame({'label': label_0, 'pred': low_prob})

    fig=plt.figure(figsize=(8,8))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.subplot(2, 2, 1)
    sns.violinplot(data=high_prob, color=color2_lst[0])
    plt.title('High Probability', fontsize=13)
    plt.xticks([])
    plt.xlabel("predition >= 0.5", fontsize=13)
    plt.ylabel("Prediction", fontsize=13)

    plt.subplot(2, 2, 2)
    sns.violinplot(data=low_prob, color=color2_lst[1])
    plt.title('Low Probability', fontsize=13)
    plt.xticks([])
    plt.xlabel("predition < 0.5", fontsize=13)
    plt.ylabel("Prediction", fontsize=13)
    
    plt.subplot(2, 2, 3)
    sns.violinplot(data=high_1, color=color2_lst[2])
    plt.title('True positive', fontsize=13)
    plt.xticks([])
    plt.xlabel("For label 1", fontsize=13)
    plt.ylabel("Prediction", fontsize=13)
    
    plt.subplot(2, 2, 4)
    sns.violinplot(data=low_0, color=color2_lst[3])
    plt.title('True negative', fontsize=13)
    plt.xticks([])
    plt.xlabel("For label 0", fontsize=13)
    plt.ylabel("Prediction", fontsize=13)
    
    ############### save the figure 2 ###############
    file_name2=file_output_path+file_id+'_violinplot.pdf'
    plt.savefig(file_name2, format='pdf', bbox_inches='tight')
    #################################################

    plt.show()

    return high_pred,low_pred


# #### Function `dev_conv`
# 
# * Auxiliary function for creating dataframe with original sequence from `dev.tsv`
# * Input: file path for `dev.tsv`
# * Output: dataframe that accommodates the original sequence, before the kmerization

# In[5]:


def dev_conv(dev_file_path):
    dev_df=pd.read_csv(dev_file_path,sep="\t")
    dev_df.fillna(" ", inplace=True) # change the nan into empty string
    assert dev_df["sequence"].isnull().sum()==0, "check the dev file for nan values"
    
    def kmer2seq_or_space(seq):
        if seq == " ":
            return " "
        else:
            return kmer2seq(seq)
    
    dev_df["ori_seq"] = dev_df["sequence"].apply(kmer2seq_or_space)

    return dev_df


# ## 5-2. Motif

# #### Function `motif_df_initProcessing`
# 
# * Usage: Initial processing for motif dataframe, created by `motif_utils.py`. Adding the columns like '
# * Input: motif dataframe 

# In[1]:


def motif_df_initProcessing(motif_df="../database/motif/compNg_condw24min5ins3_df.csv"):
    fname=motif_df
    hparam=r'cond(\d+)?|w(\d+)|min(\d+)|ins(\d+)'
    matches=re.findall(hparam,fname)
    numbers=[num for match in matches for num in match if num]
    if not matches[0][0]: # if no number after cond (which is actually cond1 AND cond2)
        cond='_'  # replace it with underscore
        win=numbers[0]
        min_len=numbers[1]
        min_ins=numbers[2]
    else:   
        cond=numbers[0]
        win=numbers[1]
        min_len=numbers[2]
        min_ins=numbers[3]
    
    print("condition: {}, windows: {}, min_length: {}, min_instance: {}".format(cond,win,min_len,min_ins))
    
    # add columns "pro_x" and "length" to the dataframe
    df=pd.read_csv(motif_df, engine='python')
    df_sorted=df.sort_values(by="p")   # sort by p-value, ascending order
    df_sorted["pro_x"]=df_sorted["x"]/df_sorted["n"] # add columns for proportional x over n
    df_sorted["length"]=[len(motif) for motif in df_sorted['motif'].tolist()] # and for length
    
    max_motif_len=max(df_sorted["length"])
    min_motif_len=min(df_sorted["length"])
    print("Total found motif number (p-val<0.05): {}".format(len(df_sorted)))
    print("max motif length: {}, min motif length: {}".format(max_motif_len,min_motif_len))
    
    # list of colored motif
    motif_lst=df_sorted["motif"].tolist()
    colored_motif=[colored_css_str_as_is(motif) for motif in motif_lst]
    
    return df_sorted, colored_motif   


# ### 5-2-1. Motif visualization

# #### Function `create_motif_wordcloud`
# 
# * Usage: create a word cloud using `wordcloud` package for representing the frequency of each motif
# * Input: path of the motif (where the file name is like `motif_AAAAA_3_txt`
# * Output: word cloud of the motif

# In[23]:


def create_motif_wordcloud(path, color_map="viridis"):
    target=[word for word in path.split("/")[-2:] if word !=""][0]
    print("target", target)
    file_lst=[os.path.join(path,file) for file in os.listdir(path) if ".txt" in file]
    motifs={}
    for file_name in file_lst:
        motif, num_txt=file_name.split("/")[-1].split("_")[1:3]
        freq=num_txt.split(".")[0]
        motifs[motif]=int(freq)
    print("motifs = ", motifs)
    wc=WordCloud(width=800, height=400, background_color="white", colormap=color_map)
    wordcloud=wc.generate_from_frequencies(motifs)
    plt.figure(figsize=(6,2), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


# #### Function `motif_vis`
# 
# * Usage: create and save a motif on the attention matrix with colored text (for motif only)
# * Input: file_name (e.g. "compNg") and the path for dev.tsv and prediction attention matrix, motif, an instance include motif 
# * Output: pdf file saved at "../database/figs/"

# In[6]:


def motif_vis(file_name, dev_path, atten_path, motif_str, motif_inst):
    """
    input examples) 
     (1) dev_path = "../database/fine_tune/CompG_and_intergenic/4mer/dev.tsv"
     (2) atten_path = "../database/ft_result/pred/4_compNg/atten.npy"
    output: attention matrix segment which shows the motif on it
    """
    dev_df=dev_conv(dev_path)
    atten_mat=np.load(atten_path)
    for i, seq in enumerate(dev_df["ori_seq"].to_list()):
        if motif_inst in seq and dev_df["label"].iloc[i]==1:
            if len(seq)>81:
                seq=seq[:81]  # cut for visualization purpose
            print(motif_str, i, len(seq), seq, "\n")
            
            letters=seq
            
            figure=plt.figure(figsize=(35,2))
            ax =sns.heatmap(data=atten_mat[i:i+1], cmap="viridis")
            sequence = motif_str
            sequence_indices = [i for i in range(len(letters)) if letters.startswith(sequence, i)]
            for j, letter in enumerate(letters):
                if j in sequence_indices or j-1 in sequence_indices or j-2 in sequence_indices or j-3 in sequence_indices or j-4 in sequence_indices:
                    ax.text(j + 0.5, -0.2, letter, ha='center', va='center', color=state_col_dict[letter], weight='bold', fontsize=30)
                else:
                    ax.text(j + 0.5, -0.2, letter, ha='center', va='center',fontsize=16)
           
            plt.tight_layout()
            output_path="../database/figs/motif_vis_"+file_name+"_"+motif_str+"_in_"+motif_inst+".pdf"
            plt.savefig(output_path, format='pdf')
            
            plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




