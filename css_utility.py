#!/usr/bin/env python
# coding: utf-8

# ### Utility file
# Various functions to process the initial bed data

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from motif_utils import seq2kmer


# #### Preparing the .bed file list

# In[8]:


# file name reader
# make a list of all the filename

path='../database/bed/unzipped/'
bed_files=os.listdir(path)

pickle_path='../database/cell_pickle'
pickle_files=os.listdir(pickle_path)

def file_list_maker(path, files):
    all_files=[]
    for file in files:
        file_path=os.path.join(path,file)
        all_files.append(file_path)
    return all_files

all_files=file_list_maker(path, bed_files)
all_cell_pickles=file_list_maker(pickle_path, pickle_files)


# test file
test_filename='../database/bed/unzipped/E017_15_coreMarks_stateno.bed'


# In[5]:


state_dict={1:"A", 2:"B", 3:"C", 4:"D", 5:"E",6:"F",7:"G",8:"H" ,
                9:"I" ,10:"J",11:"K", 12:"L", 13:"M", 14:"N", 15:"O"}


# In[6]:


css_name=['TssA','TssAFlnk','TxFlnk','Tx','TxWk','EnhG','Enh','ZNF/Rpts',
          'Het','TssBiv','BivFlnk','EnhBiv','ReprPC','ReprPcWk','Quies']


# In[7]:


css_dict=dict(zip(list(state_dict.values()), css_name))  # css_dict={"A":"TssA", "B":"TssAFlnk", ... }


# In[8]:


css_color_dict={'TssA':(219, 57, 50), 'TssAFlnk': (222, 87, 54), 'TxFlnk': (107, 187, 87),
                'Tx': (57, 124, 72), 'TxWk': (48, 98, 58), 'EnhG': (197, 213, 80), 'Enh': (245, 196, 98),
                'ZNF/Rpts': (129, 194, 169), 'Het': (137,143,189), 'TssBiv': (192, 98, 95), 
                'BivFlnk': (223, 156, 127), 'EnhBiv': (188, 182, 115), 'ReprPC': (147, 149, 153),
                'ReprPCWk': (200, 202, 203), 'Quies': (240, 240, 240)}


# #### Function to convert RGB into decimal RGB

# In[9]:


def colors2color_dec(css_color_dict):
    colors=list(css_color_dict.values())
    color_dec_list=[]
    for color in colors:
        color_dec=tuple(rgb_elm/255 for rgb_elm in color)
        color_dec_list.append(color_dec)        
    return color_dec_list


# In[10]:


state_col_dict=dict(zip(list(state_dict.values()),colors2color_dec(css_color_dict)))


# In[11]:


state_col_255_dict=dict(zip(list(state_dict.values()),list(css_color_dict.values())))


# #### Function to create pickle file (dataframe, expanded version) for an individual cell

# In[12]:


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


# #### Functions to make .bed to dataframe

# In[13]:


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


# In[14]:


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
    df["unit"]=(df["length"]/100).astype(int)
               
    df["state_seq"]=df["state"].map(state_dict)
    df["state_seq_full"]=df["unit"]*df["state_seq"]
    
    return df 


# In[15]:


def total_df_maker(all_files):
    
    """Create a list of dataframe from a list of bed files.]
    This function utilizes the function named 'bed2df_expanded.'"""
    
    total_df=[]
    for filename in all_files:
        df=bed2df_expanded(filename)
        total_df.append(df)
    return total_df


# #### Functions for analyzing an individual dataframe
# 
# CSS here refers Chromatin state sequence

# In[16]:


def numchr(df):
    assert "chromosome" in df.columns, "Check your df has the column named 'chromosome'"
    return df["chromosome"].nunique()    


# In[17]:


# create a large piece of string of the whole state_seq_full 
# CSS: chromatin-state sequence

def df2css_allchr(df):
    
    """Create a large piece of string of the whole state_seq_full 
    This function generates a string from the entire chromosomes"""
    
    state_seq_full_list=df["state_seq_full"].tolist()
    state_seq_full_to_str=''.join([elm for elm in state_seq_full_list ])
    return state_seq_full_to_str


# #### Create CSS chromosome-wise

# In[18]:


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


# #### Create CSS chromosome-wise, string only

# In[19]:


# create a list of dataframes, each of which contains the name of chromosome and chromosome-wise string of state_seq_full
# This is prerequisite function for df2css_chr_string

def df2css_chr(df):
   
    """Create a list of dataframes, each of which containing 
    the chromosome name and the state_seq_full per chromosome"""
    
    start=0
    df_chr_list=[]
    chr_index=df2chr_index(df)
    
    for index in chr_index:
        df_chr=df[["chromosome","state_seq_full"]][start:index+1] # note that python [i:j] means from i to j-1
        chr_name=df["chromosome"].iloc[start] # string, such as chr1, chr2, ...
        df_name='df_'+chr_name  # the chromosome-wise data stored like df_chr1, df_chr2, ...
        locals()[df_name]=df_chr # make a string into a variable name
        df_chr_list.append(df_chr)
        start=index+1
    
    return df_chr_list    


# In[20]:


def df2css_chr_str(df):
    
    """Create a list of strings which is the state_seq_full, all-connected per chromosome"""
    
    chr_index=df2chr_index(df)  
    chr_index_num=len(chr_index) 

    df_chr_list=df2css_chr(df)  # contains a list of df: chromosome name, state_seq_full (2-column datafame)
    chr_css_list=[]

    for num in range(chr_index_num): 
        css_full_list=df_chr_list[num]["state_seq_full"].tolist()  # extract the state_seq_full only and make it a list
        css_full_to_str=''.join([elm for elm in css_full_list]) # make it a long string of all-connected state_seq_full (chromosome-wise)
        chr_css_list.append(css_full_to_str)
    return chr_css_list


# #### CSS Pattern analysis
# 
# Now the dataframe has been transformed into a list of string all connected css, chromosome-wise.<br>
# The variable of the above list is now called chr_css_list.<br>
# Following functions will analyze the statistics of the each strings.

# In[21]:


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





def colored_css_str(sub_str):
    col_str=""
    for letter in sub_str:
        for state in list(state_col_255_dict.keys()):
            if letter==state:
                r=state_col_255_dict[letter][0]
                g=state_col_255_dict[letter][1]
                b=state_col_255_dict[letter][2]
                col_letter="\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r,g,b,letter)
                col_str+=col_letter
    return print("\033[1m"+col_str+"\033[0;0m") 







