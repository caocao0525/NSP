#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import seaborn as snsg_hexp
from css_utility import *
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
import glob


# In[9]:


whole_gene_file


# In[3]:


g_df_chr_collapsed_lst=Gexp_Gene2GLChr(exp_gene_file='../database/bed/gene_expression/E050/gene_highlyexpressed.refFlat')


# In[21]:


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
                           


# In[24]:


extExpGenic_byCell_1_ver01(output_path="../database/temp_files/expressed/byCellType/refFlat/", all_file=True, high_only=True, verbose=True, exp=0, high_exp=30)


# In[34]:


def extExpGenic_byCell_2_ver01(output_path="../database/temp_files/expressed/byCellType/",all_file=True, high_only=True, high_exp=50, verbose=True, **kwargs):
    """
    
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


# In[35]:


extExpGenic_byCell_2_ver01(output_path="../database/temp_files/expressed/byCellType/",all_file=False, high_only=True, high_exp=30, verbose=True, file="E003")


# In[37]:


extExpGenic_byCell_2_ver01(output_path="../database/temp_files/expressed/byCellType/",all_file=True, high_only=True, high_exp=20, verbose=True)


# In[38]:


extExpGenic_byCell_2_ver01(output_path="../database/temp_files/expressed/byCellType/",all_file=True, high_only=True, high_exp=10, verbose=True)


# In[40]:


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
    


# In[41]:


save_kmers_ver01(output_path="../database/pretrain/expressed/", high_exp=30, k=4,kind="highly_expressed")


# **Description for `prep_and_saveTF_ver01` function**
# 
# Modify the previous function `prep_and_saveTF` to `prep_and_saveTF_ver01`, to deal with the various RPKM cases for Ghexp. Previous function only distinguish cl1 anc cl2, which are respectively given as "expressed", "not_expressed", or "highly expressed", and the internal process of the function consider that there is only one file for each kind in the input path, which is `../database/temp_files/expressed/byCellType/`. However, now there are more than one file for highly_expressed, and the folder names indicate their different condition. 
# Therefore, in the new function, `incl_hexp` argument will be added to distinguish whether either cl1 and cl2 is highly_expressed case, and deal with the different name and process for that. 
# The output path also should be adjusted according to this kind of change.

# In[14]:


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


# In[15]:


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


# In[25]:


input_path="../database/temp_files/expressed/byCellType/"
output_path="../database/fine_tune/gene_exp/4mer_test/"
cl1="rpkm30_highly_expressed"
cl2="not_expressed"
epi_num_lst=["E003","E006","E016","E037","E050","E097","E098", "E112", "E122", "E128"]


# In[27]:


prep_and_saveTF_ver01(input_path, output_path, cl1, cl2, epi_num_lst, cut_thres=510, k=4, len_tr=40000, len_dev=1000)


# In[29]:


test_df=pd.read_csv("../database/fine_tune/gene_exp/4mer_test/tr_len_40k/train.tsv", sep="\t")
test_df["sequence"].isna().sum()
# test_df.head()


# In[ ]:





# In[58]:


input_path="../database/temp_files/expressed/byCellType/"
output_path="../database/fine_tune/gene_exp/4mer/Ghexp_rpkm10_or_not/"
cl1="rpkm10_highly_expressed"
cl2="not_expressed"
epi_num_lst=["E003","E006","E016","E037","E050","E097","E098", "E112", "E122", "E128"]


# In[59]:


prep_and_saveTF_ver01(input_path, output_path, cl1, cl2, epi_num_lst, cut_thres=510, k=4, len_tr=40000, len_dev=1000)


# In[60]:


input_path="../database/temp_files/expressed/byCellType/"
output_path="../database/fine_tune/gene_exp/4mer/Ghexp_rpkm20_or_not/"
cl1="rpkm20_highly_expressed"
cl2="not_expressed"
epi_num_lst=["E003","E006","E016","E037","E050","E097","E098", "E112", "E122", "E128"]
prep_and_saveTF_ver01(input_path, output_path, cl1, cl2, epi_num_lst, cut_thres=510, k=4, len_tr=40000, len_dev=1000)


# In[61]:


input_path="../database/temp_files/expressed/byCellType/"
output_path="../database/fine_tune/gene_exp/4mer/Ghexp_rpkm30_or_not/"
cl1="rpkm30_highly_expressed"
cl2="not_expressed"
epi_num_lst=["E003","E006","E016","E037","E050","E097","E098", "E112", "E122", "E128"]
prep_and_saveTF_ver01(input_path, output_path, cl1, cl2, epi_num_lst, cut_thres=510, k=4, len_tr=40000, len_dev=1000)


# In[39]:


epi_essential=["E003","E006","E016","E037","E050","E097","E098", "E112", "E122", "E128"]
total_epi_df=pd.read_csv("../database/bed/gene_expression/EG.name.txt", sep="\t", header=None)
total_epi=total_epi_df[0].dropna().to_list()
remaining_epi=[epi for epi in total_epi if epi not in epi_essential]

total_selection=20
epi_selection=random.sample(remaining_epi, total_selection-len(epi_essential))
epi_to_use=epi_selection+epi_essential


# In[ ]:


input_path="../database/temp_files/expressed/byCellType/"
output_path="../database/fine_tune/gene_exp/4mer/Ghexp_rpkm50_or_not/"
cl1="rpkm50_highly_expressed"
cl2="not_expressed"
prep_and_saveTF_ver01(input_path, output_path, cl1, cl2, epi_to_use, cut_thres=510, k=4, len_tr=40000, len_dev=1000)


# In[ ]:





# In[5]:


type(g_df_chr_collapsed_lst)


# In[127]:


# genic area for pretrainig test
# test for only one cell, and check the size of the dataset
with open("../database/temp_files/whole_gene_unit/E003_css_gene_unit_lst_all.pkl", "rb") as f:
    test_genic=pickle.load(f)


# In[134]:


test_genic=flatLst(test_genic)


# In[135]:


splitted_5, kmerized_unit_css_5=css_CUT_Kmer(test_genic,k=5)


# In[ ]:





# In[ ]:





# In[211]:


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
    


# In[212]:


save_kmers(k=4,kind="whole_gene")


# In[213]:


save_kmers(k=4,kind="highly_expressed")


# In[214]:


save_kmers(k=4,kind="expressed")


# In[215]:


save_kmers(k=4,kind="not_expressed")


# In[216]:


save_kmers(k=5,kind="whole_gene")


# In[217]:


save_kmers(k=5,kind="highly_expressed")


# In[218]:


save_kmers(k=5,kind="expressed")


# In[219]:


save_kmers(k=5,kind="not_expressed")


# In[ ]:





# In[ ]:





# In[175]:


epi_num_lst=pd.read_csv("../database/temp_files/whole_gene_unit/epigenome_lst.txt", header=None, names=["num"])


# In[ ]:





# In[ ]:





# In[140]:


with open("../database/test_E003_5mer.txt","wb") as g:
    pickle.dump(kmerized_unit_css_5,g)


# In[ ]:





# In[ ]:





# In[9]:


# Now the all genes (as css) are saved, then how to create the training and test dataset


# In[32]:


css_hexp_path="../database/temp_files/expressed/byCellType/highly_expressed/"
css_exp_path="../database/temp_files/expressed/byCellType/expressed/"
css_whole="../database/temp_files/whole_gene_unit/"


# In[47]:


# total_path=os.listdir(css_whole)
# total_path


# In[58]:


whole_gene_files=sorted(glob.glob(css_whole+"*.pkl"))
exp_gene_files=sorted(glob.glob(css_exp_path+"*.pkl"))

# extract whole gene list for different cell types
not_exp_gene_all=[]
for gene_file in whole_gene_files:
    pattern=r'E[0-9]+'
    epi_name=re.findall(pattern, gene_file)[0] # e.g.) 'E003'
    # take expressed gene list for the same cell type
    for file in exp_gene_files:
        if epi_name in file:
            exp_gene_file=file
    with open(gene_file,"rb") as f:
        whole_gene=flatLst(pickle.load(f))
    with open(exp_gene_file, "rb") as g:
        exp_gene=pickle.load(g)
    
    not_exp_gene=[]
    for gene in whole_gene:
        if gene in exp_gene:
            pass
        else:
            not_exp_gene.append(gene)
    not_exp_gene_all.append(not_exp_gene)

len(not_exp_gene_all)
        


# In[115]:


def extNOTexp_Geneic_byCell(output_path="../database/temp_files/expressed/byCellType/not_expressed/", all_file=True, verbose=True, **kwargs):
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

    


# In[116]:


extNOTexp_Geneic_byCell(output_path="../database/temp_files/expressed/byCellType/not_expressed/",file="E003_exp_gene_css.pkl")


# In[107]:


extNOTexp_Geneic_byCell(output_path="../database/temp_files/expressed/byCellType/not_expressed/",all_file=False,file="E003_exp_gene_css.pkl")


# In[121]:


test_path="../database/temp_files/expressed/byCellType/not_expressed/E004_not_exp_gene_css.pkl"
with open (test_path,"rb") as f:
    test=pickle.load(f)
len(test)


# In[125]:


test[140]


# In[34]:


css_hexp_path="../database/temp_files/expressed/byCellType/highly_expressed/"

css_hexp_count_lst=[]
for file in sorted(os.listdir(css_hexp_path)):
    with open(css_hexp_path+file,"rb") as f:
        css_hexp=pickle.load(f)
    css_hexp_count_lst.append(len(css_hexp))
    
print("Total count= {}".format(np.sum(css_hexp_count_lst)))
sns.histplot(css_hexp_count_lst, element="step", fill=False, color="orange")


# In[36]:


css_exp_path="../database/temp_files/expressed/byCellType/expressed/"

css_exp_count_lst=[]
for file in sorted(os.listdir(css_exp_path)):
    with open(css_exp_path+file,"rb") as f:gi
        css_exp=pickle.load(f)
    css_exp_count_lst.append(len(css_exp))
    
print("Total count= {}".format(np.sum(css_exp_count_lst)))
sns.histplot(css_exp_count_lst, element="step", fill=False, color="maroon")


# In[ ]:





# In[ ]:





# In[ ]:


with open(css_E122_hexp_path,"rb") as f:
    css_E122_hexp=pickle.load(f)

css_E122_hexp_len=[len(elm) for elm in css_E122_hexp]
sns.histplot(css_E122_hexp_len, log_scale=True, element="step", fill=False, color="orange", bins=15)
print("Total count= {}".format(len(css_E122_hexp_len)))
plt.xlabel("Length (x 200bps)", fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# ### Whole gene extraction
# 
# 1. Whole gene extraction for a cell (e.g. E112: HUVEC) from ROADMAP data
#     * Path: `"../database/bed/unzipped/E112_15_coreMarks_stateno.bed"`
#     * Function: `compGene2css(whole_gene_file, df)` where `df` comes from the function `bed2df_expanded`
# 2. Reduce to the unit length 
#     * Input: previous output
#     * Function: `Convert2unitCSS_main`

# In[418]:


whole_gene_file  # stored at css_utility.py


# In[417]:


df_e112=bed2df_expanded("../database/bed/unzipped/E112_15_coreMarks_stateno.bed")


# In[419]:


css_E112_gene_lst_all=compGene2css(whole_gene_file,df_e112)


# In[420]:


css_E112_gene_unit_lst_all=Convert2unitCSS_main(css_E112_gene_lst_all,unit=200)


# In[422]:


# Visualization: length distibution
css_E112_gene_unit_all=flatLst(css_E112_gene_unit_lst_all)
css_E112_gene_unit_all_len=[len(elm) for elm in css_E112_gene_unit_all]
sns.histplot(css_E112_gene_unit_all_len, log_scale=True, element="step", fill=False, color="royalblue", bins=15)
print("Total count= {}".format(len(css_E112_gene_unit_all_len)))
plt.xlabel("Length (x 200bps)", fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# In[8]:


css_E112_comp_path="../database/temp_files/complexity/thres_mean/byCellType/E112_comp_gene_css.pkl"
with open(css_E112_comp_path,"rb") as f:
    css_E112_comp=pickle.load(f)
type(css_E112_comp)


# In[13]:


css_E112_comp_len=[len(elm) for elm in css_E112_comp]
sns.histplot(css_E112_comp_len, log_scale=True, element="step", fill=False, color="yellowgreen", bins=15)
print("Total count= {}".format(len(css_E112_comp_len)))
plt.xlabel("Length (x 200bps)", fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# In[14]:


css_E112_exp_path="../database/temp_files/expressed/byCellType/expressed/E112_exp_gene_css.pkl"
with open(css_E112_exp_path,"rb") as f:
    css_E112_exp=pickle.load(f)
type(css_E112_exp)


# In[17]:


css_E112_exp_len=[len(elm) for elm in css_E112_exp]
sns.histplot(css_E112_exp_len, log_scale=True, element="step", fill=False, color="maroon", bins=15)
print("Total count= {}".format(len(css_E112_exp_len)))
plt.xlabel("Length (x 200bps)", fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# In[18]:


css_E112_hexp_path="../database/temp_files/expressed/byCellType/highly_expressed/E112_highly_exp_gene_css.pkl"
with open(css_E112_hexp_path,"rb") as f:
    css_E112_hexp=pickle.load(f)
type(css_E112_hexp)


# In[19]:


css_E112_hexp_len=[len(elm) for elm in css_E112_hexp]
sns.histplot(css_E112_hexp_len, log_scale=True, element="step", fill=False, color="orange", bins=15)
print("Total count= {}".format(len(css_E112_hexp_len)))
plt.xlabel("Length (x 200bps)", fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# In[20]:


css_E122_gene_path="../database/temp_files/whole_gene_unit/E122_css_gene_unit_lst_all.pkl"
with open(css_E122_gene_path,"rb") as f:
    css_E122_gene=pickle.load(f)
type(css_E122_gene)


# In[22]:


css_E122_gene_len=[len(elm) for elm in flatLst(css_E122_gene)]
sns.histplot(css_E122_gene_len, log_scale=True, element="step", fill=False, color="royalblue", bins=15)
print("Total count= {}".format(len(css_E122_gene_len)))
plt.xlabel("Length (x 200bps)", fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# In[23]:


css_E122_comp_path="../database/temp_files/complexity/thres_mean/byCellType/E122_comp_gene_css.pkl"
with open(css_E122_comp_path,"rb") as f:
    css_E122_comp=pickle.load(f)
css_E122_comp_len=[len(elm) for elm in css_E122_comp]
sns.histplot(css_E122_comp_len, log_scale=True, element="step", fill=False, color="yellowgreen", bins=15)
print("Total count= {}".format(len(css_E122_comp_len)))
plt.xlabel("Length (x 200bps)", fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# In[25]:


css_E122_exp_path="../database/temp_files/expressed/byCellType/expressed/E122_exp_gene_css.pkl"
with open(css_E122_exp_path,"rb") as f:
    css_E122_exp=pickle.load(f)
css_E122_exp_len=[len(elm) for elm in css_E122_exp]
sns.histplot(css_E122_exp_len, log_scale=True, element="step", fill=False, color="maroon", bins=15)
print("Total count= {}".format(len(css_E122_exp_len)))
plt.xlabel("Length (x 200bps)", fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# In[26]:


css_E122_hexp_path="../database/temp_files/expressed/byCellType/highly_expressed/E122_highly_exp_gene_css.pkl"
with open(css_E122_hexp_path,"rb") as f:
    css_E122_hexp=pickle.load(f)

css_E122_hexp_len=[len(elm) for elm in css_E122_hexp]
sns.histplot(css_E122_hexp_len, log_scale=True, element="step", fill=False, color="orange", bins=15)
print("Total count= {}".format(len(css_E122_hexp_len)))
plt.xlabel("Length (x 200bps)", fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# In[442]:


epi_name_path="../database/bed/gene_expression/EG.name.txt"
epi_name_df=pd.read_csv(epi_name_path,sep="\t",header=None,index_col=False)
epi_num=epi_name_df[0].dropna().to_list()


# In[448]:


bed_file_path="../database/bed/unzipped/"
bed_file_lst=sorted(os.listdir(bed_file_path))
len(epi_name)


# In[449]:


len(bed_file_lst)


# In[450]:


epi_target_tuple=[(num, bed_file) for num in epi_num for bed_file in bed_file_lst if num in bed_file]
epi_target=[tup[1] for tup in epi_target_tuple]
len(epi_target)


# In[475]:


bed_file_path="../database/bed/unzipped/"
epi_name_path="../database/bed/gene_expression/EG.name.txt"

epi_name_df=pd.read_csv(epi_name_path, names=["epi_num","epi_name"], sep="\t", header=None, index_col=False)
epi_num=epi_name_df["epi_num"].dropna().to_list() # number, 0th field
epi_name=epi_name_df["epi_name"].dropna().to_list() # name, 1st field
bed_file_lst=sorted(os.listdir(bed_file_path))


# In[476]:


epi_target_tuple=[(num, bed_file) for num in epi_num for bed_file in bed_file_lst if num in bed_file]
epi_target=[tup[1] for tup in epi_target_tuple]


# In[481]:


epi_name_df["epi_name"][epi_name_df["epi_num"]=="E003"][1]


# In[544]:


# Save the whole gene area of the 57 epigenomes, in CSS unit sequences
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


# In[545]:


extGenic_byCell()


# In[ ]:





# ### Complex gene extraction
# 
# 1. Based on the CSS extracted for genic region, use `extract_complex_css(gene_css_all, thres="mean")` to build the function for all the epigenome.
# 2. Whole gene CSS for each epigenomes are saved at `../database/temp_files/whole_gene_unit`

# In[579]:


# Save the complex and less complex genic area of the 57 epigenomes, in CSS unit sequences
# Following function has been already executed, and pickled at "../database/temp_files/complexity/thres_mean/byCellType/"

def extCompGenic_byCell(output_path="../database/temp_files/complexity/", thres="mean", all_file=True, verbose=True, **kwargs):
    """
    This function extract CSS complex and less-complex genic region, according to the threshold.
    (1) To process all the .pkl file in ../database/temp_files/whole_gene_unit/, set 'all_file=True'.
        If you want to process only one file at a time, set e.g.) all_file=False, file=E003_css_gene_unit_lst_all.pkl
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

# extract_complex_css(gene_css_all, thres="mean")


# In[578]:


extCompGenic_byCell()


# In[ ]:





# In[ ]:





# In[658]:


def extExpGenic_byCell_1(output_path="../database/temp_files/expressed/byCellType/refFlat/", all_file=True, verbose=True, exp=0, high_exp=50, **kwargs):
    """
    RUN THE SECOND function 'extExpGenic_byCell_2' after running this function.
    This function extract CSS expressed genic region, mainly for "expressed" and "highly-expressed"
    (1) To process all the  ... set 'all_file=True'.
        If you want to process only one file at a time, set e.g.) all_file=False, file="E050_15_coreMarks_stateno.bed"
    (2) Current version is only for expressed/ highly expressed cases.
    (3) Outputs are e.g.) "E112_gene_expressed.refFlat", "E112_gene_highlyexpressed.refFlat" at output path
    """
    
    path="../database/bed/gene_expression/"
    script="classifygenes_ROADMAP_RPKM.py"
    epi_rpkm_tsv="57epigenomes.RPKM.pc.tsv"
    gene_ref="chr.gene.refFlat"
    original_path="~/Work/chromatin_state/NSP/"
    
    save_path="./byCellType/refFlat/"
#     css_bed_path="../database/bed/unzipped/"

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

            exp_file_name=save_path+epi_num+"_"+"gene_expressed.refFlat"
            hexp_file_name=save_path+epi_num+"_"+"gene_highlyexpressed.refFlat"
            get_ipython().run_line_magic('mv', '"gene_expressed.refFlat" {exp_file_name}')
            get_ipython().run_line_magic('mv', '"gene_highlyexpressed.refFlat" {hexp_file_name}')
            get_ipython().run_line_magic('cd', '-q {original_path}')
            ########################################################################################
            
    else:
        raise ValueError("Set all_file=True, or desginate any file name to proceed!")
    assert os.getcwd()[-3:]=="NSP", "Check the current working directory."
    
    return print("Results are stored at {}, and current working directory is : {}".format(output_path, os.getcwd()))
                           


# In[657]:


#test
extExpGenic_byCell_1(all_file=False,file="E122_15_coreMarks_stateno.bed")


# In[659]:


#main
extExpGenic_byCell_1()


# In[ ]:


css_gene_lst_all=comp_expGene2css("../database/bed/gene_expression/gene_highlyexpressed.refFlat",df_e050)
css_unit_lst_all=Convert2unitCSS_main(css_gene_lst_all, unit=200)
g_hexp_css_all=flatLst(css_unit_lst_all)


# In[5]:


def extExpGenic_byCell_2(output_path="../database/temp_files/expressed/byCellType/",all_file=True, verbose=True, **kwargs):
    """
    
    """
    ref_path="../database/bed/gene_expression/byCellType/refFlat/"
    ref_file_all=sorted(os.listdir(ref_path))
    ref_hexp_all=[elm for elm in ref_file_all if 'high' in elm and elm.startswith('E')]
    ref_exp_all=[elm for elm in ref_file_all if elm not in ref_hexp_all and elm.startswith('E')]
    
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
            hexp=ref_path+target_hexp_ref[0]
            exp=ref_path+target_exp_ref[0]
            
            css_hexp_gene_lst=comp_expGene2css(hexp,df)
            css_exp_gene_lst=comp_expGene2css(exp,df)
            css_hexp_gene_unit_lst=flatLst(Convert2unitCSS_main(css_hexp_gene_lst, unit=200))
            css_exp_gene_unit_lst=flatLst(Convert2unitCSS_main(css_exp_gene_lst, unit=200))

            with open(output_path+"highly_expressed/"+epi_num+"_highly_exp_gene_css.pkl","wb") as f:
                pickle.dump(css_hexp_gene_unit_lst,f)
            with open(output_path+"expressed/"+epi_num+"_exp_gene_css.pkl","wb") as g:
                pickle.dump(css_exp_gene_unit_lst,g)
            
#     elif len(kwargs)>0:
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
        hexp=ref_path+target_hexp_ref[0]
        exp=ref_path+target_exp_ref[0]

        css_hexp_gene_lst=comp_expGene2css(hexp,df)
        css_exp_gene_lst=comp_expGene2css(exp,df)
        css_hexp_gene_unit_lst=flatLst(Convert2unitCSS_main(css_hexp_gene_lst, unit=200))
        css_exp_gene_unit_lst=flatLst(Convert2unitCSS_main(css_exp_gene_lst, unit=200))

        with open(output_path+"highly_expressed/"+epi_num+"_highly_exp_gene_css.pkl","wb") as f:
            pickle.dump(css_hexp_gene_unit_lst,f)
        with open(output_path+"expressed/"+epi_num+"_exp_gene_css.pkl","wb") as g:
            pickle.dump(css_exp_gene_unit_lst,g)

    else:
        raise ValueError("Set all_file=True, or desginate any file name to proceed!")
    return


# In[ ]:





# In[ ]:





# In[6]:


extExpGenic_byCell_2(all_file=False,file="E122")


# In[7]:


extExpGenic_byCell_2()


# In[660]:


ref_path="../database/bed/gene_expression/byCellType/refFlat/"
ref_file_all=sorted(os.listdir(ref_path))


# In[665]:


ref_hexp_all=[elm for elm in ref_file_all if 'high' in elm and elm.startswith('E')]
ref_exp_all=[elm for elm in ref_file_all if elm not in ref_hexp_all and elm.startswith('E')]


# In[629]:


print(os.getcwd()[-3:]=="NSP")


# In[668]:


css_bed_path="../database/bed/unzipped/"
css_bed_file_all=sorted(os.listdir(css_bed_path))

target_bed=[elm for elm in css_bed_file_all if elm[:4]==epi_num]


# In[670]:


for epi_css in ref_hexp_all:
    epi_num=epi_css[:4]
    print(epi_num)
    target_bed=[elm for elm in css_bed_file_all if elm[:4]==epi_num]
    break


# In[673]:


target_bed[0]


# In[ ]:





# In[397]:


### First, whole gene for E050
# df_e050=bed2df_expanded("../database/bed/unzipped/E050_15_coreMarks_stateno.bed")
css_E050_gene_lst_all=compGene2css(whole_gene_file,df_e050)


# In[398]:


css_E050_gene_unit_lst_all=Convert2unitCSS_main(css_E050_gene_lst_all, unit=200)


# In[403]:


g_e050_css_all_len=[len(elm) for elm in flatLst(css_E050_gene_unit_lst_all)]


# In[404]:


sns.histplot(g_e050_css_all_len,log_scale=True, element="step", fill=False, color="royalblue", bins=15)
print("Total count= {}".format(len(g_e050_css_all_len)))
plt.xlabel("Length (x 200bps)", fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# In[ ]:


## next, complex gene for e050


# In[405]:


e050_comp_gene_css_all,e050_less_comp_gene_css_all=extract_complex_css(css_E050_gene_unit_lst_all, thres="mean")


# In[411]:


path_comp_e050="../database/temp_files/complexity/thres_mean/byCellType/E050/comp.pkl"


# In[412]:


with open(path_comp_e050, "wb") as f:
    pickle.dump(e050_comp_gene_css_all,f)


# In[413]:


path_less_e050="../database/temp_files/complexity/thres_mean/byCellType/E050/less_comp.pkl"


# In[414]:


with open(path_less_e050, "wb") as g:
    pickle.dump(e050_less_comp_gene_css_all,g)


# In[415]:


e050_comp_gene_css_all_len=[len(elm) for elm in e050_comp_gene_css_all]


# In[416]:


sns.histplot(e050_comp_gene_css_all_len,log_scale=True, element="step", fill=False, color="yellowgreen", bins=15)
print("Total count= {}".format(len(e050_comp_gene_css_all_len)))
plt.xlabel("Length (x 200bps)", fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# In[ ]:





# In[69]:


path="../database/bed/gene_expression/"


# In[200]:


g_hexp=pd.read_csv(path+"gene_highlyexpressed.refFlat", sep="\t", index_col=False, header=0)
# g_hexp.columns[0]="gene_id"
# g_hexp.index = g_hexp.iloc[:,0]
# g_hexp.index.rename("gene_id")

g_hexp=g_hexp.iloc[:,1:]
g_hexp.rename(columns={"name":"gene_id"}, inplace=True)
g_hexp.rename(columns={"#geneName":"geneName"}, inplace=True)
# g_exp.index = g_exp.iloc[:,0]

g_hexp.head()


# In[258]:


# function for preprocess the whole gene data and produce chromosome-wise gene lists
# each element is dataframe

def Gexp_Gene2GLChr(exp_gene_file='../database/bed/gene_expression/gene_highlyexpressed.refFlat'):
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
        assert g_df_temp["exonCount"][i]==len(int_lst) # make sure the no. element in exon start = count
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


# In[259]:


# highly expressed case
g_df_chr_collapsed_lst=Gexp_Gene2GLChr(exp_gene_file='../database/bed/gene_expression/gene_highlyexpressed.refFlat')


# In[376]:


# just expressed case
gexp_df_chr_collapsed_lst=Gexp_Gene2GLChr(exp_gene_file='../database/bed/gene_expression/gene_expressed.refFlat')


# In[381]:


# just expressed case - convert to css 
css_exp_gene_lst_all=comp_expGene2css("../database/bed/gene_expression/gene_expressed.refFlat",df_e050)
css_exp_unit_lst_all=Convert2unitCSS_main(css_exp_gene_lst_all, unit=200)


# In[382]:


g_exp_css_all=flatLst(css_exp_unit_lst_all)


# In[383]:


len(g_exp_css_all)


# In[384]:


g_exp_css_all_len=[len(elm) for elm in g_exp_css_all]


# In[389]:


sns.histplot(g_exp_css_all_len,log_scale=True, element="step", fill=False, color="maroon", bins=15)
print("Total count= {}".format(len(g_exp_css_all_len)))
plt.xlabel("Length (x 200bps)", fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# ### So now we have bed files for highly expressed.. then we need CSS for E050 

# In[260]:


df_e050=bed2df_expanded("../database/bed/unzipped/E050_15_coreMarks_stateno.bed")
df_e050.head()


# In[395]:


df_e050.tail()


# In[394]:


len(df_e050)


# In[262]:


all_unit_css=df2unitcss(df_e050)


# In[265]:


len(all_unit_css[0]) # unit length


# In[266]:


css_lst_chr=df2longcss(df_e050)


# In[268]:


len(css_lst_chr[0]) # original 


# In[289]:


def comp_expGene2css(exp_gene_file,df):   # df indicates css, created by bed2df_expanded
    """
    modified from `compGene2css`
    Input: Reference gene file, df (CSS)
    Output: list of chromosome-wise list that contains the css at genic area only.
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


# In[290]:


css_gene_lst_all=comp_expGene2css("../database/bed/gene_expression/gene_highlyexpressed.refFlat",df_e050)


# In[291]:


css_unit_lst_all=Convert2unitCSS_main(css_gene_lst_all, unit=200)


# In[ ]:





# In[296]:


len(css_unit_lst_all[0])


# In[293]:


g_hexp_css_all=flatLst(css_unit_lst_all)


# In[380]:


len(g_hexp_css_all)


# In[298]:


g_hexp_css_all[-1]


# In[299]:


g_hexp_css_len=[len(elm) for elm in g_hexp_css_all]


# In[333]:


len(g_hexp_css_len)


# In[373]:


sns.set(rc={"font.size":15, "font.family":"serif"})
sns.set_style("white")
print("Total count= {}".format(len(g_hexp_css_len)))
p=sns.histplot(g_hexp_css_len, log_scale=True, element="step", fill=False, color="orange", bins=15)
plt.xlabel("Length (x 200bps)", fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()
# p.set_xticks(p.get_xticks())
# p.set_xticklabels(p.get_xticks(), fontfamily="serif", fontsize=12)


# #### Is this length normal..? looks like too short. Let me compare it with complexity case

# In[361]:


with open("../database/temp_files/complexity/thres_mean/comp","rb") as f:
    comp_lst=pickle.load(f)


# In[364]:


len(comp_lst)


# In[365]:


comp_css_len=[len(elm) for elm in comp_lst]


# In[375]:


sns.histplot(comp_css_len,log_scale=True,element="step", fill=False, color="teal", bins=15)
print("Total count= {}".format(len(comp_css_len)))
plt.xlabel("Length (x 200bps)", fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# In[ ]:





# #### Convert string to int: exonStarts and exonEnds

# In[124]:


exon_start_test=[int(elm) for elm in g_hexp["exonStarts"][0].replace("[","").replace("]","").split(",")]


# In[125]:


exon_end_test=[int(elm) for elm in g_hexp["exonEnds"][0].replace("[","").replace("]","").split(",")]


# #### Exon is short

# In[128]:


np.array(exon_end_test)-np.array(exon_start_test)


# #### Tx is long

# In[136]:


print(np.mean(g_hexp["txEnd"]-g_hexp["txStart"]))
sns.histplot(g_hexp["txEnd"]-g_hexp["txStart"], bins=50, log_scale=True)


# In[ ]:





# In[90]:


g_hexp.sort_values("chrom", inplace=True)
g_hexp.head(20)


# In[94]:


g_hexp["chrom"].unique()


# In[ ]:





# In[13]:


epi57_raw=pd.read_csv(path+"57epigenomes.RPKM.pc.tsv", sep="\t",index_col=False, header=0)


# In[15]:


epi57_raw.head()


# In[14]:


epi57=epi57_raw


# In[6]:


epi57.head()


# In[8]:


epi57.index = epi57.iloc[:,0]


# In[10]:


epi57.head()


# In[11]:


epi57=epi57.iloc[:,1:]


# In[12]:


epi57.head()


# In[18]:


len(epi57)


# In[17]:


epi57.describe()


# In[47]:


epi57["gene_id"]


# In[ ]:




