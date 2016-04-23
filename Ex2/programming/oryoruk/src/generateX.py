#!usr/bin/python

# NAME:  ONUR YORUK
# CLASS: GCB537 - SPRING 2016
# EXERCISE 2 PROGRAMMING

# coding: utf-8

## Ex2

### Libraries
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from sklearn import metrics


### Constants
#constants:
PSSM_ROWS   = "ACGT"
DELTA       = 10**-4
#these constanst have changed for Ex2:
INPUT_DIR   = '../Ex2Prog/'
OUTPUT_DIR  = '../output/'


### Functions for Ex0
#functions:
def read_fasta(fasta_file_address):
    seq_dict = {}
    fasta_file = open(fasta_file_address,"r")
    fasta_file_content = fasta_file.read()
    sequences = fasta_file_content.split('>')[1:]
    for sequence in sequences:
        seq_name = sequence.split('\n')[0]
        seq = ''.join( sequence.split('\n')[1:-1]).replace('\n','')
        seq_dict[seq_name] = seq
    fasta_file.close()
    return seq_dict

#function to check if all probabilites add up to 1
def motif_probs_add_up_to_one(motif):
    motif_len = len(motif.itervalues().next())
    for i in range(motif_len):
        pos_total_prob = 0.0
        for nuc in motif:
            pos_total_prob += motif[nuc][i]
        if (pos_total_prob - 1.0) > DELTA:
            return False
    return True

def fix_zero_probs(motif):
    #print motif
    for nuc in motif:
        #add 10**-6 to zero probabilities for being able to take the logarithm
        fixed_zero_probs =                [prob + 10**-6 if prob<10**-5 else prob for prob in  motif[nuc] ]
        #print fixed_zero_probs
        motif[nuc]=fixed_zero_probs
    #normalize probs to 1:
    motif_len = len(motif.itervalues().next())
    for i in range(motif_len):
        pos_sum = 0.0
        for nuc in motif:
            pos_sum+= motif[nuc][i]
        #print pos_sum
        if pos_sum!= 1.0:
            multiplier = 1.0/pos_sum
            for nuc in motif:
                motif[nuc][i] = (motif[nuc][i])*multiplier
    return motif

def read_pssm(pssm_file_address):
    pssm_dict = {}
    pssm_file = open(pssm_file_address,"r")
    for row_index, row in enumerate(pssm_file):
        pssm_dict[PSSM_ROWS[row_index]] = [float(prob) for prob in row.split()]
    pssm_dict = fix_zero_probs(pssm_dict)
    pssm_file.close()
    return pssm_dict   
    
def subseq_motif_log10_prob(subseq, motif, motif_len):
    log10_prob = 0.0
    for i in range(motif_len):
        log10_prob += np.log10(motif[subseq[i]][i])
        #print motif[subseq[i]][i]
    return log10_prob

def seq_motif_log10_probs(seq, motif):
    log10_probs = []
    motif_len = len(motif.itervalues().next())
    #for all potential windows, slide the window:
    for i in range(len(seq)-motif_len+1):
        #calculate log10 prob for window
        log10_prob = subseq_motif_log10_prob(seq[i:i+motif_len],motif,motif_len)
        log10_probs.append( log10_prob )
    return log10_probs


### New Functions for Ex1
def subseq_motif_prob_log(sub_seq_array,motif_array,motif_len):
    motif_log_array = np.log(motif_array)
    log_prob = np.dot(sub_seq_array,motif_log_array)
    return np.trace(log_prob)

def roc_curve_plot(pair_values, labels=[], title='ROC', out_filename='./roc'):
    """ 
        ROC plot generator functions 
        @param pair_values: List of N elements. The ROC will have N ROC curves. 
                            Each element consist in 2 list ([R, P]) of M values, the first 
                            list are the true values, the second list the predicted 
                            values, this 2 list are paired element by element so 
                            the i-th position in each list represent the same sample.
        @param labels: List of N strings representing the name/description of the ROC curves.
    @param title: String with the title of the ROC plot.
        @param out_filename: output filename will be out_filename.pdf

    """
    try:
        assert len(pair_values) == len(labels), "Different number of labels and ROC pairs"


        for ii, vals in enumerate(pair_values):
            real_vals, pred_vals = vals
            assert len(real_vals) == len(pred_vals), "Different number of samples in true values and predicted values"
            (fpr, tpr, thresholds) = metrics.roc_curve(real_vals, pred_vals, pos_label=1)
            auc_met = metrics.auc(fpr, tpr)
            pl.plot(fpr, tpr, label='%s - %2.2f%%' % (labels[ii], auc_met*100))

    except AssertionError as e:
        msg = "ERROR, paramenter pair_values and/or labels are not correct. \n Please check function definition\n         @param pair_values: List of N elements. The ROC will have N ROC curves. \n                             Each element consist in 2 list ([R, P]) of M values, the first \n                             list are the true values, the second list the predicted\n                             values, this 2 list are paired element by element so \n                             the i-th position in each list represent the same sample.\n         @param labels: List of N strings representing the name/description of the ROC curves.\n"

        print e.args
        print msg
        raise

    ''' Creating plot '''
    pl.plot([0, 1], [0, 1], 'k--', label='random')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title(title)
    pl.legend(loc="lower right")
    pl.savefig('%s.pdf' % out_filename)
    pl.savefig('%s.png' % out_filename)

'''
def background_motif():
    bg_motif = {}
    for nuc in PSSM_ROWS:
        bg_motif[nuc]=[0.25]
    return bg_motif
'''

def float_to_str(floating_point, precision = 4):
    return (("%." + str(precision)+"f")%floating_point)

"""
def ZOOPS(seq, motif, gamma = 0.1):
    seq_len = len(seq)
    motif_len = len(motif.itervalues().next())
    temp2 = []
    for nuc in PSSM_ROWS:
        temp2.append(motif[nuc])
    motif_array= np.array(temp2)

    #print seq_len, motif_len
    pos_reg_prob = 0.0
    max_bind_prob = 0.0
    likelihood = 0.0
    
    M = seq_len - motif_len + 1
    
    prob_sum = 0.0
    log_prob_sum = 0.0

    #max_site_prob = 0.0
    max_site_prob_log = -np.inf

    bg_log_prob = np.log(0.25)*(seq_len-motif_len)
    
    for i in range(M):        
        sub_seq = seq[i:i+motif_len]
        temp_list = []
        for nuc in sub_seq:
            temp_row = [0,0,0,0]
            temp_row[PSSM_ROWS.index(nuc)] = 1
            temp_list.append(temp_row)
        sub_seq_array = np.array(temp_list)
        smpl =subseq_motif_prob_log(sub_seq_array,motif_array,motif_len  ) 
        subseq_motif_prob = np.exp(smpl)
        #if subseq_motif_prob>max_site_prob:
        #    max_site_prob = subseq_motif_prob
        if smpl>max_site_prob_log:
            max_site_prob_log = smpl
            
        #prob_sum += np.exp(bg_log_prob +  smpl )
        if i == 0: log_prob_sum = (bg_log_prob +  smpl)
        else: log_prob_sum = np.logaddexp(log_prob_sum, bg_log_prob +  smpl)
    #print max_site_prob
        
    #prob_seq_if_reg = np.exp(log_prob_sum- np.log(M))
    #print prob_seq_if_reg
    #likelihood = prob_seq_if_reg *gamma + np.exp(np.log(0.25)*seq_len) *(1-gamma)
    #likelihood = np.exp(np.log(prob_seq_if_reg)+np.log(gamma)) + np.exp(np.log(0.25)*seq_len +np.log(1-gamma))
    log_likelihood = np.logaddexp ( ((log_prob_sum- np.log(M))+np.log(gamma)),(np.log(0.25)*seq_len +np.log(1-gamma)) )
    #print log_likelihood    
    
    #pos_reg_prob =np.exp(np.log(prob_seq_if_reg)+np.log(gamma)-log_likelihood)
    pos_reg_prob =np.exp(log_prob_sum- np.log(M)+np.log(gamma)-log_likelihood)

    
    #max_bind_prob = np.exp(((max_site_prob_log+bg_log_prob)-(np.log(M)+np.log(prob_seq_if_reg)))+np.log(pos_reg_prob))
    max_bind_prob = np.exp(((max_site_prob_log+bg_log_prob)-(np.log(M)+log_prob_sum- np.log(M)))+log_prob_sum- np.log(M)+np.log(gamma)-log_likelihood)


    return pos_reg_prob, max_bind_prob, log_likelihood
"""

### New Functions for Ex2
#functions:

#function to read in expression files
def read_exp_file(exp_file_address):
    exp_dict = {}
    exp_file = open(exp_file_address, "r")
    #first line is the header, so ignore it:
    exp_file.readline()
    #read the rest of the file into an expression dictionary
    for row_index, row in enumerate(exp_file):
        gene, val = row.split()
        exp_dict[gene] = float(val)
    exp_file.close()
    return exp_dict

#format for pssm files is different, hence new function 
def read_pssmlist(pssm_file_address):
    pssm_list = []
    pssm_file = open(pssm_file_address,"r")
    while(True):
        #read in motif name
        motif_name = pssm_file.readline()
        if motif_name == '': break
        #move to start of motif
        for i in range(4): pssm_file.readline()
        #read in motif
        pssm_dict = {}
        for i in range(4):
            row = pssm_file.readline()
            pssm_dict[PSSM_ROWS[i]] = [float(prob) for prob in row.split()]
        pssm_dict = fix_zero_probs(pssm_dict)
        pssm_list.append(pssm_dict) 
        #move to next motif
        for i in range(2): pssm_file.readline()
    pssm_file.close()
    return pssm_list   
    
#function to read background pssm file
def read_bg_pssm(pssm_file_address):
    pssm_dict = {}
    pssm_file = open(pssm_file_address,"r")
    pssm_file.readline()
    for row_index, row in enumerate(pssm_file):
        pssm_dict[PSSM_ROWS[row_index]] = [float(prob) for prob in row.split()]
    pssm_dict = fix_zero_probs(pssm_dict)
    pssm_file.close()
    return pssm_dict       

#format for sequences file is the same so I will reuse function from older exercises


### Variables for Ex2
pssmlist_file_address = INPUT_DIR + 'pssmlist'
fasta_file_addresses = []
fasta_file_addresses.append(INPUT_DIR+'sequence.fa')
seq_dict = {}
for fasta_file_address in fasta_file_addresses:
    seq_dict.update(read_fasta(fasta_file_address))
pssm_list = read_pssmlist(pssmlist_file_address)
bg_motif = read_bg_pssm(INPUT_DIR+'background_pssm.pssm')
exp_list = []
for i in range(4):
    exp_list.append( read_exp_file(INPUT_DIR+'Expression'+str(i+1)+'.tab') )

## Ex2 Starts Below
### Firstly, new ZOOPS that allows non-uniform background probabilities

def subseq_motif_prob_log(sub_seq_array,motif_array,motif_len):
    motif_log_array = np.log(motif_array)
    log_prob = np.dot(sub_seq_array,motif_log_array)
    return np.trace(log_prob)

def motif_to_array(motif, length = 1, background = False):
    motif_array = []
    if background==False: length = 1
    for nuc in PSSM_ROWS:
        motif_array.append(motif[nuc]*length)
    return np.array(motif_array)

def seq_to_array(seq):
    seq_array = []
    for nuc in seq:
        temp_row = [0,0,0,0]
        temp_row[PSSM_ROWS.index(nuc)] = 1
        seq_array.append(temp_row)
    seq_array = np.array(seq_array)
    return seq_array

def ZOOPS(seq, motif, bg_motif, gamma = 0.1):
    seq_len = len(seq)
    motif_len = len(motif.itervalues().next())
    motif_array = motif_to_array(motif)
    pos_reg_prob, log_prob_sum = 0.0, 0.0
    M = seq_len - motif_len + 1
    max_site_prob_log = -np.inf
    
    seq_array = seq_to_array(seq)
    bg_motif_array = motif_to_array(bg_motif, seq_len, background= True)   
    full_bg_log_prob =subseq_motif_prob_log(seq_array,bg_motif_array,seq_len)
        
    bg_motif_array = motif_to_array(bg_motif, motif_len, background= True)   
    
    for i in range(M):        
        sub_seq = seq[i:i+motif_len]
        sub_seq_array = seq_to_array(sub_seq)
        smpl = subseq_motif_prob_log(sub_seq_array,motif_array,motif_len  ) 
        subseq_motif_prob = np.exp(smpl)
        #if subseq_motif_prob>max_site_prob:
        #    max_site_prob = subseq_motif_prob
        if smpl>max_site_prob_log:
            max_site_prob_log = smpl
        motif_footprint_bg_log_prob = subseq_motif_prob_log(sub_seq_array,bg_motif_array,motif_len  )
        bg_log_prob = full_bg_log_prob - motif_footprint_bg_log_prob
        if i == 0: log_prob_sum = (bg_log_prob +  smpl)
        else: log_prob_sum = np.logaddexp(log_prob_sum, bg_log_prob +  smpl)
        
    log_likelihood = np.logaddexp ( ((log_prob_sum- np.log(M))+np.log(gamma)),(full_bg_log_prob +np.log(1-gamma)) )
    
    pos_reg_prob =np.exp(log_prob_sum- np.log(M)+np.log(gamma)-log_likelihood)

    return pos_reg_prob


### Then, construct X, load y's
#let's time this run:
then =  datetime.now()

#initialize X array and y vectors
n = len(seq_dict)
p = len(pssm_list)
exp_no = len(exp_list)
X = np.zeros((n, p), dtype=np.float)
y_array = np.zeros((n, exp_no), dtype=np.float)


#for each sequence, scan the sequence for each pssm, store probability in a matrix
#for each sequence
print "starting scanning sequences with given motifs to construct X"
for i, seq_id in enumerate(seq_dict):
    if i>1: break
    seq = seq_dict[seq_id]
    motif_probs = []
    #scan the sequence for each pssm    
    for j, motif in enumerate(pssm_list):
        prob = ZOOPS(seq_dict[seq_id],motif, bg_motif)
        X[i,j] = prob
    for j, exp in enumerate(exp_list):
        y_array[i,j] = exp[seq_id]      

now  =  datetime.now()
print round((now-then).total_seconds(), 2), "seconds passed"

y_array_filename = OUTPUT_DIR + 'experiments.pickle'
fileObject = open(y_array_filename, 'wb')
pickle.dump(y_array,fileObject)
fileObject.close()

X_filename = OUTPUT_DIR + 'probs_of_being_regulated.pickle'
fileObject = open(X_filename, 'wb')
pickle.dump(X, fileObject)
fileObject.close()

print 'pickles saved'