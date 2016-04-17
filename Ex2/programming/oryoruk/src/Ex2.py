#!usr/bin/python

#NAME:  ONUR YORUK
#CLASS: GCB537 - SPRING 2016

#libraries:
import numpy as np
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from sklearn import metrics
from scipy import linalg


#constants:
PSSM_ROWS   = "ACGT"
DELTA       = 10**-4
INPUT_DIR   = '../ex1/'
OUTPUT_DIR  = '../output/'

#functions:

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
        msg = "ERROR, paramenter pair_values and/or labels are not correct. \n Please check function definition\n \
        @param pair_values: List of N elements. The ROC will have N ROC curves. \n \
                            Each element consist in 2 list ([R, P]) of M values, the first \n \
                            list are the true values, the second list the predicted\n \
                            values, this 2 list are paired element by element so \n \
                            the i-th position in each list represent the same sample.\n \
        @param labels: List of N strings representing the name/description of the ROC curves.\n"

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
    return seq_dict

#function to check if all probabilites add up to 1
def motif_probs_add_up_to_one(motif):
    motif_len = len(motif.itervalues().next())
    for i in range(motif_len):
        pos_total_prob = 0.0
        for nuc in motif:
            pos_total_prob += motif[nuc][i]
        if (pos_total_prob - 1.0)>DELTA: return False
    return True

def fix_zero_probs(motif):
    #print motif
    for nuc in motif:
        #add 10**-6 to zero probabilities for being able to take the logarithm
        fixed_zero_probs =\
                [prob + 10**-6 if prob<10**-5 else prob for prob in  motif[nuc] ]
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
    return pssm_dict

'''
def background_motif():
    bg_motif = {}
    for nuc in PSSM_ROWS:
        bg_motif[nuc]=[0.25]
    return bg_motif
'''


def subseq_motif_prob_log(sub_seq_array,motif_array,motif_len):
    motif_log_array = np.log(motif_array)
    log_prob = np.dot(sub_seq_array,motif_log_array)
    return np.trace(log_prob)

def float_to_str(floating_point, precision = 4):
    return (("%." + str(precision)+"f")%floating_point)

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






#inputs:
pssm_file_addresses = []
fasta_file_addresses = []

fasta_file_addresses.append(INPUT_DIR+'sequences.fa')

for i in range(4):
    pssm_file_addresses.append(INPUT_DIR+"motif"+str(i+1)+".pssm")


seq_dict = {}
for fasta_file_address in fasta_file_addresses:
    seq_dict.update(read_fasta(fasta_file_address))

#with open(INPUT_DIR + '../ex1/ex1.labels') as f:
#    for line in open
labels = {}
for line in open(INPUT_DIR + 'ex1.labels'):
    labels[line.split()[0]] = int(line.split()[1])



#for each motif
for pssm_file_address in pssm_file_addresses:
    motif = read_pssm(pssm_file_address)
    if motif_probs_add_up_to_one(motif):        
        motif_name = pssm_file_address.strip(INPUT_DIR)[:-5]
        fo = open(OUTPUT_DIR+ motif_name + ".output", "w")
        #for each sequence in fasta files
        output = ''
        totalLL = 0.0
        list_of_pos_reg_probs =[]
        list_of_max_bind_probs =[]
        list_of_log10_likelihoods = []
        list_of_true_labels = []
        #for each sequence
        for i, seq_name in enumerate(seq_dict):
            seq = seq_dict[seq_name]
            pos_reg_prob, max_bind_prob, log_likelihood = ZOOPS(seq, motif)
            if seq_name in labels:
                list_of_pos_reg_probs.append(pos_reg_prob)
                list_of_max_bind_probs.append(max_bind_prob)
                list_of_log10_likelihoods.append(log_likelihood/np.log(10))
                list_of_true_labels.append(labels[seq_name])
            #output:
            output_line = seq_name + '\t'
            output_line += float_to_str( pos_reg_prob ) + '\t'
            output_line += float_to_str( max_bind_prob ) + '\t'
            output_line += float_to_str(  log_likelihood/np.log(10) ) + '\n'
            #if i == 0: log_prob_sum = (bg_log_prob +  smpl)
            #else: log_prob_sum = np.logaddexp(log_prob_sum, bg_log_prob +  smpl)
            output += output_line
            if i==0: totalLL = log_likelihood
            else: totalLL = np.logaddexp(totalLL,log_likelihood)
        header_lines = "TotalLL: " + float_to_str( totalLL/np.log(10)) + '\n'
        header_lines += 'NAME\tP(R|S)\tMaxP(B|S)\tLL(S)\n'
        
        
        #ROC CURVES
        roc_curve_plot([  [list_of_true_labels,list_of_pos_reg_probs]  , [list_of_true_labels,list_of_max_bind_probs], [list_of_true_labels,list_of_log10_likelihoods] ],['pos_reg_probs','max_bind_prob','log10_likelihoods'],motif_name + " (TotalLL:"+str(totalLL/np.log(10))+")",OUTPUT_DIR+motif_name+"_roc_curves")
        pl.clf()
        
        fo.write( header_lines + output )
        fo.close()
    else: print('Motif probabilities are not adding up to 1.0')




