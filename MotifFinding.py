#!/usr/bin/env python
# coding: utf-8

# In[19]:


import random 
import numpy as np 
from pyseqlogo.pyseqlogo import draw_logo, setup_axis
import pandas as pd 
import matplotlib as plt 

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import RMSprop,SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv1D,MaxPooling1D
from tensorflow.keras import regularizers 


class Fasta:
    """ 
    load Fasta file and change it into one hot encoding format 
    """
    def __init__(self,label=1,max_len=100):
        self.name = label
        self.fa = []
        self.encode = []
        self.label = []
        self.max_len = max_len
        
    def load_fasta(self,file,eq_win=True):
        f_handle = open(file,'r+')
        line = f_handle.readline()
        seq_l = []
        max_len = self.max_len 
        for line in f_handle: 
            if line.startswith(">"):
                pass 
            else:
                sequence = line.strip("\n")
                seq_l.append(sequence) 
                if len(sequence) > max_len:
                    max_len = len(sequence)
            line = f_handle.readline()
        for seq in seq_l:
            n = len(seq)
            if eq_win:
                self.fa.append(seq.upper() + ''.join(np.random.choice(['A', 'C', 'G', 'T'], max_len - n, p=np.array([1,1,1,1])/4.0)))
            else:
                self.fa.append(seq.upper())
        seq = []         
        
    def one_hot_encoding(self):
        base_dict = {'A': 0,'C':1,'G':2,'T':3}
        seq_code = np.zeros([len(self.fa),len(self.fa[0])] + [4],dtype=int)
       
        for i in range(len(self.fa)):
            for j in range(len(self.fa[0])):
                seq_code[i,j,base_dict[self.fa[i][j]]] = 1          
        self.encode = seq_code
        self.label = np.array([self.name for x in range(len(self.fa))])
        
    def merge_one_hot(self,other):
        self.encode = np.concatenate((self.encode,other.encode),axis=0)
        self.label = np.concatenate((self.label,other.label),axis = 0)
        
        

class Motif:
    motifs = [] 
    """
    create motif logo using pyseqlogo 
    """
    def __init__(self,motif='',ppm = []): 
        self.motif = motif 
        self.ppm = ppm 
        Motif.motifs.append(self)
        
    def logo(self):
        base = ['A','C','G','T']
        logo = []
        for i in range(len(self.ppm)):
            base_freq = []
            for j in range(len(self.ppm[i])):
                base_freq.append((base[j],self.ppm[i][j]))
            base_freq.sort(key = lambda x: x[1])
            logo.append(base_freq)
        plt.rcParams['figure.dpi'] = 300
        fig, axarr = draw_logo(logo, coordinate_type='data')
        fig.tight_layout()


        
    
        
        
class Gibbsampling:
    #########################################
    #### k, the length of the motif length. 
    #### N, iteration numbers to exit the PWM updating 
    #########################################
    
    def __init__(self,k=11,N=1000):
        self.k = k 
        self.N = N
        self.mat = []
        
    def profile(self,motif):
        """
        Calulate position probabilty matrix from selected k-mer DNA sequence and consensus matrix ß
        update self.ppm and self.consensus 
        """
        mat = np.array(self.mat)
        base = ['A','C','G','T']
        pwm_m = []
        concensus = ''
        for i in range(self.k):
            col_s = ''.join(x[i] for x in mat)
            count_base = [col_s.count('A'),col_s.count('C'),col_s.count('G'),col_s.count('T')]
            count_base = [x+1 for x in count_base]
            count_max = np.argmax(count_base)
            pwm = count_base / np.sum(count_base)
            concensus += base[count_max]
            pwm_m.append(pwm)
        
        motif.ppm = pwm_m
        motif.motif = concensus

    def updating_motif(self,fasta,motif):
        """
        self.N: maximal number of iterations 
        randomly selected one sequence from fasta file, score all k-mer in the selected sequence,
        choose one k-mer fragment depending on the scores. High score fragment has high probability to be selected. 
        """
        for j in range(self.N): 
            i = random.randint(0,len(fasta.fa)-1) 
            replace = self._samplingwithd(fasta.fa[i],motif)
            self.mat[i] = replace
            self.profile(motif)
            
    def _samplingwithd(self,dna,motif): 
        """ 
        calculated the probability to be selected for every k-mer of dna. 
        """
        prob = [] 
        for index in range(len(dna) - self.k + 1):
            kmer = dna[index: index + self.k]
            prob_subset = self._probability(kmer,motif)
            prob.append(prob_subset)
        
        i = random.choices(range(len(dna) - self.k + 1),weights=prob,k=1)
        subset = dna[i[0]:i[0]+self.k]
        return subset       

    def _probability(self,kmer,motif):
        base = ['A','C','G','T']
        res = 1
        for i in range(self.k):
            base_substr = kmer[i] 
            index = base.index(kmer[i])
            res *= motif.ppm[i][index]
        return res 
            
    

                      
        
    def sampling_k(self,fasta):
        """
        Random samping K-mer from the fasta file. Each fasta returns a k-mer fragment. 
        update the self.mat. 
        """
        matrix_l = []
        for seq in fasta.fa:
            index = random.randrange(len(seq) - self.k + 1) 
            subseq = seq[index: index + self.k]
            self.mat.append(subseq)

            
            
            
class Expectation_Max(Gibbsampling):    
    """ 
    Expectation and Maximization to find the motif. 
    it heriates the class of Gibbsampling, and heriates the methods, like sampling_k, profile functions. 
    """
    def __init__(self,k,N):
        super().__init__(k,N)
    
    def _expectation(self,dna,motif):
        """ 
        Calculate the position weighted matrix depending on the randomly seleted k-mer matrix sequence. 
        
        """
        base = ['A','C','G','T']
        pwm = motif.ppm  
        z = []
        index = []
        for i in range(len(dna) - self.k + 1):
            prob = 1 
            subseq = dna[i: i + self.k]
            tmp = []
            for j in range(self.k) : 
                prob *= pwm[j][base.index(subseq[j])] 
                tmp.append(base.index(subseq[j])) 
            z.append(prob)
            index.append(tmp) 
            
        z = z/sum(z)  
        index = np.array(index).transpose()
        ppm = np.zeros((self.k,4))
        for i in range(self.k):
            for j in range(len(dna) - self.k + 1):
                ppm[i][index[i][j]] += z[j]       
        return  ppm 
    
    
               
    def maximization(self,fasta,motif):
        """
        input: Fasta class and motif class. 
        it inheritate Gibbsampling class. like self.N: Miximum # of iterations. 
        """
        base = ['A','C','G','T']
        for j in range(self.N):
            ppm = np.zeros((self.k,4))
            z_sum = 0 
            for i,seq in enumerate(fasta.fa):
                ppm_dna = self._expectation(seq,motif)
                ppm += ppm_dna     
            ppm = (ppm + 1)/(len(fasta.fa) + 4)
            motif.ppm = ppm 
        motif.ppm = np.around(motif.ppm, decimals=2)
        
        # update motif concensus string. 
        motif_str = ''
        for i in range(len(motif.ppm)):
            l1 = np.array(motif.ppm[i])
            motif_str += base[np.argmax(l1)]
        motif.motif = motif_str
        
        
class CNN:

    """ 
    Convolution Neural Network methods to do the motif finding.  
    """
    def __init__(self,filter_num = 2,filter_len=10,filter_strides = 1, pool_size=13,pool_strides = 13):
        self.filter_num = filter_num 
        self.filter_len = filter_len
        self.filter_strides = 1 
        self.pool_size = pool_size
        self.pool_strides = pool_strides

    def learning(self,fasta): 
        model = Sequential()
        model.add(Conv1D(self.filter_num,self.filter_len, padding='same',input_shape=(fasta.encode[0].shape), activation='relu',strides = 1 ))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Dropout(0.7))
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])
        model.fit(fasta.encode, fasta.label, validation_split=0.2, epochs=10)
        return model
    
    
    def feature_extract(self,model):
        base = ['A','C','G','T']
        for layer in model.layers:
            if 'conv' not in layer.name:
                continue
            # get filter weights
            filters, biases = layer.get_weights() 
            
            motif = ['' for x in range(filters.shape[2])]
            motif_pwm = [[] for x in range(filters.shape[2])]
            for i in range(filters.shape[0]):
                t = filters[i].transpose()
                
                for j in range(len(t)):
                    f_min, f_max = t[j].min(), t[j].max()
                    t[j] = (t[j] - f_min) / (f_max - f_min)
                    normalized_t  = t[j]/sum(t[j])
                    motif_pwm[j].append(normalized_t)
                    motif[j] += (base[np.argmax(t[j])])
            
            for i in range(len(motif_pwm)):
                Motif(motif[i],motif_pwm[i])
            
            return Motif.motifs 
           
class HMM: 
    def __init__(self,trans,em_prob,states,em_alphbet,start_states,end_states):
        self.trans = trans 
        self.ems = em_prob 
        self.states = states 
        self.em_alphbet = em_alphbet 
        self.start_states = start_states
        self.end_states = end_states 
    
    def _forward_prob(self,seq): 
        """
        function f_t(i+1) = e_t(y[i+1])* sum{f_s(i)*a_s_t} 
        log(f_t(i+1)) = log(e_t(y[i+1])) + log(sum(f_s(i)*a_s_t))
        """
        f_prob = np.zeros((len(self.states),len(seq)))
        for j in self.states: 
            for i,ch in enumerate(seq):
                if i == 0: 
                    f_prob[j,i] = np.log(self.start_states[j]) + np.log(self.ems[j,self.em_alphbet[ch]])
                else: 
                    state_j_prob = [f_prob[st,i-1] + np.log(self.trans[st,j]) for st in (self.states)]
                    f_prob[j,i] = logsumexp(state_j_prob) + np.log(self.ems[j,self.em_alphbet[ch]]) 
     
        end_state_j = list(map(lambda x,y:x+y, f_prob[:,-1],np.log(self.end_states)))

        prob_y = logsumexp(end_state_j) 
                    
        return f_prob, prob_y
    
    def _backward_prob(self,seq): 
        """
        function b_t(i-1) = sum{a_t_s*e_s(y[i])*b_s(i)}
        log(b_t(i-1)) = log(sum{a_t_s*e_s(y[i])*b_s(i)})
        """
        b_prob = np.zeros((len(self.states),len(seq)))
        for j in self.states: 
            for i in range(-1,-(len(seq)+1),-1): 
                if i == -1:
                    b_prob[j,i] = np.log(self.end_states[j]) + np.log(self.ems[j,self.em_alphbet[seq[i]]])

                else:                    
                    state_j_prob = [np.log(self.ems[st,self.em_alphbet[seq[i]]]) + np.log(self.trans[j,st]) +  b_prob[st,i+1]  for st in self.states]
                    b_prob[j,i] = logsumexp(state_j_prob)

        return b_prob   
    
    def _xi_prob(self,f_mat,b_mat,prob_y,seq): 
        """ 
        calcuate probability that ith element in S state and i + 1 th element in t states. called Xi_prob and updates the transition matrix for one sequence  
        function : p(s_i,t_i+1) = f_s(i) * b_t(i+1) * trans(s,t) * ems(t,seq[i+1]).  
        """
        
        Xi_prob = np.zeros((len(self.states),len(self.states),len(seq) - 1))
        trans = np.zeros((len(self.states),len(self.states)))
    
        for s in self.states:
            for t in self.states: 
                for i in range(len(seq) - 1): 
                    Xi_prob[s,t,i] = f_mat[s,i] + b_mat[t,i+1] + np.log(self.trans[s,t]) + np.log(self.ems[t,self.em_alphbet[seq[i+1]]]) - prob_y 
                
                trans[s,t] = logsumexp(Xi_prob[s,t,:]) 
                
                row_sum = np.sum(np.exp(trans),axis= 1) 
                trans = trans / row_sum[:,np.newaxis]
               
        return trans  
    
                   
    def _gamma_prob(self,f_mat,b_mat,prob_y,seq):
        """ 
        gamma_prob, also called si probability, indicates the probability that at index i the probability is s. 
        
        """

        em = np.zeros((len(self.states),len(self.em_alphbet.keys())))
        
        for s in self.states: 
            for sym,index in self.em_alphbet.items(): 
                tmp_sum = 0 
                for i,ch in enumerate(seq): 
                    if sym == ch: 
                        tmp = f_mat[s,i] + b_mat[s,i]
                        tmp_sum = np.logaddexp(tmp_sum,tmp)
                em[s,index] = tmp_sum - prob_y  
        return em
                          
        
    def bw_training(self,train_seqs,iterations = 5): 
        trans = np.zeros((len(self.states),len(self.states)))
        em = np.zeros((len(self.states),len(self.em_alphbet.keys())))
        
        for _ in range(iterations): 
            for seq in train_seqs: 
                f_mat,prob_y = self._forward_prob(seq)
                b_mat = self._backward_prob(seq) 
                seq_trans = self._xi_prob(f_mat,b_mat,prob_y,seq)
                
                seq_em = self._gamma_prob(f_mat,b_mat,prob_y,seq)

                #add trans among all sequence 
                for s1 in self.states:
                    for s2 in self.states: 
                        trans[s1,s2] = np.logaddexp(trans[s1,s2],seq_trans[s1,s2]) 

                #add emssion among all sequence 
                for s in self.states:
                    for sym,index in self.em_alphbet.items(): 
                        em[s,index] = np.logaddexp(em[s,index],seq_em[s,index]) 

            
            trans = np.exp(trans) 
            trans_sum = trans.sum(axis=1)
            trans = trans/trans_sum[:,np.newaxis]

            em = np.exp(em)
            em_sum = em.sum(axis=1)
            em = em/em_sum[:,np.newaxis]
        
            self.trans = trans 
            self.em = em 
        print(trans)
        print(em)

 
        
            
    
#### gibbs sampling methods to do the motif finding 
    
# fa = Fasta()
# fa.load_fasta("./fasta/positiv.fasta")
# gibb = Gibbsampling(18,1000)
# gibb.sampling_k(fa)
# motif = Motif()
# gibb.profile(motif)
# gibb.updating_motif(fa,motif)  
# print(motif.motif)
# for i in range(len(motif.ppm)):
#     print(motif.ppm[i])
# motif.logo()

#### CNN methods to do the motif finding 
pos = Fasta(label=1)
pos.load_fasta("./fasta/positiv.fasta")
pos.one_hot_encoding()
neg = Fasta(label=0)
neg.load_fasta("./fasta/neg.fasta")
neg.one_hot_encoding()
pos.merge_one_hot(neg) 
models = CNN(filter_num = 1,filter_len=18,filter_strides = 1, pool_size=13,pool_strides = 13)
model_res = models.learning(pos)
motifs = models.feature_extract(model_res)   
for ob in motifs:
    print(ob.motif) 
    for i in ob.ppm:
        print(i)
    ob.logo()

#### Eexpectation and Maximization  ####

# fa = Fasta()
# fa.load_fasta("./fasta/positiv.fasta")
# EM = Expectation_Max(30,10) 
# EM.sampling_k(fa)
# motif = Motif() 
# EM.profile(motif)
# EM.maximization(fa,motif)
# print(motif.motif)
# print(motif.ppm)
# motif.logo()







            

# motif_len = 10 

# #initial a trans matrix 
# tmp = (1-(0.99+0.009))/(motif_len -1) 
# trans_mat = np.full((1+motif_len,1+motif_len),tmp) 
# trans_mat[0,:2] = [0.99,0.009]
# np.fill_diagonal(trans_mat[1:motif_len+1,2:motif_len+1],0.999)
# trans_mat[motif_len,:2] = [0.99,0.009]

# ## initial a emit matrix 
# em_prob = np.full((1+motif_len,4),0.25) 
# for i in range(0,1+motif_len):
#     index = random.randint(0,3)
#     em = [0.31 if j == index else 0.23 for j in range(4)]
#     em_prob[i] = em
# # for i in range(1,1+motif_len,4):
# #     index = 2
# #     em = [0.999 if j == index else 0.001 for j in range(4)]
# #     em_prob[i] = em
# # for i in range(2,1+motif_len,4):
# #     index = 2
# #     em = [0.999 if j == index else 0.001 for j in range(4)]
# #     em_prob[i] = em
# # for i in range(3,1+motif_len,4):
# #     index = 0
# #     em = [0.999 if j == index else 0.001 for j in range(4)]
# #     em_prob[i] = em
# # for i in range(4,1+motif_len,4):
# #     index = 0
# #     em = [0.999 if j == index else 0.001 for j in range(4)]
# #     em_prob[i] = em
    
# em_alphbet = {'A':0,'C':1,'G':2,'T':3}
# states = [x for x in range(0,1+motif_len)]

# ###initial a start matrix, we do not update start matrix during training 
# motif_states = [0.01 for x in range(motif_len-1)]
# sum_motif = 0.1 * (motif_len - 1)
# start_bg_p = 0.9* (1-sum_motif)
# first_base_start_p = 0.01*(1-sum_motif)
# start_states = [start_bg_p,first_base_start_p] + motif_states

# ### initial a end matrix, we do not update end matrix during training 
# end_motif = [0.01 for x in range(motif_len -1)]
# end_motif_last  = 0.1*(1-sum_motif)
# end_state_bg = 0.9*(1-sum_motif) 
# end_states = [end_state_bg] + end_motif + [end_motif_last]


# fa = Fasta()
# fa.load_fasta("./fasta/gata.fasta",eq_win = False)

# hmm1 = HMM(trans_mat,em_prob,states,em_alphbet,start_states,end_states)
# hmm1.bw_training(fa.fa,iterations=10)




# In[ ]:




