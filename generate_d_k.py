import numpy as np
import time

''' Generate the percent correct and percent agreement models.
Autor: Bashar Awwad Shiekh Hasan
basharawwad.sh@gmail.com
'''
#start with a radom seed
seed = np.int(time.time().as_integer_ratio()[0]/10**8)

np.random.seed(seed)

no_trials=100
no_simulations=100

ext_noise_mean=0
in_noise_mean=0

sd=1



dprime_values=np.linspace(0,10,51)
inoise_values=np.linspace(0,10,51)

choice_values=list(range(10,1,-1))

percent_correct = np.zeros((9,len(dprime_values),len(inoise_values)))
percent_agreement = np.zeros((9,len(dprime_values),len(inoise_values)))
print(percent_correct.shape)

for afc in range(10,1,-1):
    print("simulating " + str(afc) + " classes")
    for dprime_pointer in range(len(dprime_values)):
        dprime= dprime_values[dprime_pointer]
        for inoise_pointer in range(len(inoise_values)):
            inoise=inoise_values[inoise_pointer]
            percent_correct_temp=[]
            percent_agreement_temp=[]
            for simulation in range(no_simulations):
                #generate external noise for all trials
                noise_set = np.random.normal(ext_noise_mean, sd, (afc, no_trials))
                #add signal to the first stimulus
                noise_set[0,]=noise_set[0,]+dprime
                #replicate external noise for double pass
                noise_set= np.hstack((noise_set,noise_set))
                #add internal noise source
                noise_set=noise_set+np.random.normal(size=(noise_set.shape))
                #compute percent correct
                response = np.argmax(noise_set,axis=0)==0 # the signal is always in the first stimulus
                percent_correct_temp.append(np.mean(response))
                #compute percent agreement
                percent_agreement_temp.append(np.mean(response[range(no_trials)]==response[range(no_trials,2*no_trials)]))
            percent_correct[afc-2, dprime_pointer, inoise_pointer] = np.mean(percent_correct_temp)
            #print(np.mean(percent_correct_temp))
            percent_agreement[afc-2, dprime_pointer, inoise_pointer] = np.mean(percent_agreement_temp)
            #print(np.mean(percent_agreement_temp))

np.savez("pc_pa_record",pc=percent_correct,pa=percent_agreement)