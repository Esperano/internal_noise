import numpy as np

'''
Autor: Bashar Awwad Shiekh Hasan
basharawwad.sh@gmail.com
'''

def calculate_PC_PA(first_pass_resp,second_pass_resp,first_pass_label,second_pass_label):
    ''' calcualtes the percent correct and percent agreement

    :param first_pass_resp: the responses in the first pass in a double-pass experiment
    :param second_pass_resp: the responses in the second pass
    :param first_pass_label: the ground truth of the first pass
    :param second_pass_label: the ground truth of the second pass
    :return:
    '''

    all_response = np.hstack((first_pass_resp,second_pass_resp))

    pc = np.mean(all_response==np.hstack((first_pass_label,second_pass_label)))
    pa = np.mean(first_pass_resp==second_pass_resp)
    return pc,pa

def calculate_dp_in(pc,pa,pc_matrix,pa_matrix,num_classes,dprime_pars,inoise_pars):
    ''' calculate d_prime and the standard deviation of the internal noise using the simulated model.

    :param pc: percent correct
    :param pa: percent agreement
    :param pc_matrix: the matrix of the percent correct model
    :param pa_matrix: the matrix of the percent agreement model
    :param num_classes: the total number of AFCs
    :param dprime_pars: the d_prime simulated values
    :param inoise_pars: the internal noise simulated values
    :return:
    '''
    dk = (np.squeeze(pc_matrix[num_classes-2,:,:])-pc)**2+(np.squeeze(pa_matrix[num_classes-2,:,:])-pa)**2
    x,y=np.where(dk==np.min(dk))
    d_prime = (y[0] / dk.shape[1]) * (dprime_pars[1] - dprime_pars[0])
    internal_noise = (x[0] / dk.shape[0]) * (inoise_pars[1] - inoise_pars[0])
    return d_prime,internal_noise

def dp_inoise(first_pass_resp,second_pass_resp,first_pass_label,second_pass_label):
    '''
    a utility function to ease the call of the models
    :param first_pass_resp: the responses in the first pass in a double-pass experiment
    :param second_pass_resp: the responses in the second pass
    :param first_pass_label: the ground truth of the first pass
    :param second_pass_label: the ground truth of the second pass
    :return:
    '''
    # load the internal noise matrixes
    npzfile = np.load("pc_pa_record.npz")

    pc_matrix = npzfile['pc']
    pa_matrix = npzfile['pa']
    dprime_pars = [0, 10, 0.2]
    inoise_pars = [0, 10, 0.2]

    num_classes = len(np.unique(first_pass_label))
    pc, pa = calculate_PC_PA(first_pass_resp,second_pass_resp,first_pass_label,second_pass_label)

    return calculate_dp_in(pc,pa,pc_matrix,pa_matrix,num_classes,dprime_pars,inoise_pars)