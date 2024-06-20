import numpy as np
import math

def markov_states(transformed_values):
    arithmetic_mean = arithmetic_mean(transformed_values)
    standard_deviation = standard_deviation(arithmetic_mean, transformed_values)
    states = states_estimation(transformed_values, arithmetic_mean, standard_deviation)

    return states

def markov_transition_matrix(states, transformed_values):
    transition_matrix = get_transition_matrix(states, transformed_values)
    return transition_matrix

def markov_cumulative_transition_matrix(transition_matrix):
    cumulative_transition_matrix = cumulative_transition_matrix(transition_matrix)

    return cumulative_transition_matrix
    
def markov_static_states(transformed_values, number_states=10):
    min_value = min(transformed_values)
    max_value = max(transformed_values)
        
    interval = (max_value - min_value)/(number_states - 1)
        
    states = []
        
    for i in range(number_states):
        states.append(min_value + interval*i)
            
    states.append(max_value)
            
    return states

def arithmetic_mean(m):
    add = 0

    for i in range(len(m)):
        add = add + m[i]

    return add / (len(m))

def standard_deviation(mean, m):
    add = 0

    for i in range(len(m)):
        add = add + math.pow(abs(m[i] - mean), 2)

    return math.sqrt(add / (len(m)))

def states_estimation(m, mean, standard):

    states = []
    i = 0

    while (mean + i * standard) < max(m):
        states.append(mean + i * standard)

        i = i + 1

    states.append(mean + i * standard)

    i = 0

    while (mean - i * standard) > min(m):
        states.append(mean - i * standard)

        i = i + 1

    states.append(min(m))

    states = list(set(states))
    states.sort()

    return states
    
def get_state_for_instance(instance, states):
    for i in range(0, len(states)):
        if states[i] <= instance <= states[i + 1]:
            return i

def get_transition_matrix(t, m):
    p = np.zeros((len(t) - 1, len(t) - 1))
    e_group = np.zeros((len(t) - 1))
    e = []

    for i in m:
        indice = np.searchsorted(t, i, side='right') - 1
        group = indice if indice < len(t) - 1 else len(t) - 2
        e.append(group)

    for i in range(len(e) - 1):
        p[e[i]][e[i+1]] = p[e[i]][e[i+1]] + 1
        e_group[e[i]] = e_group[e[i]] + 1

    for i in range(len(t) - 1):
        for j in range(len(t) - 1):
            if e_group[i] != 0:
                p[i][j] = p[i][j] / e_group[i]

    return p
    
def cumulative_transition_matrix(transition_matrix):
    cumulative_matrix = np.ones((transition_matrix.shape))

    for i in range(len(cumulative_matrix)):
        cumulative_value = 0    
        j = 0

        while cumulative_value < 1.0 and j < len(cumulative_matrix):
            if transition_matrix[i][j] != 0:
                cumulative_value = cumulative_value + transition_matrix[i][j]

            cumulative_matrix[i][j] =  cumulative_value                    
            j = j +1

    return cumulative_matrix
        
def predict_value(cumulative, states, current_value):
    state = np.searchsorted(states, current_value, side='right')
    state[np.where(state < len(states))] = state[np.where(state < len(states))] - 1
    state[np.where(state >= len(states))] = state[np.where(state >= len(states))] - 2
        

    random_value = np.random.uniform(0, 1, size=current_value.shape[0])

    future_state = np.zeros((state.shape), dtype=np.int)

    for i in np.unique(state):
        current_index = np.where(state == i)
            
        current_state = np.searchsorted(cumulative[i, :], random_value[current_index], side='right')
        current_state[np.where(current_state >= (len(states) - 1))] = len(states) - 2
            
        future_state[current_index] = current_state

    v_l = np.take(states, future_state)
    v_r = np.take(states, future_state + 1)
        
    return v_l + random_value * (v_r - v_l)

def compute_MTM(data, transition_matrix, states):
    shape = len(data)

    mtf_image = np.zeros((shape, shape))

    for i in range(1, shape):
        for j in range(1, shape):
            state_i = get_state_for_instance(data[i], states)
            state_j = get_state_for_instance(data[j], states)

            mtf_image[i, j] = transition_matrix[state_i, state_j]
                
    return mtf_image
    
def compute_transition(data, states=None):
    if states is not None:
        states = markov_static_states(data, number_states = states)
    else:
        mean = arithmetic_mean(data)
        std = standard_deviation(mean, data)

        states = states_estimation(data, mean, std)

    return markov_transition_matrix(states, data), states

def get_MTM(data, states=None):
    transition_matrix, states = compute_transition(data, states)
    
    return compute_MTM(data, transition_matrix, states)

def get_MTM_images(data, states, channels=1):
    images = []
    f_states = []

    for i in range(data.shape[0]):
        sequence = data.iloc[i, :]
        current_computation = compute_transition(sequence, states=states)
        images.append(current_computation[0])
        f_states.append(current_computation[1])

    images = np.asarray(images)
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], channels)
        
    return images, np.asarray(f_states)

def get_predictions(images, states, sequences, n_iterations=100):
    '''
    Get the predictions of the Markov model.

    Parameters
    ----------
        images : numpy.ndarray. The images of the data.
        states : numpy.ndarray. The states of the data.
        sequences : numpy.ndarray. The sequences of the data.
        n_iterations : int. The number of iterations for the predictions.
    
    Returns
    -------
        y_pred : numpy.ndarray. The predictions of the Markov model.
        y_min : numpy.ndarray. The minimum predictions of the Markov model.
        y_max : numpy.ndarray. The maximum predictions of the Markov model.
    '''
    y_pred = np.zeros((sequences.shape[0], sequences.shape[1]))
    y_min = np.zeros((sequences.shape[0], sequences.shape[1]))
    y_max = np.zeros((sequences.shape[0], sequences.shape[1]))
        
    for i in range(sequences.shape[0]):
        predictions = np.zeros((n_iterations, sequences.shape[1]))
        cumulative_matrix = cumulative_transition_matrix(images[i, :, :, 0])
        
        for j in range(n_iterations):
            predictions[j, :] = predict_value(cumulative_matrix, states[i], sequences[i])
            
        y_pred[i, :] = np.mean(predictions, axis=0)
    
        y_min[i, :] = np.mean(predictions, axis=0) + 3*np.std(predictions, axis=0)
        y_max[i, :] = np.mean(predictions, axis=0) - 3*np.std(predictions, axis=0)

    return np.asarray(y_pred), np.asarray(y_min), np.asarray(y_max)