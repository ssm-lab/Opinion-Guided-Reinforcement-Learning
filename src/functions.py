import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# human advice

def identify_states(n):
    """
    Returns lists of the states in the top row, bottom row, left column, and right column for a n * n environment
    """
    top_row = []
    for i in range(0, n):
        top_row.append(i)

    bottom_row = []
    for i in range(0, n):
        state = n ** 2 - n + i
        bottom_row.append(state)

    left_col = []
    for i in range(0, n):
        state = n * i
        left_col.append(state)

    right_col = []
    for i in range(0, n):
        state = n * i + (n - 1)
        right_col.append(state)

    return top_row, bottom_row, left_col, right_col

def get_pairs(state, n):
    """
    Returns a list of 2-tuples (state, action) of states and actions that directly lead to indicated state
    """
    top_row, bottom_row, left_col, right_col = identify_states(n)

    list = []
    
    if state == 0: # top left corner
        list.append((state + 1, 0)) # left action to state
        list.append((state + n, 3)) # up action to state
    elif state == (n - 1): #top right corner
        list.append((state - 1, 2)) # right action to state
        list.append((state + n, 3)) # up action to state
    elif state == (n * (n - 1)): #bottom left corner
        list.append((state + 1, 0)) # left action to state
        list.append((state - n, 1)) # down action to state 
    elif state == ((n ** 2) - 1): #bottom right corner
        list.append((state - 1, 2)) # right action to state
        list.append((state - n, 1)) # down action to state 
    elif state in left_col:
        list.append((state + 1, 0)) # left action to state
        list.append((state - n, 1)) # down action to state 
        list.append((state + n, 3)) # up action to state
    elif state in right_col:
        list.append((state - n, 1)) # down action to state 
        list.append((state - 1, 2)) # right action to state
        list.append((state + n, 3)) # up action to state
    elif state in top_row:
        list.append((state + 1, 0)) # left action to state
        list.append((state - 1, 2)) # right action to state
        list.append((state + n, 3)) # up action to state
    elif state in bottom_row:
        list.append((state + 1, 0)) # left action to state
        list.append((state - n, 1)) # down action to state 
        list.append((state - 1, 2)) # right action to state
    elif state in range(0, n*n):
        list.append((state + 1, 0)) # left action to state
        list.append((state - n, 1)) # down action to state 
        list.append((state - 1, 2)) # right action to state
        list.append((state + n, 3)) # up action to state
    else:
        raise Exception("State does not exist in the environment.")

    return list

def create_human_matrix(num_states, num_actions):
    """
    Reads opinion_tuples.txt file of human opinions to create a matrix of human opinions that is saved as as opinion.npy
    """
    n = int(np.sqrt(num_states))
    top_row, bottom_row, left_col, right_col = identify_states(n)

    # make template matrix
    opinion = np.zeros((num_states, num_actions), dtype = "f, f, f, f")
    for state in range(num_states):
        for action in range(num_actions):
            opinion[state, action] = (0, 0, 1, 1/num_actions)

    #read file
    file = open('src/files/opinion_tuples.txt', 'r')
    content = file.read()
    file.close()
    lines = content.splitlines()

    # u is first line 
    # b + d + u = 1, so based on user's uncertainty, we will split the remaining value (1 - u) between b and d based on their rating on belief/disbelief scale
    u = float(lines[0])

    for i in range(1, len(lines)):

        a = lines[i].split(',')
        state = int(a[0])

        # user is unsure, split remaining value evenly between b and d
        if (a[1] == ' +' or a[1]== ' -') and (int(a[2]) == 0):
            b = round((1 - u) /2, 2)
            d = round(b, 2)

        # user has high belief, set 100% of remaining value to b
        elif a[1] == ' +' and int(a[2]) == 2: 
            b = round(1 - u, 2)
            d = round(1 - (u + b), 2)

        # user has some belief, set b to 75% of remaining value 
        elif a[1] == ' +' and int(a[2]) == 1: 
            b = round(0.75 * (1 - u), 2)
            d = round(1 - (u + b), 2)

        # user has high disbelief, set d to 100% of remaining value 
        elif a[1] == ' -' and int(a[2]) == 2: 
            d = round(1 - u, 2)
            b = round(1 - (u + d), 2)

        # user has some disbelief, set d to 75% of remaining value 
        elif a[1] == ' -' and int(a[2]) == 1: 
            d = round(0.75 * (1 - u), 2)
            b = round(1 - (u + d), 2)

        num_states_num_actions = get_pairs(state, n)

        for pair in num_states_num_actions:
            curr_state = pair[0]
            curr_action = pair[1]
            opinion[curr_state, curr_action]= (b, d, u, 1/num_actions)

    opinion = np.save('src/files/opinion', opinion)

def theta_to_prob(theta):
    """
    Transforms matrix of numerical preferences to matrix of probabilities
    """
    rows = theta.shape[0]
    cols = theta.shape[1]
    probs = np.zeros((rows, cols))
    logits = np.zeros(cols)

    for state in range(rows):
        for action in range(cols):
            logit = np.exp(theta[state, action])
            logits[action] = logit
        probs[state] = logits / np.sum(logits)
    return probs

def policy_to_certanity(theta):
    """
    Transforms matrix of probabilities to matrix of certainties
    """
    probs = theta_to_prob(theta)
    rows = probs.shape[0]
    cols = probs.shape[1]
    certs = np.zeros((rows, cols), dtype = "f, f, f, f")
    
    for state in range(rows):
        for action in range(cols):
            p = probs[state, action]
            certs[state, action] = (p, 1-p, 0, p)
    return(certs)

def cert_to_policy(certs):
    """
    Transforms matrix of certainties to matrix of probabilities
    """
    rows = certs.shape[0]
    cols = certs.shape[1]
    probs = np.zeros((rows, cols))
    for state in range(rows):
        for action in range(cols):
            b = (certs[state, action])[0]
            u = (certs[state, action])[2]
            a = (certs[state, action])[3]
            probs[state, action] = b + a * u

    # normalize probability matrix so each row sums to 1
    probs = normalize(probs, axis=1, norm='l1')

    theta = np.zeros((rows, cols))

    for state in range(rows):
        mu = probs[state]
        log_sum = 0 
        for action in range(cols):
            log_sum += np.log(mu[action])
        c = (-1 / cols) * log_sum

        for action in range(cols):
            theta[state, action] = np.log(mu[action]) + c

    return theta

def beliefConstraintFusion(matrix1, matrix2):
    """
    Belief constraint fusion for two matrices with the same dimensions
    """
    rows = matrix1.shape[0]
    cols = matrix2.shape[1]
    fused = np.zeros((rows, cols), dtype = "f, f, f, f")
    for state in range(rows):
        for action in range(cols):
            b1 = (matrix1[state, action])[0]
            d1 = (matrix1[state, action])[1]
            u1 =  (matrix1[state, action])[2]
            a1 = (matrix1[state, action])[3]

            b2 = (matrix2[state, action])[0]
            d2 = (matrix2[state, action])[1]
            u2 =  (matrix2[state, action])[2]
            a2 = (matrix2[state, action])[3]

            harmony = b1*b2 + b1*u2 + b2*u1
            conflict = b1*d2 + b2*d1
            b = harmony / (1 - conflict)
            u = u1 * u2 / (1 - conflict)
            a = (a1 * (1 - u1) + a2 * (1 - u2)) / (2 - u1 - u2)
            d = 1 - (b + u)

            (fused[state, action])[0] = b
            (fused[state, action])[1] = d
            (fused[state, action])[2] = u
            (fused[state, action])[3] = a
    return fused

def human_advice(num_states, num_actions):
    # create initialized theta matrix of zeros
    policy_num = np.zeros((num_states, num_actions))

    # 1a/2b convert policy from agent into a matrix of certainties
    policy_cert = policy_to_certanity(policy_num)

    # 1b capture human opinion 
    create_human_matrix(num_states, num_actions)
    opinion = np.load('src/files/opinion.npy')

    # 2b fusion
    fused = beliefConstraintFusion(policy_cert, opinion)

    # 3 convert certainties to policy for agent 
    theta = cert_to_policy(fused)

    np.savetxt('src/files/human_advised_policy', theta, delimiter= ',')

# policy grad
    
def pi(state, theta, num_actions):
    logits = np.zeros(num_actions)
    for action in range(num_actions):
        logit = np.exp(theta[state, action])
        logits[action] = logit
        
    return logits / np.sum(logits)

def update_policy(ep_states, ep_actions, ep_probs, ep_returns, theta, alpha, num_actions):
    
    for t in range(0,len(ep_states)):
        state = ep_states[t]
        action = ep_actions[t]
        prob = ep_probs[t]
        action_return = ep_returns[t]

        phi = np.zeros([1, num_actions])
        phi[0, action] = 1
 
        score = phi - prob
        theta[state, :] = theta[state, :] + alpha * action_return * score

    return theta

def calc_return(rewards, gamma):
    # https://stackoverflow.com/questions/65233426/discount-reward-in-reinforce-deep-reinforcement-learning-algorithm
    ep_rewards = np.asarray(rewards)
    t_steps = np.arange(ep_rewards.size)
    ep_returns = ep_rewards * gamma**t_steps
    ep_returns = ep_returns[::-1].cumsum()[::-1] / gamma**t_steps
    return ep_returns.tolist()

def discrete_policy_grad(num_states, num_actions, map_size, MAX_EPISODES, slippery, alpha, gamma, advice):

    # train model using human advice if available
    if advice == True: 
        human_advice(num_states, num_actions)
        theta = np.loadtxt('src/files/human_advised_policy', delimiter=",")
    else: theta = np.zeros((num_states, num_actions))


    # train model using discrete policy grad
    env = gym.make('FrozenLake-v1', map_name = map_size, is_slippery = slippery)
    
    total_reward, total_successes = [], 0
    
    for episode in range(MAX_EPISODES):
        state = env.reset()[0]
        ep_states, ep_actions, ep_probs, ep_rewards, total_ep_rewards = [], [], [], [], 0
        terminated, truncated = False, False

        # gather trajectory
        while not terminated and not truncated:
            # add state to ep_states list
            ep_states.append(state)
            # pass state thru policy to get action_probs
            action_probs = pi(state, theta, num_actions)
            # add action probabilities to action_probs list
            ep_probs.append(action_probs)
            # choose an action
            action = np.random.choice(np.array([0, 1, 2, 3]), p = action_probs)
            # add action to ep_actions list
            ep_actions.append(action)
            # take step in environment
            state, reward, terminated, truncated, __ = env.step(action)
            # add reward to ep_rewards list
            ep_rewards.append(reward)
            total_ep_rewards += reward
            if reward == 1: total_successes += 1

        # calculate episode return & add total episode reward to totalReward
        ep_returns = calc_return(ep_rewards, gamma)
        total_reward.append(sum(ep_rewards))
            
        # update policy
        update_policy(ep_states, ep_actions, ep_probs, ep_returns, theta, alpha, num_actions)

    #np.savetxt('theta', theta, delimiter= ',')
    env.close()
    
    # evaluate model

    # success rate
    success_rate = (total_successes/MAX_EPISODES) * 100
    
    #print(f"Success rate after {MAX_EPISODES} episodes = {success_rate}%")

    # graph of cumulative reward
    cumulative_reward = np.cumsum(total_reward)
    #plt.plot(cumulative_reward)
    #plt.title('Cumulative reward per episode')
    #plt.xlabel('Episode')
    #plt.ylabel('Cumulative Reward')
    #plt.show()

    return success_rate

def evaluate(experiments, MAX_EPISODES, num_states, num_actions, map_size, slippery, alpha, gamma):
    no_advice_success_rates = []
    for i in range(experiments):
        iteration = discrete_policy_grad(num_states, num_actions, map_size, MAX_EPISODES, slippery, alpha, gamma, advice = False)
        no_advice_success_rates.append(iteration)

    advice_success_rates = []
    for i in range(experiments):
        iteration = discrete_policy_grad(num_states, num_actions, map_size, MAX_EPISODES, slippery, alpha, gamma, advice = True)
        advice_success_rates.append(iteration)

    plt.plot(no_advice_success_rates, label = 'No advice')
    plt.plot(advice_success_rates, label = 'Advice')
    plt.title('Training on ' + str(map_size) + ' map ' + 'for ' + str(MAX_EPISODES) + ' episodes ' + ', is_slippery = ' + str(slippery))
    plt.xlabel('Iteration')
    plt.ylabel('Success Rate %')
    plt.legend()
    plt.show()