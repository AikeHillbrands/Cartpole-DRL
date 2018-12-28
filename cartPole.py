import gym
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import keras
from keras.models import load_model


episode_len = 500
reward_dropoff = 0.0 #A reward dropoff, also known as discount factor wasnt usefull for the cartpole game. I guess it was pushing the reward function to high due to the fact that the agent is getting a reward every time step
learning_rate = 0.01 #the lr of 0.1 worked the best for the current settings. The lr isnt used for the gradient decent algorithm but for the Q-Value function

env = gym.make('CartPole-v1')
env.reset()

def actions_to_action_lists(actions):                                                                           #making an one hot array from the decimal based actions that are accepted in the cartpole game
    result = []
    for action in actions:
        if action == 0:
            result.append([1,0])
        else:
            result.append([0,1])
    return result

def build_model(inp_size,out_size):                                                                             #Model is build
    model = keras.Sequential()
    model.add(keras.layers.Dense(128,input_dim=inp_size, activation = "relu"))
    model.add(keras.layers.Dense(64,activation = "relu"))
    model.add(keras.layers.Dense(out_size,activation = "relu"))
    model.compile(loss="mse",optimizer=keras.optimizers.Adam())
    return model

def import_model():                                                                                             #Importing the model from a file
    return load_model('model.h5')

def play_episods_random(count_games,req_score):                                                                 #for playing the first episode with random actions and giving a good base to train on to the ai
    states = []
    actions = []
    for games in range(count_games):
        env.reset()
        game_states = []
        game_actions = []
        game_score = 0
        prev_state = []
        for step in range(episode_len):
            action = rnd.randrange(0,2)
            if len(prev_state) > 0:
                game_states.append(prev_state)
                game_actions.append(action)
            
            prev_state, reward, done, info = env.step(action)                                                   #Doing the step and recording the Values
            game_score+=reward
            if done:
                break
        if game_score >= req_score:                                                                             #the played and recorded game is checked if it has a higher score than the required score to save it and later fit the model to the recorded data
            states = states + game_states
            actions = actions + actions_to_action_lists(game_actions)

    states =  np.array(states).reshape((len(states),4))                                                         #The final state and action data is reshaped in shapes the model can use
    actions = np.array(actions).reshape((len(actions),2))
    

    return states, actions
   

def train_model_qlearning(model,episodes=40,generations = 3,episode_len = 2000):                                #the algorithm that trains the model on its own played games. Works like q-learning
    scores = []
    for generation in range(generations):                                                                       #each generation is seperately trained on its own data. The data is collected in the episodes
        gen_score = []
        print("Generation:"+str(generation) +"/"+ str(generations))
        train_states = []
        train_actions = []
        for episode in range(episodes):                                                                         #a singel game is an episode
            env.reset()
            states = []
            actions = []
            rewards = []
            prev_state, reward, done, info = env.step(0)
            for step in range(episode_len):                                                                     #the game gets played and the data's beeing recorded
                action = model.predict(np.array([prev_state]))
                action = add_noise(action,1/(10**generation))
                actions.append(action)

                states.append(prev_state)
                action = np.argmax(action)
                prev_state, reward, done, info = env.step(action)
                rewards.append(reward)                                                                          #the reward is saved in a list to be processed later with the reward function for the new q-values

                if done:                                                                                        #When the agent loses the score gets tracked for statistics
                    scores.append(step)
                    gen_score.append(step)
                    #rewards[-1]=-3                                                                             #losing is punished. Turned out to not be that influential on the problem. It rather blocks success with bad scores
                    break       
            

            y = calculate_q_values(actions,rewards)                                                             #q-values beeing calculated based on the actions and rewards of the whole episode (played game)          
            for s in y:
                train_actions.append(s)                                                                         #the states and q-values (actions with q-value calculated) are added to lists to be later converted to np.arrays
            for s in states:
                train_states.append(s)
        
        print("Generation mean score:"+str(np.mean(gen_score)))
        train_states=np.array(train_states)                                                                     #the states and actions beeing transformed in np.arrays and also reshaped to fit the nn
        train_actions = np.array(train_actions).reshape((len(train_actions),2))

        train_model(model,train_states,train_actions)                                                           #the model is simply trained using backpropergation by keras
    plt.plot(scores)
    plt.show()

def add_noise(array,intensity):
    noise = (np.random.random(array.shape)-0.5)*intensity
    return array+noise

def calculate_rewards(rewards):
    for i in reversed(range(len(rewards)-1)):
        rewards[i] += rewards[i+1] * reward_dropoff #the list of rewards is calculated form back to front. the reward from t+1 is multiplyed with a dropoff and added to the reward of t
    
        
def calculate_q_values(actions,rewards): #function that calculates the q-values from the actions and rewards
    """calculate_rewards(rewards)
    for i in range(len(actions)):
        actions[i] = (1-learning_rate) * actions[i] + learning_rate*(rewards[i]*one_hot(actions[i])) #this is not the official q-value function but its working well
    return actions""" #<--This q value function was working but the network was unstable so i implemented the official Bellman equation
    actions[-1][0][np.argmax(actions[-1][0])]=rewards[-1]
    for i in reversed(range(len(actions)-2)):
        actions[i][0][np.argmax(actions[i][0])]=actions[i][0][np.argmax(actions[i][0])] * (1-learning_rate)   +  learning_rate * (rewards[i] + reward_dropoff * np.max(actions[i+1][0]))
    return actions

def one_hot(array): #the output array of the nn is not one hot so I'm converting it to one hot but with the biggest value instead of one for the hot values
    i = np.argmax(array)
    result = np.zeros(array.shape)
    result[0,i]=array[0,i]
    return result
    
    

def train_model(model,x,y):
    model.fit(x=x,y=y,batch_size = 500,epochs = 10,verbose=0)

def test_model(model): #function to test the model with render on
    scores = []
    for _ in range(10):
        env.reset()
        
        state, reward, done, info = env.step(0)
        for i in range(2000):
            if i % 2 == 0:
                env.render()
            action = model.predict(np.array([state]))
            action = np.argmax(action)
            state, reward, done, info = env.step(action)
            if done:
                scores.append(i)
                break
        print(sum(scores)/len(scores))
    

def build_and_pretrain_model(): #creates the model and pretrains it with random episodes to give a good start to the ai
    states,actions = play_episods_random(100,60)
    model = build_model(4,2)
    train_model(model,states,actions)
    model.save('model.h5')
    return model

#model = build_and_pretrain_model() #the pre training of the model is important because the pre training acts like the exploration which is important to find. Use this line if you want to train the model from the beginning
model = import_model()
#model = build_model(4,2)

#train_model_qlearning(model,generations=20,episodes=30)

test_model(model)