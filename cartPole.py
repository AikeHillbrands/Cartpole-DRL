import gym
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import keras
from keras.models import load_model


episode_len = 500
#A reward dropoff, also known as discount factor wasnt usefull for the cartpole game.
#I guess it was pushing the reward function to high due to the fact that the agent is getting a reward every time step
reward_dropoff = 0.0
#The lr of 0.1 worked the best for the current settings.
#The lr isnt used for the gradient decent algorithm but for the Q-Value function
learning_rate = 0.01

env = gym.make('CartPole-v1')
env.reset()

#Making an one hot array from the decimal based actions that are accepted in the cartpole game
def actions_to_action_lists(actions):                
    result = []
    for action in actions:
        if action == 0:
            result.append([1,0])
        else:
            result.append([0,1])
    return result

#Model is build
def build_model(inp_size,out_size):                                                                             
    model = keras.Sequential()
    model.add(keras.layers.Dense(128,input_dim=inp_size, activation = "relu"))
    model.add(keras.layers.Dense(64,activation = "relu"))
    model.add(keras.layers.Dense(out_size,activation = "relu"))
    model.compile(loss="mse",optimizer=keras.optimizers.Adam())
    return model

#Importing the model from a file
def import_model():
    return load_model('model.h5')

#For playing the first episode with random actions and giving a good base to train on to the ai
def play_episods_random(count_games,req_score):                                                                 
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
            
            #Doing the step and recording the Values
            prev_state, reward, done, info = env.step(action)                                                   
            game_score+=reward
            if done:
                break
        #The played and recorded game is checked if it has a higher score than the required score
        #to save it and later fit the model to the recorded data
        if game_score >= req_score:                                                                             
            states = states + game_states
            actions = actions + actions_to_action_lists(game_actions)

    #The final state and action data is reshaped in shapes the model can use
    states =  np.array(states).reshape((len(states),4))                                                         
    actions = np.array(actions).reshape((len(actions),2))
    

    return states, actions
   

#the algorithm that trains the model on its own played games. Works like q-learning
def train_model_qlearning(model,episodes=40,generations = 3,episode_len = 2000):
    scores = []
    #Each generation is seperately trained on its own data. The data is collected in the episodes
    for generation in range(generations):                                                  
        gen_score = []
        print("Generation:"+str(generation) +"/"+ str(generations))
        train_states = []
        train_actions = []
        #a singel game is an episode
        for episode in range(episodes):                                                                         
            env.reset()
            states = []
            actions = []
            rewards = []
            prev_state, reward, done, info = env.step(0)
            #the game gets played and the data's beeing recorded
            for step in range(episode_len):                                                                     
                action = model.predict(np.array([prev_state]))
                action = add_noise(action,1/(10**generation))
                actions.append(action)

                states.append(prev_state)
                action = np.argmax(action)
                prev_state, reward, done, info = env.step(action)
                
                #the reward is saved in a list to be processed later with the reward function for the new q-values
                rewards.append(reward)                                                                          

                #When the agent loses the score gets tracked for statistics
                if done:                                                                                       
                    scores.append(step)
                    gen_score.append(step)                                                                          
                    break       
            

            #q-values beeing calculated based on the actions and rewards of the whole episode (played game)          
            y = calculate_q_values(actions,rewards)  
            
            #the states and q-values (actions with q-value calculated) are added to lists to be later converted to np.arrays
            for s in y:
                train_actions.append(s)                                                                         
            for s in states:
                train_states.append(s)
        
        print("Generation mean score:"+str(np.mean(gen_score)))
        
        #the states and actions beeing transformed in np.arrays and also reshaped to fit the nn
        train_states=np.array(train_states)                                                                     
        train_actions = np.array(train_actions).reshape((len(train_actions),2))

        #the model is simply trained using backpropergation by keras
        train_model(model,train_states,train_actions)                                                           
    plt.plot(scores)
    plt.show()

def add_noise(array,intensity):
    noise = (np.random.random(array.shape)-0.5)*intensity
    return array+noise

   
#function that calculates the q-values from the actions and rewards    
def calculate_q_values(actions,rewards): 
    actions[-1][0][np.argmax(actions[-1][0])]=rewards[-1]
    for i in reversed(range(len(actions)-2)):
        actions[i][0][np.argmax(actions[i][0])]=actions[i][0][np.argmax(actions[i][0])] * (1-learning_rate)   +  learning_rate * (rewards[i] + reward_dropoff * np.max(actions[i+1][0]))
    return actions

#the output array of the nn is not one hot so I'm converting it to one hot but with the biggest value instead of one for the hot values
def one_hot(array): 
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
    
    
#creates the model and pretrains it with random episodes to give a good start to the ai
def build_and_pretrain_model(): 
    states,actions = play_episods_random(100,60)
    model = build_model(4,2)
    train_model(model,states,actions)
    model.save('model.h5')
    return model

model = build_and_pretrain_model() #the pre training of the model is important because the pre training acts like the exploration which is important to find. 
#^Use this line if you want to train the model from the beginning in combination with the next line
train_model_qlearning(model,generations=20,episodes=30)

#model = import_model()



test_model(model)
