from keras.layers import Dense
from keras.models import Sequential
from keras import callbacks
import numpy as np
import gym
from easy_tf_log import tflog
from datetime import datetime
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

UP_ACTION = 2
DOWN_ACTION = 3
gamma = 0.99
x_train, y_train, rewards = [],[],[]
reward_sum = 0
episode_nb = 0
resume = True
running_reward = None
epochs_before_saving = 100
log_dir = './log' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
pre_trained_file = "my_model_weights20190507-214912.h5"

def preprocess(input):
  input = input[35:195] # crop
  input = input[::2, ::2, 0] # downsample by factor of 2
  input[input == 144] = 0 # erase background (background type 1)
  input[input == 109] = 0 # erase background (background type 2)
  input[input != 0] = 1 # everything else (paddles, ball) just set to 1
  return input.astype(np.float).ravel()

def discount_rewards(r, gamma):
  r = np.array(r)
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # if the game ended (in Pong), reset the reward sum
    running_add = running_add * gamma + r[t] # the point here is to use Horner's method to compute those rewards efficiently
    discounted_r[t] = running_add
  discounted_r -= np.mean(discounted_r) #normalizing the result
  discounted_r /= np.std(discounted_r) #idem
  return discounted_r

model = Sequential()
model.add(Dense(units=200,input_dim=80*80, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

env = gym.make("Pong-v0")
observation = env.reset()
prev_input = None

if (resume and os.path.isfile(pre_trained_file)):
    print("loading previous weights")
    model.load_weights(pre_trained_file)

tbCallBack = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                   write_graph=True, write_images=True)

while (True):
    cur_input = preprocess(observation)
    x = cur_input - prev_input if prev_input is not None else np.zeros(80 * 80)
    prev_input = cur_input

    proba = model.predict(np.expand_dims(x, axis=1).T)
    action = UP_ACTION if np.random.uniform() < proba else DOWN_ACTION
    y = 1 if action == 2 else 0  # 0 and 1 are our labels

    x_train.append(x)
    y_train.append(y)

    observation, reward, done, info = env.step(action)
    rewards.append(reward)
    reward_sum += reward

    if done:
        print('At the end of episode', episode_nb, 'the total reward was :', reward_sum)
        episode_nb += 1
        model.fit(x=np.vstack(x_train), y=np.vstack(y_train), verbose=1, callbacks=[tbCallBack],
                  sample_weight=discount_rewards(rewards, gamma))

        if episode_nb % epochs_before_saving == 0:
            model_file = 'my_model_weights' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5'
            model.save_weights()
            uploaded = drive.CreateFile({'title': 'my_model_weights' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5'})
            uploaded.SetContentFile(model_file)
            uploaded.Upload()
            print('Uploaded file with ID {}'.format(uploaded.get('id')))

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        tflog('running_reward', running_reward, custom_dir=log_dir)

        x_train, y_train, rewards = [], [], []
        observation = env.reset()
        reward_sum = 0
        prev_input = None