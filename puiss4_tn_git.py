from random import randint
import tensorflow as tf
import numpy as np
import scipy.signal
import random
from fct_perso import *

"""
    Exemple of the Policy Gradient Algorithm
"""

class Buffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(Buffer.combined_shape(size, obs_dim), dtype=np.float32)
        # Actions buffer
        self.act_buf = np.zeros(size, dtype=np.float32)
        # Advantages buffer
        self.adv_buf = np.zeros(size, dtype=np.float32)
        # Rewards buffer
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # Log probability of action a with the policy
        self.logp_buf = np.zeros(size, dtype=np.float32)
        # Gamma and lam to compute the advantage
        self.gamma, self.lam = gamma, lam
        # ptr: Position to insert the next tuple
        # path_start_idx Posittion of the current trajectory
        # max_size Max size of the buffer
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    @staticmethod
    def discount_cumsum(x, discount):
        """
            x = [x0, x1, x2]
            output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    @staticmethod
    def combined_shape(length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def store(self, obs, act, rew, logp):
        """
            Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        # Select the path
        path_slice = slice(self.path_start_idx, self.ptr)
        # Append the last_val to the trajectory
        rews = np.append(self.rew_buf[path_slice], last_val)
        # Advantage
        self.adv_buf[path_slice] = Buffer.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # Normalize the Advantage
        np_std=np.std(self.adv_buf)
        if np_std==0:
            np_std=1
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / np_std
        return self.obs_buf, self.act_buf, self.adv_buf, self.logp_buf


class PolicyGradient(object):
    """
        Implementation of Policy gradient algorithm
    """
    def __init__(self, input_space, action_space, pi_lr, buffer_size, seed):
        super(PolicyGradient, self).__init__()

        # Stored the spaces
        self.input_space = input_space
        self.action_space = action_space
        self.seed = seed
        # NET Buffer defined above
        self.buffer = Buffer(
            obs_dim=input_space,
            act_dim=action_space,
            size=buffer_size
        )
        # Learning rate of the policy network
        self.pi_lr = pi_lr
        # The tensorflow session (set later)
        self.sess = None
        # Apply a random seed on tensorflow and numpy
        tf.set_random_seed(42)
        np.random.seed(42)

    def compile(self):
        """
            Compile the model
        """
        # tf_map: Input: Input state
        # tf_adv: Input: Advantage
        self.tf_map, self.tf_a, self.tf_adv = PolicyGradient.inputs(
            map_space=self.input_space,
            action_space=self.action_space
        )
        # mu_op: Used to get the exploited prediction of the model
        # pi_op: Used to get the prediction of the model
        # logp_a_op: Used to get the log likelihood of taking action a with the current policy
        # logp_pi_op: Used to get the log likelihood of the predicted action @pi_op
        # log_std: Used to get the currently used log_std
        self.mu_op, self.pi_op, self.logp_a_op, self.logp_pi_op, self.spacial_action_logits = PolicyGradient.mlp(
            tf_map=self.tf_map,
            tf_a=self.tf_a,
            action_space=self.action_space,
            seed=self.seed
        )
        # Error
        self.pi_loss = PolicyGradient.net_objectives(
            tf_adv=self.tf_adv,
            logp_a_op=self.logp_a_op
        )
        # Optimization
        self.train_pi = tf.train.AdamOptimizer(learning_rate=self.pi_lr).minimize(self.pi_loss)
        # Entropy
        self.approx_ent = tf.reduce_mean(-self.logp_a_op)


    def set_sess(self, sess):
        # Set the tensorflow used to run this model
        self.sess = sess

    def step(self, states):
        # Take actions given the states
        # Return mu (policy without exploration), pi (policy with the current exploration) and
        # the log probability of the action chossen by pi
        mu, pi, logp_pi, s_a_l = self.sess.run([self.mu_op, self.pi_op, self.logp_pi_op, self.spacial_action_logits], feed_dict={
            self.tf_map: states
        })
        return mu, pi, logp_pi, s_a_l

    def store(self, obs, act, rew, logp):
        # Store the observation, action, reward and the log probability of the action
        # into the buffer
        self.buffer.store(obs, act, rew, logp)

    def finish_path(self, last_val=0):
        self.buffer.finish_path(last_val=last_val)

    def train(self, additional_infos={}):
        # Get buffer
        obs_buf, act_buf, adv_buf, logp_last_buf = self.buffer.get()
        # Train the model
        pi_loss_list = []
        entropy_list = []

        for step in range(5):
            _, entropy, pi_loss = self.sess.run([self.train_pi, self.approx_ent, self.pi_loss], feed_dict= {
                self.tf_map: obs_buf,
                self.tf_a:act_buf,
                self.tf_adv: adv_buf
            })

            pi_loss_list.append(pi_loss)
            entropy_list.append(entropy)

        print("Entropy : %s, Loss: %s" % (np.mean(entropy_list), np.mean(pi_loss_list)), end="\r")


    @staticmethod
    def gaussian_likelihood(x, mu, log_std):
        # Compute the gaussian likelihood of x with a normal gaussian distribution of mean @mu
        # and a std @log_std
        pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    @staticmethod
    def inputs(map_space, action_space):
        """
            @map_space Tuple of the space. Ex (size,)
            @action_space Tuple describing the action space. Ex (size,)
        """
        # Map of the game
        tf_map = tf.placeholder(tf.float32, shape=(None, *map_space), name="tf_map")
        # Possible actions (Should be two: x,y for the beacon game)
        tf_a = tf.placeholder(tf.int32, shape=(None,), name="tf_a")
        # Advantage
        tf_adv = tf.placeholder(tf.float32, shape=(None,), name="tf_adv")
        return tf_map, tf_a, tf_adv

    @staticmethod
    def mlp(tf_map, tf_a, action_space, seed=None):
        if seed is not None:
            tf.random.set_random_seed(seed)

        # Expand the dimension of the input
        tf_map_expand = tf.expand_dims(tf_map, axis=3)

        flatten = tf.layers.flatten(tf_map_expand)
        hidden1 = tf.layers.dense(flatten, units=256, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, units=128, activation=tf.nn.relu)
        hidden3 = tf.layers.dense(hidden2, units=64, activation=tf.nn.relu)
        spacial_action_logits = tf.layers.dense(hidden3, units=action_space, activation=None)

        # Add take the log of the softmax
        logp_all = tf.nn.log_softmax(spacial_action_logits)
        # Take random actions according to the logits (Exploration)
        pi_op = tf.squeeze(tf.multinomial(spacial_action_logits,1), axis=1)
        mu = tf.argmax(spacial_action_logits, axis=1)

        # Gives log probability, according to  the policy, of taking actions @a in states @x
        logp_a_op = tf.reduce_sum(tf.one_hot(tf_a, depth=action_space) * logp_all, axis=1)
        # Gives log probability, according to the policy, of the action sampled by pi.
        logp_pi_op = tf.reduce_sum(tf.one_hot(pi_op, depth=action_space) * logp_all, axis=1)

        return mu, pi_op, logp_a_op, logp_pi_op, spacial_action_logits

    @staticmethod
    def net_objectives(logp_a_op, tf_adv, clip_ratio=0.2):
        """
            @v_op: Predicted value function
            @tf_tv: Expected advantage
            @logp_a_op: Log likelihood of taking action under the current policy
            @tf_logp_old_pi: Log likelihood of the last policy
            @tf_adv: Advantage input
        """
        pi_loss = -tf.reduce_mean(logp_a_op*tf_adv)
        return pi_loss


# Fonctions définies pour faire fonctionner le jeu puissance 4

def afficher_grille(grille):
    print("----------------------")
    for i in range(len(grille)):
        print(list(int(x) if x==0.0 else int(2) if x==-1 else int(x) for x in grille[i]))
    print("----------------------")
    
def actualise_grille(grille,emplacement,joueur):
    grille_return=np.copy(grille)
    for i in range(len(grille_return)-1,-1,-1):
        if grille_return[i][emplacement]==0 :
            grille_return[i][emplacement]=joueur
            return grille_return

def position_dernier_jeton(grille,emplacement):
    for i in range(len(grille)):
        if grille[i][emplacement]!=0 :
            return i, emplacement
    return i, emplacement

def direction_horizontale(grille,i,j):
    joueur=grille[i][j]
    j_init=j
    n=1
    while j<np.shape(grille)[1]-1 and grille[i][j+1]==joueur:   #j<6
        j+=1
        n+=1
    j=j_init
    while j>0 and grille[i][j-1]==joueur:   #j>0
        j=j-1
        n+=1
    return True if n>=4 else False

def direction_verticale(grille,i,j):
    joueur=grille[i][j]
    i_init=i
    n=1
    while i<np.shape(grille)[0]-1 and grille[i+1][j]==joueur:   #i<5
        i+=1
        n+=1
    return True if n>=4 else False

def direction_oblique_descendante(grille,i,j):
    joueur=grille[i][j]
    i_init=i
    j_init=j
    n=1
    while i<np.shape(grille)[0]-1 and j<np.shape(grille)[1]-1 and grille[i+1][j+1]==joueur:     #i<5 and j<6
        j+=1
        i+=1
        n+=1
    i=i_init
    j=j_init
    while i>0 and j>0 and grille[i-1][j-1]==joueur:     #i>0 and j>0
        j=j-1
        i=i-1
        n+=1
    return True if n>=4 else False

def direction_oblique_montante(grille,i,j):
    joueur=grille[i][j]
    i_init=i
    j_init=j
    n=1
    while i>0 and j<np.shape(grille)[1]-1 and grille[i-1][j+1]==joueur:     #i>0 and j<6
        j+=1
        i=i-1
        n+=1
    i=i_init
    j=j_init
    while i<np.shape(grille)[0]-1 and j>0 and grille[i+1][j-1]==joueur:     #i<5 and j>0
        j=j-1
        i+=1
        n+=1
    return True if n>=4 else False

def liste_gagnante(grille,i,j):
    return [direction_horizontale(grille,i,j),
            direction_verticale(grille,i,j),
            direction_oblique_descendante(grille,i,j),
            direction_oblique_montante(grille,i,j)
            ]

def partie_finie(grille,emplacement):
    if list(grille[0]).count(0)==0 :
        return True     #partie terminée à égalitée
    else:
        i,j=position_dernier_jeton(grille,emplacement)
        if liste_gagnante(grille,i,j).count(True)>=1:
            return True     #partie terminée avec vainqueur
        return False    #partie pas terminée

def colonne_pleine(grille,emplacement):
    i,j=position_dernier_jeton(grille,emplacement)
    if i==0:
        return True
    else:
        return False

def vecteur_one_hot(tableau):
    liste_return=[]
    for i in range(len(tableau)):
        for val in tableau[i]:
            liste_return.append(val)
    return np.array(liste_return)

def somme_avantages(liste_rewards,gamma):
    liste_return=[]
    liste_rewards.reverse()
    for i in range(len(liste_rewards)):
        rewards=0
        liste_temp=list(liste_rewards[0:i+1])
        puissance=len(liste_temp)
        for recompense in liste_temp :
            puissance=puissance-1
            rewards=rewards+(gamma**puissance)*recompense
        liste_return.append(rewards)
    liste_return.reverse()
    return liste_return

def recompense(grille,emplacement,joueur):
    i,j=position_dernier_jeton(grille,emplacement)
    if liste_gagnante(grille,i,j).count(True)>=1:
        if grille[i][j]==1:
            r1=2    #récompense de victoire
            r2=-5   #récompense de défaite
        else:
            r1=-5
            r2=2
    elif list(grille[0]).count(0)==0 :
        r1=-0.05    #récompense en cas d'égalité => obtenu uniquement lorsque la grille est pleine et qu'il n'y a pas de gagnant
        r2=-0.05
    elif colonne_pleine(grille,emplacement)==True:
        if joueur==1:
            r1=-1000    #récompense lorsque le joueur joue sur une colonne entièrement pleine
            r2=0        #l'autre joueur n'est pas pénalisé
        else:
            r1=0
            r2=-1000
    return r1, r2

def init_grille():
    return  np.zeros((6,7))




def main():
    grille=init_grille()
    afficher_grille(grille)

    buffer_size = 1000

    # Create the NET class
    agent1 = PolicyGradient(
        input_space=(6, 7),
        action_space=7,
        pi_lr=0.001,
        buffer_size=buffer_size,
        seed=42
    )
    agent1.compile()
    agent2 = PolicyGradient(
        input_space=(6, 7),
        action_space=7,
        pi_lr=0.001,
        buffer_size=buffer_size,
        seed=42
    )
    agent2.compile()
    # Init Session
    sess = tf.Session()
    # Init variables
    sess.run(tf.global_variables_initializer())
    # Set the session
    agent1.set_sess(sess)
    agent2.set_sess(sess)

    rewards1 = []
    rewards2 = []
    b1 = 0
    b2 = 0
    for epoch in range(10000):

        done = False
        grille=init_grille()
        state = grille

        joueur=1
        tour=0
        while not done:
            if joueur==1:
                _, pi, logpi, s_a_l = agent1.step([state])
                pi1_prev=pi[0]
                logpi1_prev=logpi
                if b1==0:
                    state1=np.copy([state])
                else:
                    state1=np.vstack([state1,[state]])
            else:
                # ---- joueur random ----
                # if random.randint(1,10)<=5:             #0->j2 joue identique que j1        10->j2 joue random à 100%
                #     pi=[random.randint(0,6)]
                # else:
                #     pi=[pi1_prev]
                # ---- joueur random ----

                # ---- joueur IA2 ----
                _, pi, logpi, s_a_l = agent2.step([state])
                # ---- joueur IA2 ----
                
                pi2_prev=pi[0]
                logpi2_prev=logpi
                if b2==0:
                    state2=np.copy([state])
                else:
                    state2=np.vstack([state2,[state]])

            done=colonne_pleine(state,pi[0])
            if done==False:
                n_state=actualise_grille(state,pi[0],joueur)
                done=partie_finie(n_state,pi[0])
                # afficher_grille(n_state)
                if done==False and tour!=0:
                    if joueur==-1:
                        agent1.store(state1[-1],pi1_prev, 0, logpi1_prev)
                        b1 += 1
                        joueur=1
                    else:
                        agent2.store(state2[-1],pi2_prev, 0, logpi2_prev)
                        b2 += 1
                        joueur=-1
                elif tour==0:       #uniquement valable au 1er tour de chaque partie
                    if joueur==1:
                        joueur=-1
                    else:
                        joueur=1
            
            state = n_state

            if done:
                reward1, reward2=recompense(state,pi[0],joueur)
                agent1.store(state1[-1],pi1_prev, reward1, logpi1_prev)
                b1 += 1
                agent2.store(state2[-1],pi2_prev, reward2, logpi2_prev)
                b2 += 1
                agent1.finish_path(reward1)
                rewards1.append(reward1)
                agent2.finish_path(reward2)
                rewards2.append(reward2)
                if len(rewards1) > 1000:
                    rewards1.pop(0)
                if len(rewards2) > 1000:
                    rewards2.pop(0)
            if b1 == buffer_size:
                if not done:
                    agent1.finish_path(0)
                    rewards1.append(0)
                    rewards2.append(0)
                    done=True
                agent1.train()
                b1 = 0
            if b2 == buffer_size:
                if not done:
                    agent2.finish_path(0)
                    rewards1.append(0)
                    rewards2.append(0)
                    done=True
                agent2.train()
                b2 = 0
            tour+=1

        if (epoch+1) % 1000 == 0:
            print("")
            print("Rewards1 mean:%s" % np.mean(rewards1))
            print("Rewards2 mean:%s" % np.mean(rewards2))


    # pour tester le comportement des IA
    for epoch in range(3):

        done = False
        grille=init_grille()
        state = grille

        joueur=1
        while not done:
            if joueur==1:
                _, pi, logpi, s_a_l = agent1.step([state])
            else:
                # ---- joueur IA2 ----
                _, pi, logpi, s_a_l = agent2.step([state])
                # ---- joueur IA2 ----

                # ---- joueur random ----
                # plein=True
                # while plein:
                #     if random.randint(1,10)<=5:             #0->j2 joue identique que j1        10->j2 joue random à 100%
                #         pi=[random.randint(0,6)]
                #         plein=colonne_pleine(state,pi[0])
                #     else:
                #         pi=[pi1_prev]
                #         plein=False
                # ---- joueur random ----

            print("Joueur "+str(joueur)+" a joué à l'emplacement n°",pi[0])
            done=colonne_pleine(state,pi[0])
            if done==False:
                n_state=actualise_grille(state,pi[0],joueur)
                afficher_grille(n_state)
                if joueur==1:
                    joueur=-1
                else:
                    joueur=1
                done=partie_finie(n_state,pi[0])
            state=n_state


main()