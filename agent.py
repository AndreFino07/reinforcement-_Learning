import torch
import random
import numpy as np
from collections import deque # serve per memorizzare
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
import os

MAX_MEMORY = 100000 # numeo massimo memoria di elementi
BATCH_SIZE = 1000 #numero di "ricordi" che l'IA ripassa ogni volta
LR = 0.0005 #per capire di quanto Torch deve cambiare i "pesi" della rete neurale 

class Agent:
    def __init__(self):
        self.n_games = 0 # per vedere numero parttiìte fatte
        self.epsilon =0 #parametro pe conntrollare la casualita
        self.gamma = 0.95 # tasso di sconto
        self.memory = deque(maxlen=MAX_MEMORY)#se supera MAX allora cancella memoria prima
        self.model = Linear_QNet(11,512,3) #primo dimensione stato, 3 perche output sono 3 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        if os.path.exists("./model/model.pth"):
            self.model.load_state_dict(torch.load("./model/model.pth"))
            self.model.eval() # Importante per dire al modello che stiamo usando i pesi salvati
            print("Pesi caricati! Il serpente ha memoria.")


    def get_state(self, game):
        #lo stato sono 11 variabili 
        head = game.snake[0]
        # 20 è il blocco e sever per vedere se vicino c'è quacosa o no
        point_l = Point(head.x-20, head.y)
        point_r = Point(head.x+20, head.y)
        point_u = Point(head.x, head.y-20)
        point_d = Point(head.x, head.y+20)

        #per vedere se la direzione in quel momento è uguale 
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        #state sono 11 elementi che gli passo tutti insieme che sono:
        #Pericolo dritto, Pericolo destra,pericolo sinistra,Direzione Sinistra,Direzione Destra
        #Direzione Su,Direzione Giù,Cibo a sinistra,Cibo a destra,Cibo su,Cibo giù
        state = [
            # pericolo dritto
            (dir_r and game.is_collision(point_r)) or # es: se sto andando aa destra e a destra c'è un pericolo di collisione
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            #pericolo destra
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # sistra
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            #direzione in quel momento
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # dov'è cibo
            game.food.x < game.head.x, 
            game.food.x > game.head.x,  
            game.food.y < game.head.y,  
            game.food.y > game.head.y 
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #toglie se memoria è > di max

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample (self. memory, BATCH_SIZE) # per campione random, da lista tule
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample) # dividi tutti state....
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        #mosse casuali
        self.epsilon = 80-self.n_games
        final_move =[0,0,0]
        if random.randint(0,200)<self.epsilon: #se ho poche partite va casualmente piu volte con questo
            move = random.randint(0,2)
            final_move[move] =1#metti 1 dove ha vinto
        else:
            state0 = torch.tensor(state, dtype=torch.float) #trasforma lista in tensor
            prediction = self.model(state0) #predice cosa fare
            move = torch.argmax(prediction).item() #Cerca l'indice del numero più grande all'interno della previsione
            final_move[move] =1 #metti 1 dove ha vinto
        return final_move

def train():
    record = 0
    n_games_start = 0
    if os.path.exists("record.txt"):
        with open("record.txt", "r") as f:
            data = f.read().split(",")
            record = int(data[0])
            n_games_start = int(data[1])
            print(f"Riprendo da Partita: {n_games_start}, Record: {record}")

    agent =Agent()
    agent.n_games = n_games_start
    game = SnakeGameAI()
    while True: #andiamo all'nifinito finche non chiudo 
        #prendo vecchio stato
        state_old = agent.get_state(game)

        #prendo mossa
        final_move = agent.get_action(state_old)

        #eseguo mossa e faccio nuovo stato
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #alleno memoria breve 
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #ricordi
        agent.remember(state_old, final_move, reward, state_new, done)
 
        if done: #se perdo
            # alleno memoria lunga 
            game.reset()
            agent.n_games +=1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            # Qui scrivi nel file ogni volta che finisce una partita
            with open("record.txt", "w") as f:
                f.write(f"{record},{agent.n_games}")

            print("game", agent.n_games, " score", score, " record", record)

if __name__ == "__main__":
    train()