import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)#Prende i dati in entrata e li trasmette a uno strato intermedio
        self.linear2 = nn.Linear(hidden_size, output_size)#Prende i calcoli fatti dallo strato nascosto e li restringe al numero di azioni possibili

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name="model.pth"):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        # Salva i pesi attuali di questa rete (self)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model 
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)# ottimizzatore
        self.criterion = nn.MSELoss()#calcola quanto è grande l'errore fatto

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float) #dire sempre tipo
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        #n,x

        if len(state.shape) ==1:
            #1,x
            #unsqueeze serve a far diventare i dati da lista a matrice
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            #done diventa tupla
            done = (done, )

        #predizione valore q
        pred = self.model(state)

        target = pred.clone()
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            #se il gioco non è finito dopo questa mossa, calcoliamo il valore futuro
            if not done[idx]:
                # torch.max cerca la mossa migliore che prevede di poter fare
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        #ricompensa + gamma * max(prossimo q valore)

        #cancella i vecchi errori
        self.optimizer.zero_grad()
        #calcola la differenza tra mossa giusta e mossa fatta
        loss = self.criterion(target, pred)
        #capisce cosa ha sbagliato
        loss.backward()
        #corregge
        self.optimizer.step()


