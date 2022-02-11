import torch 
from torch import multiprocessing,nn 
 
output_size = 13 
input_dim = output_size 
hidden_dim = 128 
layer_dim = 1 
#  
lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True) 
torch.save(lstm.state_dict(),"plain.pt") 
# model = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True) 
# model.share_memory()
fcn = torch.nn.Sequential(
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.Linear(100,1),
            nn.Sigmoid()
        )
fcn.share_memory()
def run(state): 
    # print(i)
    model = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True) 
    # model.load_state_dict(torch.load("plain.pt")) 
    x=model(state) 
    return x 
 
state=torch.rand(1,10,13)
threads=[] 
for i in range(1): 
    p = multiprocessing.Process(target=run,args=(state,)) 
    threads.append(p) 
    p.start() 
for i in threads: 
    i.join()
# multiprocessing.spawn(run, args=(state,), nprocs=5, join=False) 
print("CHECKPOINT One") 
# run(state) 
a = fcn(torch.randn(2,100))
print("CHECKPOINT Two")
threadss=[] 
# multiprocessing.spawn(run, args=(state,)) 

for i in range(1): 
    p = multiprocessing.Process(target=run,args=(state,)) 
    threadss.append(p) 
    p.start() 
for i in threadss: 
    i.join()
print("CHECKPOINT Three")