from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler
from random import shuffle

class class_balancer(Sampler):
    def __init__(self, arr, bs ,ratio,bn):
        
        self.ratio=np.array(ratio)
        self.counts=[int(ratio[0]*(bs//self.ratio.sum())),int(ratio[1]*(bs//self.ratio.sum())),
                     int(ratio[2]*(bs//self.ratio.sum())),
               int(ratio[3]*(bs//self.ratio.sum())),int(ratio[4]*(bs//self.ratio.sum()))] #this can be generalized for any classes
        
       
        self.bs=bs
        self.arr=arr
        self.batch_num=bn
        #self.trn_idx=trn_idx
        
    def __iter__(self):
        print('y1')
        flat_batch=[]
        
        
        for i in range(self.batch_num):
            
            sample=[random.sample(np.where(self.arr==i)[0].tolist(),c) for i,c in enumerate(self.counts)]
           
            sample=np.hstack(sample).tolist()
            if len(sample)<self.bs:
                
                sample=sample+random.sample(sample,self.bs-len(sample))
            random.shuffle(sample)
            flat_batch.append(sample )
        #sample=np.hstack(sample).tolist()
        flat_batch=np.hstack(flat_batch).tolist()
        
        #np.array(flat_batch).flatten().tolist()
        flat_batch=flat_batch[0:self.arr.shape[0]]
        print('unique',np.unique(self.arr[flat_batch],return_counts=True))
        #print(type(flat_batch),len(flat_batch))
        return iter(flat_batch)

    def __len__(self):
        
        return len(self.arr.tolist())

class Sampling_call_back(LearnerCallback):
    def __init__(self,learn:Learner,weights=None,bn=None):
        super().__init__(learn)
        labels = self.learn.data.train_dl.dataset.y.items.astype(int)
        self.labels_array=np.array(list(labels))
        _,self.counts = np.unique(labels, return_counts=True)
        self.bn=bn
       
       
       
       
        self.ratio=weights
        self.bs=self.learn.data.train_dl.batch_size
        self.learn=learn
        #self.weights = (weights if weights is not None else torch.DoubleTensor(counts[labels]))
       
    def on_epoch_begin(self,**kwargs):
        print('e')
        self.sample= class_balancer(arr=self.labels_array,bs=self.bs,ratio=self.ratio,bn=self.bn)
        self.learn.data.train_dl.dl.batch_sampler = BatchSampler(self.sample,self.learn.data.train_dl.batch_size, False)
 
 #pass this as a callback to Learner object 
 learn.call_back_fns.append(partial(Sampling_call_back,weights,len(learn.data.train_dl))
