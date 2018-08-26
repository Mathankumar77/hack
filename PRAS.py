
# coding: utf-8

# In[ ]:



import timeit 
import glob 
import cv2 
  
from skimage import transform 
import skimage 
from skimage import io 
  
import sklearn 
from sklearn.model_selection import train_test_split   
import keras 
from keras.preprocessing import image as image_utils 
from keras.callbacks import ModelCheckpoint 
   
rcdefaults() 
matplotlib.rc('font',family='Bitstream Vera Serif') 
  


# In[ ]:


def load_set(videofile): 
'''The input is the path to the video file - the training videos are 99 frames long and have resolution of 720x1248 
This will be used for each video, individially, to turn the video into a sequence/stack of frames as arrays 
The shape returned (img) will be 99 (frames per video), 144 (pixels per column), 256 (pixels per row)) 
''' 
vidcap = cv2.VideoCapture(videofile) 
success,image = vidcap.read() 
count = 0      
error = ''      
success = True  
 

   img = []        
while success:
success, img = vidcap.read() 
count += 1  
frames = []  
for j in range(0,99): 
try: 
success, img = vidcap.read() 
               
                 tmp = skimage.color.rgb2gray(numpy.array(img)) 
                 
                  
                  
tmp = skimage.transform.downscale_local_mean(tmp, (5,5)) 
                 frames.append(tmp) 
                 count+=99 
              
             except: 
               count+=1 
                 pass#print 'There are ', count, ' frame; delete last'        read_frames(videofile, name) 
      


# In[ ]:


if numpy.shape(frames)==(99, 144, 256): 
             all_frames.append(frames) 
          
         elif numpy.shape(frames[0])==(144,256): 
             
             all_frames.append(numpy.concatenate((all_frames[-1][-(99-len(frames)):], frames))) 
       elif numpy.shape(frames[0])!=(144,256): 
             error = 'Video is not the correct resolution.' 
   vidcap.release() 
     del frames; del image 
   return all_frames, error
 
img_filepath = '/pathway/to/videos'  
 neg_all = glob.glob(img_filepath + 'negative/*.mp4')               
pos_2 = glob.glob(img_filepath + 'positive/*.mp4')                  
 pos_1 = glob.glob(img_filepath + '../YTpickles/*.pkl')             
pos_all = concatenate((pos_1, pos_2)) 
 

 all_files = concatenate((pos_all, neg_all)) 
print len(neg_all), len(pos_all)                                   


# In[ ]:


def label_matrix(values): 
   '''transforms labels for videos to one-hot encoding/dummy variables''' 
     n_values = numpy.max(values) + 1                                                
   return numpy.eye(n_values)[values]  

labels = numpy.concatenate(([1]*len(pos_all), [0]*len(neg_all[0:len(pos_all)])))  
 labels = label_matrix(labels)            


# In[ ]:


def make_dataset(rand): 
      seq1 = numpy.zeros((len(rand), 99, 144, 256))    
    for i,fi in enumerate(rand):                     
          print (i, fi)                               
        if fi[-4:] == '.mp4': 
              t = load_set(fi)                         
        elif fi[-4:]=='.pkl': 
              t = pickle.load(open(fi, 'rb'))         
        if shape(t)==(99,144,256):                   
              seq1[i] = t                              
        else:# TypeError: 
              'Image has shape ', shape(t), 'but needs to be shape', shape(seq1[0]) 
            pass                                   


# In[ ]:


print (shape(seq1)) 
   return seq1  
 # The data is then split (with the labels created above) into training and validation sets, 
# with 60% of the total set as training and 20% as validation (the remaining 20% of data is 
# left as a holdout test set). 
# The split fractions may look a little odd, but they are essentially ensuring that the validation 
# and test sets are the same size (an overall 60-20-20 for training-validation-test).  

x_train, x_t1, y_train, y_t1 = train_test_split(all_files, labels, test_size=0.40, random_state=0)  
x_train = array(x_train); y_train = array(y_train)                            
 
x_testA = array(x_t1[len(x_t1)/2:]); y_testA = array(y_t1[len(y_t1)/2:])    
 
 x_testB = array(x_t1[:len(x_t1)/2]); y_test = array(y_t1[:len(y_t1)/2])     
x_test = make_dataset(x_testB) 
# Below, a test was run to check whether there is signal above the noise -- fake data was 
# generated from random numbers to show that the real data performed better than data/patterns 
# picked up from random data. 
# Thankfully, the model could barely reach 50% accuracy when run with random numbers. 
#seq3 = zeros((60,99,144,256)) 
#for j in range(60):  
#    [np.random.random((244,256)) for i in range(99)]    
#print (shape(seq3))               
#x_train2, x_test2, y_train2, y_test2 = train_test_split(seq3, labels, test_size=0.2, random_state=0)  ### split 
 #x_train2 = array(x_train2); y_train2 = array(y_train2)     
#x_test2 = array(x_test2); y_test2 = array(y_test2)         
# Below, the HRNN is set up to run on the train and validation set. 
 
"""HRNNs can learn across multiple levels of temporal hiearchy over a complex sequence. 
Usually, the first recurrent layer of an HRNN encodes a time-dependent video (e.g. set of images) 
 into a vector. The second recurrent layer then encodes those vectors (encoded by the first layer) into a second layer. 
# References 
   - [A Hierarchical Neural Autoencoder for Paragraphs and Documents](https://arxiv.org/abs/1506.01057) 
    - [Hierarchical recurrent neural network for skeleton based action recognition](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298714) 
The first LSTM layer first encodes every column of pixels of shape (240, 1) to a column vector of shape (128,). 
The second LSTM layer encodes then these 240 column vectors of shape (240, 128) to a image vector representing the whole image.  
A final Dense layer is added for prediction. 
""" 


# In[ ]:


import keras 
from keras.models import Model 
from keras.layers import Input, Dense, TimeDistributed 
from keras.layers import LSTM  
batch_size = 15 
num_classes = 2 
epochs = 30  
row_hidden = 128 
col_hidden = 128  
print('x_train shape:', x_train.shape) 
print(x_train.shape[0], 'train samples') 
print(x_test.shape[0], 'test samples')  
frame, row, col = (99, 144, 256) 
 x = Input(shape=(frame, row, col)) 
encoded_rows = TimeDistributed(LSTM(row_hidden))(x)   
  encoded_columns = LSTM(col_hidden)(encoded_rows)      
prediction = Dense(num_classes, activation='softmax')(encoded_columns) 
  model = Model(x, prediction) 
model.compile(loss='categorical_crossentropy',
                optimizer='NAdam',               
                metrics=['accuracy'])           
  i=0; filepath='HRNN_pretrained_model.hdf5' 
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') 
  callbacks_list = [checkpoint]   
numpy.random.seed(18247)  


# In[ ]:


for i in range(0, 30):               
random.shuffle(c)                
x_shuff, y_shuff = zip(*c)       
      x_shuff = array(x_shuff); y_shuff=array(y_shuff) 
             x_batch = [x_shuff[i:i + batch_size] for i in range(0, len(x_shuff), batch_size)]  
y_batch = [y_shuff[i:i + batch_size] for i in range(0, len(x_shuff), batch_size)]  
      for j,xb in enumerate(x_batch):  
          xx = make_dataset(xb)        
        yy = y_batch[j]                         
model.fit(xx, yy,                            
                    batch_size=len(xx),                 
                    epochs=1,                          
                    validation_data=(x_test, y_test),  
                    callbacks=callbacks_list)           
    # evaluate 
scores = model.evaluate(x_test, y_test, verbose=0)    


# In[ ]:


print('Test loss:', scores[0])                        
print('Test accuracy:', scores[1])                    
  model.load_weights("HRNN_pretrained_model.hdf5") 
model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy']) 
 x_holdout = make_dataset(x_testA) 
plot([0,1],[0,1],'k:',alpha=0.5)                       
  ys = [y_train, y_test, y_testA]                       
labs = ['Train', 'Valid', 'Test']                      
  col = ['#4881ea', 'darkgreen', 'maroon']               
preds = []                                             
for i,xset in enumerate([x_train, x_testB, x_testA]):  


# In[ ]:



   if i==0: 
         new_pred = []                                  
       for k in xset:                                 
             d = make_dataset([k])                       
           new_pred.append(model.predict(d))          
       new_pred = array(new_pred).reshape((len(new_pred),2)) 
   else: 
         d = make_dataset(xset)                         
       new_pred = model.predict(d)                    
   preds.append(new_pred) 
     fpr, tpr, threshs = sklearn.metrics.roc_curve(ys[i][:,1], new_pred[:,1]) 
   plot(fpr, tpr, '-', color=col[i], alpha=0.7, lw=1.5, label=labs[i])      
      
   print labs[i] 
     print sklearn.metrics.auc(fpr, tpr)                
   print sklearn.metrics.accuracy_score(ys[i][:,1], [round(j) for j in new_pred[:,1]])   
     print sklearn.metrics.confusion_matrix(ys[i][:,1], [round(j) for j in new_pred[:,1]])       
       
   xlabel('False Positive Rate'); ylabel('True Positive Rate') 
plt.legend(fancybox=True, loc=4, prop={'size':10}) 
 plt.show() 
   plot([0,1],[0,1],'k:',alpha=0.5)                  
for i,p in enumerate(preds):                      
     hist(p[:,1], bins = arange(0,1,0.05), histtype='stepfilled', color=col[i], alpha=0.7, label=labs[i] 
 xlabel('False Positive Rate'); ylabel('True Positive Rate') 
 plt.legend(fancybox=True, loc=2, prop={'size':10}) 
 plt.show() 

