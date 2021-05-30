# coding: utf-8
'''
ch16_part2_ryGenNovels_v02.py

基於RNN的自動文字產生...
神雕、倚天
ryGenNovels.py

'''

import numpy as np
import tensorflow as tf


#%%

書庫= [#'movie_data.csv', 
       #'1268-0_TheMysteriousIsland.txt', 
       '神雕俠侶.txt', 
       '倚天屠龍記.txt']

text= ''
for 書 in 書庫:
    with open(書, 'r', encoding='utf8') as fp:
        text += fp.read()

#%%
#在任2漢字中插入半形空白
'''
aText= text
bText= [' ']*len(aText)*2
bText[0:-1:2]= aText
text= cText= ''.join(bText)
'''
#print(cText)

#%%

start_indx= 0 #text.find('第一回')
end_indx= -1 #text.find('已完结')
print(start_indx, end_indx)

text= text[start_indx:end_indx]
char_set= set(text)

print('Total Length:', len(text))
print('Unique Characters:', len(char_set))


#0
'''
Total Length: 1131711
Unique Characters: 88
'''

#1
'''
Total Length: 994559
Unique Characters: 3945
'''

#2
'''
Total Length: 964789
Unique Characters: 3927
'''

# 1+2
'''
Total Length: 1959349
Unique Characters: 4353
'''

#%%



chars_sorted= sorted(char_set)
char2int= {ch:i for i, ch in enumerate(chars_sorted)}
char_array= np.array(chars_sorted)

text_encoded = np.array(
    [char2int[ch] for ch in text],
    dtype=np.int32)

print('Text encoded shape: ', text_encoded.shape)

print(text[:15], '     == Encoding ==> ', text_encoded[:15])
print(text_encoded[15:21], ' == Reverse  ==> ', ''.join(char_array[text_encoded[15:21]]))


#%%








ds_text_encoded = tf.data.Dataset.from_tensor_slices(text_encoded)

for ex in ds_text_encoded.take(5):
    print('{} -> {}'.format(ex.numpy(), char_array[ex.numpy()]))




seq_length = 100
chunk_size = seq_length + 1

ds_chunks = ds_text_encoded.batch(chunk_size, drop_remainder=True)

## inspection:
for seq in ds_chunks.take(1):
    input_seq = seq[:seq_length].numpy()
    target = seq[seq_length].numpy()
    print(input_seq, ' -> ', target)
    print(repr(''.join(char_array[input_seq])), 
          ' -> ', repr(''.join(char_array[target])))





#%%


## define the function for splitting x & y
def split_input_target(chunk):
    input_seq = chunk[:-1]
    target_seq = chunk[1:]
    return input_seq, target_seq

ds_sequences = ds_chunks.map(split_input_target)

## inspection:
for example in ds_sequences.take(2):
    print(' Input (x):', repr(''.join(char_array[example[0].numpy()])))
    print('Target (y):', repr(''.join(char_array[example[1].numpy()])))
    print()



#%%
# Batch size
BATCH_SIZE=  64
BUFFER_SIZE= 10000

tf.random.set_seed(1)
ds = ds_sequences.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)# drop_remainder=True)

print(ds)
#%%

# ### Building a character-level RNN model



def build_model(vocab_size, embedding_dim, rnn_units):
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        
        tf.keras.layers.SimpleRNN(
            rnn_units, return_sequences=True),
        
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


charset_size = len(char_array)
embedding_dim = 256
rnn_units = 512

tf.random.set_seed(1)

model = build_model(
    vocab_size = charset_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

model.summary()
#%%



model.compile(
    optimizer='adam', 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ))

model.fit(ds, epochs= 20)

#%%
# ### Evaluation phase: generating new text passages



#tf.random.set_seed(1)

logits = [[1.0, 1.0, 1.0]]
print('Probabilities:', tf.math.softmax(logits).numpy()[0])

samples = tf.random.categorical(
    logits=logits, num_samples=10)
tf.print(samples.numpy())




#tf.random.set_seed(1)

logits = [[1.0, 1.0, 3.0]]
print('Probabilities:', tf.math.softmax(logits).numpy()[0])

samples = tf.random.categorical(
    logits=logits, num_samples=10)
tf.print(samples.numpy())


#%%

def generateText(model, starting_str, 
           len_generated_text= 100, 
           max_input_length=   10,
           scale_factor=       1.0):
    
    encoded_input = [char2int[s] for s in starting_str]
    encoded_input = tf.reshape(encoded_input, (1, -1))

    generated_str = starting_str

    model.reset_states()
    for i in range(len_generated_text):
        logits = model(encoded_input)
        logits = tf.squeeze(logits, 0)

        scaled_logits = logits * scale_factor
        new_char_indx = tf.random.categorical(
            scaled_logits, num_samples=1)
        
        new_char_indx = tf.squeeze(new_char_indx)[-1].numpy()    

        generated_str += str(char_array[new_char_indx])
        
        if generated_str[-1] in ['。', '？', '！','\n']: 
            break
        
        new_char_indx = tf.expand_dims([new_char_indx], 0)
        encoded_input = tf.concat(
            [encoded_input, new_char_indx],
            axis=1)
        encoded_input = encoded_input[:, -max_input_length:]

    return generated_str


#%%
for i in range(10): 
    x= generateText(model, starting_str= '楊過道')
    print(f'{i}: {x}')

#%%
for i in range(10): 
    x= generateText(model, starting_str= '張無忌道')
    print(f'{i}: {x}')

#%%

