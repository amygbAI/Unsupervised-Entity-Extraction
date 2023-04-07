import os, math, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import tensorflow_datasets as tfds
from collections import Counter

inpDim_, total_feats_ = 256, 6
#EPOCHS, sampler_, breaker_ = 30, 20, 100
EPOCHS, sampler_, breaker_ = 40, 500, 100000000
label_maps_ = {'KEY-VALUE-BELOW':2, 'KEY-VALUE-RIGHT':3, 'VALUE-KEY-LEFT':4, 'VALUE-KEY-TOP':5, 'IRR-NA':1, 'PAD':0}

def allNum( wd_ ):
  digs, special, illegal, digs2 =0, 0, 0, 0

  arr_ = wd_.split()
  ## fir conjoined DATE 31/12/2003 instead of just 21/12/2004
  if len( arr_ ) > 1 and len( arr_[-1] ) >= 3:
    chk = arr_[-1]
    for char in chk:
      if ord(char) >= 48 and ord(char) <= 57: digs += 1
      if ord(char) >= 65 and ord(char) <= 90: digs2 += 1
      if ord(char) >= 97 and ord(char) <= 122: illegal += 1
      if char in [',','.','$','S','-','/']: special += 1
    if ( digs+digs2+special == len( chk ) and digs >= 1 ) or ( digs >= 4 and illegal <= 4 ): return True

  digs, special, illegal, digs2 =0, 0, 0, 0
  for char in wd_:
    if ord(char) >= 48 and ord(char) <= 57: digs += 1
    if ord(char) >= 65 and ord(char) <= 90: digs2 += 1
    if char in [',','.','$','S','-','/']: special += 1
    if ord(char) >= 97 and ord(char) <= 122: illegal += 1

  if ( digs+digs2+special == len( wd_ ) and digs >= 2 ) or ( digs >= 4 and illegal <= 4 ): return True
  return False

def findWdFeats( refwd ):

    txt = refwd
    if ':' in txt and len( txt.split(':')[-1] ) > 1:
      txt = txt.split(':')[-1]

    returnFeats = np.zeros((total_feats_))

    if len( txt.split() ) < 1: return returnFeats, 0
    #if len( txt.split() ) < 1: return returnFeats.tolist() + np.zeros((4)).tolist()

    if allNum( txt.split()[-1] ) and len( txt.split()[-1] ) >= 4 and not\
      ( ',' in txt or 'box' in txt.lower() ):
      txt = txt.split()[-1]

    digs, caps, small, special, begcaps = 0, 0, 0, 0, 0
    for char in txt:
      if ord(char) >= 48 and ord(char) <= 57: digs += 1
      if ord(char) >= 65 and ord(char) <= 90: caps += 1
      if ord(char) >= 97 and ord(char) <= 122: small += 1
      if char in [',','.','$','S','-','/',' ']: special += 1

    lenwds_ = []
    for wd in txt.split():
      if ord( wd[0] ) >= 65 and ord( wd[0] ) <= 90: begcaps += 1
      lenwds_.append( len(wd) )

    returnFeats[0] = digs
    returnFeats[1] = caps
    returnFeats[2] = small
    returnFeats[3] = len( txt.split() )
    returnFeats[4] = begcaps
    #returnFeats[5] = len( txt )
    returnFeats[5] = np.median( lenwds_ )

    sum_ = 0
    #print('MACADEMIA NUTS ->', txt,' returnFeats ',returnFeats, ' LEN = ', len(returnFeats))
    for ctr in range( len(returnFeats) ):
      sum_ +=  returnFeats[ctr] * math.pow( 2, len(returnFeats)- ctr )

    return returnFeats, int( sum_ )

def retCoOrdIdx( coords_ ):

    h_, w_ = coords_[3]-coords_[1], coords_[2]-coords_[0]
    return int(coords_[0]*1000) + int(coords_[1]*1000) + int(w_*100) + int(h_*100)
    #return int(coords_[0]*0.1) + int(coords_[1]*0.05) + w_ + h_ 

def loadLocal( folder_ ):

    ll_, vec_sz = os.listdir( folder_ ), 256
    inpCnt_ , inptCoOrds_ , labels_, actualInp = [], [], [], []
    master_ctr_idx_, master_coord_idx_, master_neigh_vert, master_neigh_hor = [], [], [], []
    fnList_ = []

    for elem in ll_: 
      print('Processing file ->', elem)    
      with open( folder_ + elem, 'rb' ) as fp:
        np_arr_ = np.load( fp, allow_pickle=True )

      ## lets assume input vector size = 256
      
      locCnt_, locCoOrd_, locLabel_, neigh_axis0_size, neigh_axis1_size = [], [], [], 0, 0
      contour_index_ , coord_index_, actual_stuff, neigh_index_vert, neigh_index_hor = [], [], [], [], []

      for tuple_ in np_arr_:
        txt_, coords_, labels, neigh_vert, neigh_hor = tuple_
        feats_, indx = findWdFeats( txt_ ) 
        actual_stuff.append( (txt_, coords_) )
        co_ord_idx_ = retCoOrdIdx( coords_ )

        if indx < 0 or co_ord_idx_ < 0: continue

        #contour_index_.append( feats_.tolist() + neigh_ )
        contour_index_.append( indx )
        master_ctr_idx_.append( indx )

        np_neigh_vert = np.asarray( neigh_vert )
        neigh_axis0_size, neigh_axis1_size = np_neigh_vert.shape[0], np_neigh_vert.shape[1]

        np_mixed_vert = np.asarray( np.split( np_neigh_vert, np_neigh_vert.shape[0], axis=-1 ) )
        normal_vert = np_mixed_vert.reshape( np_neigh_vert.shape[0]*np_neigh_vert.shape[1] )
        #normal_vert = np_neigh_vert.reshape( np_neigh_vert.shape[0]*np_neigh_vert.shape[1] )

        np_neigh_hor = np.asarray( neigh_hor )
        np_mixed_hor = np.asarray( np.split( np_neigh_hor, np_neigh_hor.shape[0], axis=-1 ) )
        normal_hor = np_mixed_hor.reshape( np_neigh_hor.shape[0]*np_neigh_hor.shape[1] )
        #normal_hor = np_neigh_hor.reshape( np_neigh_hor.shape[0]*np_neigh_hor.shape[1] )

        neigh_index_vert.append( normal_vert )
        neigh_index_hor.append( normal_hor )

        coord_index_.append( co_ord_idx_ )
        master_coord_idx_.append( co_ord_idx_ )

        locCnt_.append( feats_ )
        locCoOrd_.append( coords_ + [ ( coords_[2] - coords_[0] ), ( coords_[3] - coords_[1] ) ] )
        locLabel_.append( label_maps_[ labels ] )    

      if len( locCnt_ ) > vec_sz:
        locCnt_ = locCnt_[ :vec_sz ]
        locCoOrd_ = locCoOrd_[ :vec_sz ]

        contour_index_ = contour_index_[ :vec_sz ]
        coord_index_   = coord_index_[ :vec_sz ]

        neigh_index_vert   = neigh_index_vert[ :vec_sz ]
        neigh_index_hor    = neigh_index_hor[ :vec_sz ]

        locLabel_ = locLabel_[ :vec_sz ]
      else:
        for ctr in range( vec_sz - len(locCnt_) ):
          locCnt_.append( np.zeros((total_feats_)) )
          locCoOrd_.append( np.zeros((total_feats_)) )

          neigh_index_vert.append( np.zeros( neigh_axis0_size* neigh_axis1_size ).tolist() )
          neigh_index_hor.append( np.zeros( neigh_axis0_size* neigh_axis1_size ).tolist() )

          coord_index_.append( 0 ) 
          contour_index_.append( 0 ) 
          locLabel_.append( label_maps_['PAD'] )    
          actual_stuff.append( ('NA',0) )

      #inpCnt_.append( locCnt_ )
      #inptCoOrds_.append( locCoOrd_ )
      inpCnt_.append( contour_index_ )
      inptCoOrds_.append( coord_index_ )
      labels_.append( locLabel_ )

      master_neigh_vert.append( neigh_index_vert )
      master_neigh_hor.append( neigh_index_hor )

      fnList_.append( elem )
      actualInp.append( actual_stuff )

      if len(inpCnt_) > breaker_: break

    #contour_index_ , coord_index_ = [], []
    print( 'Contour ids = ', np.min( contour_index_ ), np.max( master_ctr_idx_ ), \
                             np.max( master_ctr_idx_ ) - np.min( contour_index_ ) )
    print( 'Coords ids = ', np.min( coord_index_ ), np.max( master_coord_idx_ ), \
                             np.max( master_coord_idx_ ) - np.min( coord_index_ ) )

    #master_ctr_idx_, master_coord_idx_ = [], []
    return inpCnt_, inptCoOrds_, labels_, int( np.max( master_ctr_idx_ )+1 ), int( np.max( master_coord_idx_ )+1 ), \
                                        fnList_, actualInp, master_neigh_vert, master_neigh_hor


inpCnt_, inptCoOrds_, labels_, cntEmbInpDim_, coordEmbInpDim, fnmList_, \
                            actualInp, master_neigh_vert, master_neigh_hor = loadLocal('./DATA_ENHANCED_KV/') 
#inpCnt_, inptCoOrds_, labels_, cntEmbInpDim_, coordEmbInpDim, fnmList_ = loadLocal('../DATA/') 
 
print( np.asarray( inpCnt_ ).shape, np.asarray( inptCoOrds_ ).shape, np.asarray( labels_ ).shape, \
                                                  np.asarray( master_neigh_vert ).shape   )
#exit()
num_samples_ = np.asarray( inpCnt_ ).shape[0]
train_dataset = tf.data.Dataset.from_tensor_slices(( ( inpCnt_[: num_samples_ - sampler_], inptCoOrds_[: num_samples_ - sampler_], master_neigh_vert[ : num_samples_ - sampler_ ], master_neigh_hor[ : num_samples_ - sampler_ ] ), labels_[: num_samples_ - sampler_] ))

val_dataset = tf.data.Dataset.from_tensor_slices(( (inpCnt_[-1*sampler_:], inptCoOrds_[-1*sampler_:], master_neigh_vert[-1*sampler_:], master_neigh_hor[-1*sampler_:] ), labels_[-1*sampler_:] ))

val_fnm_ = fnmList_[-1*sampler_:]
val_labels_ = labels_[-1*sampler_:]

val_acutal_ = actualInp[-1*sampler_:]

BATCH_SIZE, val_batch_sz = 32, sampler_
SHUFFLE_BUFFER_SIZE = 100

'''
train_dataset = tf.data.Dataset.from_generator( dataGen_, \
                              output_signature=( \
                              ( tf.TensorSpec(shape=(), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.int32) ), \
                              tf.TensorSpec(shape=(), dtype=tf.int32) ) )

train_dataset = train_dataset.batch( BATCH_SIZE ).shuffle(SHUFFLE_BUFFER_SIZE)
'''

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.batch( val_batch_sz )

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        print('COFFEE->', inputs)
        #tf.print( "tensors-> ",tf.shape( inputs ), output_stream=sys.stdout ) 
        attn_output = self.att(inputs, inputs)
        print('COFFEE2->', attn_output)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim, name='TOKEN_EMB'
        )

        self.neigh_emb_prior = keras.layers.Dense( 128, name='NEIGH_EMB_PRIOR')
        self.n1flat = keras.layers.Flatten()
        self.neigh_emb = keras.layers.Dense( embed_dim, name='NEIGH_EMB')

        self.neigh_emb_prior1 = keras.layers.Dense( 128, name='NEIGH_EMB_PRIOR1')
        self.n2flat = keras.layers.Flatten()
        self.neigh_emb1 = keras.layers.Dense( embed_dim, name='NEIGH_EMB1')

        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim, name='POS_EMB')

    #def call(self, inputs):
    def call(self, inptxtFeats, inpCoords, neigh_vert, neigh_hor ):
        maxlen = tf.shape( inptxtFeats )[-1]
        #maxlen = tf.shape(inputs)[-1]
        #tf.print( "POS EMB -> ", inpCoords , output_stream=sys.stdout ) 
        #tf.print( "TOK EMB -> ", inptxtFeats , output_stream=sys.stdout ) 
        positions = tf.range(start=0, limit=maxlen, delta=1)

        position_embeddings = self.pos_emb( positions )

        prior_ = self.neigh_emb_prior( neigh_vert )
        neigh_embeddings_vert = self.neigh_emb( prior_ ) 

        prior_1 = self.neigh_emb_prior1( neigh_hor )
        neigh_embeddings_hor = self.neigh_emb1( prior_1 ) 

        token_embeddings = self.token_emb( inptxtFeats )
        #return neigh_embeddings
        return token_embeddings + position_embeddings + neigh_embeddings_vert + neigh_embeddings_hor

class NERModel(keras.Model):
    def __init__(
        self, num_tags, vocab_size=256, maxlen=128, embed_dim=128, num_heads=8, ff_dim=256
    ):
        super().__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout1 = layers.Dropout(0.1)
        self.ff = layers.Dense(ff_dim, activation="relu")
        self.dropout2 = layers.Dropout(0.1)
        self.ff_final = layers.Dense(num_tags)

    def call(self, inputs, training=False):
    #def call(self, inptxtFeats, inpCoords, training=False):
        print('BODO->', inputs)
        #x = self.embedding_layer(inputs)
        inptxtFeats, inpCoords, neigh_vert, neigh_hor = inputs

        x = self.embedding_layer( inptxtFeats, inpCoords, neigh_vert, neigh_hor )
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x

#cntEmbInpDim_, coordEmbInpDim
print('MEGASTHENES->', cntEmbInpDim_, coordEmbInpDim)
ner_model = NERModel( len( label_maps_ ), vocab_size=cntEmbInpDim_, maxlen=coordEmbInpDim, \
                                                         embed_dim=32, num_heads=12, ff_dim=64)

class CustomNonPaddingTokenLoss(keras.losses.Loss):
    def __init__(self, name="custom_ner_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=keras.losses.Reduction.NONE
        )
        #print('JUS B4 HIT->', y_true,  y_pred)
        loss = loss_fn(y_true, y_pred)
        #mask = tf.cast((y_true > 1), dtype=tf.float32) ## ignore both pad and IRR labels for loss calc
        mask = tf.cast((y_true > 0), dtype=tf.float32)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


loss = CustomNonPaddingTokenLoss()

ner_model.compile(optimizer="adam", loss=loss)
ner_model.fit(train_dataset, epochs=EPOCHS)

ner_model.save( './NER_MODEL/model_three_embed_multi_neigh_split' )

sample_input = val_dataset

loaded_model = keras.models.load_model( './NER_MODEL/model_three_embed_multi_neigh_split', \
        {'CustomNonPaddingTokenLoss': CustomNonPaddingTokenLoss }, compile=False )

output = loaded_model.predict(sample_input)
#output = ner_model.predict(sample_input)
print('PREDN->', output.shape)

for file_ctr in range( output.shape[0]-1 ): 
  fnm = val_fnm_[ file_ctr ] 
  file_contours_ = output[ file_ctr ]

  print('Predns for filnm->', fnm)
  fp = open('./JSON_RES_SPLIT/_'+fnm, 'w+')

  matches_, tot = 0, 0
  for contour_ctr in range( len(file_contours_) ):
    contour = file_contours_[ contour_ctr ]
    prediction = np.argmax(contour)
    #print( val_labels_[ file_ctr ][ contour_ctr ] )
    sht_ = val_labels_[ file_ctr ][ contour_ctr ]
    label_pred = list( label_maps_.keys() )[list( label_maps_.values() ).index( sht_ ) ]
    label_loc = list( label_maps_.keys() )[list( label_maps_.values() ).index( prediction )]

    if label_pred == label_loc and label_loc not in ['IRR-NA', 'PAD']: 
      matches_ += 1
      print('Act->', label_pred,'Pred->',label_loc,' || TEXT , COORDS || ', val_acutal_[ file_ctr ][contour_ctr] )
      fp.write( 'PASS#'+str(val_acutal_[ file_ctr ][contour_ctr])+'#'+str( label_pred )+'#'+str( label_loc ) + '\n' )
      tot += 1

    elif label_pred != label_loc and label_pred not in ['PAD']: 
      fp.write( 'FAIL#'+str(val_acutal_[ file_ctr ][contour_ctr])+'#'+str( label_pred )+'#'+str( label_loc ) + '\n' )
      tot += 1

    #if label_pred not in ['IRR-NA', 'PAD']: tot += 1

  fp.write( 'MATCH CTR-> ||'+str(matches_)+'||'+str(tot) )
  fp.close()
  print('Match counter fr ', fnm, ' == ', matches_, tot)
          

