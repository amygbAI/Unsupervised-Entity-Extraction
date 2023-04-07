import os, math, sys, json, cv2
import numpy as np
import urllib.request 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import tensorflow_datasets as tfds
from collections import Counter
import sys

test_fnm = sys.argv[1].split('.json')[0]
inpDim_, total_feats_ = 256, 6

src_0 = '/home/ubuntu/ABHIJEET/INVOICES/CURTISS_WRIGHT/DEV/ALL_OCR_OUTPUT/'
src_raw = '/home/ubuntu/ABHIJEET/INVOICES/CURTISS_WRIGHT/DEV/ALL_OCR_OUTPUT_ORIGINAL/'

#EPOCHS, sampler_, breaker_ = 30, 20, 100
EPOCHS, sampler_, breaker_ = 40, 500, 100000000
label_maps_ = {'KEY-VALUE-BELOW':2, 'KEY-VALUE-RIGHT':3, 'VALUE-KEY-LEFT':4, 'VALUE-KEY-TOP':5, 'IRR-NA':1, 'PAD':0}

def xOverlap( val, pts, ref_val, ref_pts, dist_=150 ):
    ## check if anything above or below
    #print( abs( pts[-1] - ref_pts[1] ), pts[0] >= ref_pts[0] and pts[2] <= ref_pts[2], pts, ref_pts )
    if abs( pts[-1] - ref_pts[1] ) <= dist_ or abs( pts[1] - ref_pts[-1] ) <= dist_:
        if ( pts[0] >= ref_pts[0] and pts[2] <= ref_pts[2] ) or \
           ( ref_pts[0] >= pts[0] and ref_pts[2] <= pts[2] ) or \
           ( pts[0] >= ref_pts[0] and pts[0] <= ref_pts[2] ) or \
           ( ref_pts[0] >= pts[0] and ref_pts[0] <= pts[2] ) or \
           ( ref_pts[0] < pts[0] and ref_pts[2] > pts[0] and ref_pts[2] <= pts[2] and ( abs( abs( ref_pts[0] + ref_pts[2] )/2 - ( abs( pts[0] + pts[2] ) )/2 ) )/( min( abs( ref_pts[0] - ref_pts[2]), abs( pts[0] - pts[2] )  )  ) < 0.8 )            or\
           ( pts[0] < ref_pts[0] and pts[2] > ref_pts[0] and pts[2] <= ref_pts[2] and ( abs( abs( ref_pts[0] + ref_pts[2] )/2 - ( abs( pts[0] + pts[2] ) )/2 ) )/( min( abs( ref_pts[0] - ref_pts[2]), abs( pts[0] - pts[2] )  )  ) < 0.8 ):
             #print( val, pts, ' X OVERLAPS with ', ref_val, ref_pts, abs( ref_pts[0] + ref_pts[2]), abs( pts[0] + pts[2] ), abs( ref_pts[0] + ref_pts[2] )/2,  abs( pts[0] + pts[2] )/2, abs( abs( ref_pts[0] + ref_pts[2] )/2 - ( abs( pts[0] + pts[2] ) )/2 )  )
             return True
    return False

def featNum( txt ):

    digs, caps, small, special = 0, 0, 0, 0
    for char in txt:
      if ord(char) >= 48 and ord(char) <= 57: digs += 1
      if ord(char) >= 65 and ord(char) <= 90: caps += 1
      if ord(char) >= 97 and ord(char) <= 122: small += 1
      if char in [',','.','$','-','/',' ']: special += 1

    if '.0' in txt and digs == 1 and caps > 0 and special > 0: digs = 0
    #print( 'DESPO->', txt, ' digs, caps, special, small = ',digs, caps, special, small)
    if digs+special == len(txt) and digs > 0: return 1 # num
    if digs+caps+special == len(txt) and digs > 0 and not ( digs == 1 and '0' in txt and caps >=3 ): return 2 # alnum
    if digs+caps+special+small == len(txt) and digs > 0 and caps > 0 and not ( digs == 1 and '0' in txt and caps >= 3 ): return 2
    if digs+caps+special+small == len(txt) and digs >=4 : return 2
    if caps+special+small == len(txt) and digs == 0 and small > 0: return 3 # mixed str
    if special+small == len(txt) and digs == 0 and small > 0: return 4 # small cap
    if special+caps == len(txt) and digs == 0: return 5 # large cap

    return 3 # default val

def nothinInBetween( potkey, potval, valpts, keypts, IRRs_, mode, ht_, wd_, assigned_pairs_ ):

    print('NIB-> potkey, potval, ref1, ref2 ', potkey, potval, valpts, keypts, mode) 
    if mode == 'HORIZONTAL':
      for txt, pts, _ in IRRs_:
        if type( pts ) is int: continue

        pts = [ int( pts[0]*wd_ ), int( pts[1]*ht_ ), int( pts[2]*wd_ ),int( pts[-1]*ht_ ) ]

        if ( pts[0] > keypts[2] and pts[0] < valpts[0] ) and abs( pts[1] - keypts[1] ) <= 10 \
          and len( txt ) >= 2:
          print('MODE->', mode, ' Found ', txt, pts, ' between ', potval, ' & ', potkey)
          return False

      for ktup, vtup in assigned_pairs_:  
        if ( ktup[1][0] > keypts[2] and ktup[1][0] < valpts[0] ) and abs( ktup[1][1] - keypts[1] ) <= 10 \
          and len( txt ) >= 2:
          print('MODE->', mode, ' Found KTUP ', ktup, ' between ', potval, ' & ', potkey)
          return False
        if ( vtup[1][0] > keypts[2] and vtup[1][0] < valpts[0] ) and abs( vtup[1][1] - keypts[1] ) <= 10 \
          and len( txt ) >= 2:
          print('MODE->', mode, ' Found VTUP ', vtup, ' between ', potval, ' & ', potkey)
          return False

    if mode == 'VERTICAL':
      for txt, pts, _ in IRRs_:
        if type( pts ) is int: continue

        pts = [ int( pts[0]*wd_ ), int( pts[1]*ht_ ), int( pts[2]*wd_ ),int( pts[-1]*ht_ ) ]

        print( 'LAUGH-> txt, pts, potkey, keypts, potval, valpts = ', txt, pts, potkey, keypts, potval, valpts,\
                 xOverlap( txt, pts, potkey, keypts ), xOverlap( txt, pts, potval, valpts ),\
                 ( abs( pts[1] - valpts[-1] ) <= 10 and pts[1] < valpts[1] ),\
                 ( abs( pts[1] - keypts[-1] ) <= 10 and pts[1] < keypts[1] )  )

        if xOverlap( txt, pts, potkey, keypts ) and xOverlap( txt, pts, potval, valpts ) and len( txt ) >= 2 and\
          ( ( pts[1] > keypts[1] and pts[1] < valpts[1] ) or ( pts[1] > valpts[1] and pts[1] < keypts[1] ) ):
          print('MODE->', mode, ' Found ', txt, pts, ' between ', potval, ' & ', potkey)
          return False

      for ktup, vtup in assigned_pairs_:  

        if xOverlap( ktup[0], ktup[1], potkey, keypts ) and xOverlap( ktup[0], ktup[1], potval, valpts ) \
          and len( txt ) >= 2 and\
          ( ( ktup[1][1] > keypts[1] and ktup[1][1] < valpts[1] ) or\
                                       ( ktup[1][1] > valpts[1] and ktup[1][1] < keypts[1] ) ):
          print('MODE->', mode, ' Found KTUP ', ktup, ' between ', potval, ' & ', potkey)
          return False

        if xOverlap( vtup[0], vtup[1], potkey, keypts ) and xOverlap( vtup[0], vtup[1], potval, valpts ) \
          and len( txt ) >= 2 and\
          ( ( vtup[1][1] > keypts[1] and vtup[1][1] < valpts[1] ) or\
                                       ( vtup[1][1] > valpts[1] and vtup[1][1] < keypts[1] ) ):
          print('MODE->', mode, ' Found VTUP ', vtup, ' between ', potval, ' & ', potkey)
          return False

    return True

def allNum( wd_, splitter_=False ):
  digs, special, illegal, digs2 =0, 0, 0, 0

  arr_ = wd_.split()
  if splitter_ is True:
    neo_ = wd_.replace(' ','')
    for char in neo_:
      if ord(char) >= 48 and ord(char) <= 57: digs += 1
      if ord(char) >= 65 and ord(char) <= 90: digs2 += 1
      if ord(char) >= 97 and ord(char) <= 122: illegal += 1
      if char in [',','.','$','S','-','/']: special += 1
    if ( digs2 >= 1 and special >= 1 ): return False
    if digs+special == len( neo_ ): return True
    
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

  if ( digs+digs2+special == len( wd_ ) and ( digs >= 2 or wd_ == '0' ) )\
         or ( digs >= 4 and illegal <= 4 ): return True
  return False

def findWdFeats( refwd ):

    txt = refwd

    if ':' in txt and len( txt.split(':')[-1] ) > 1 and len( txt.split(':')[0] ) >= 3:
      txt = ' '.join( txt.split(':')[:-1] )
    elif len( txt.split() ) > 1 and len( txt.split()[-1] ) >= 3 and allNum( txt.split()[-1] )\
      and len( ' '.join( txt.split()[:-1] ) ) >= 3:
      txt = ' '.join( txt.split()[:-1] )
    elif len( txt.split() ) > 1 and len( txt.split()[0] ) >= 3 and allNum( txt.split()[0] ) and\
      not allNum( ' '.join( txt.split()[1:] ) ):
      txt = txt.split()[0]

    returnFeats = np.zeros((total_feats_))
    #print('INCOMING->', refwd,' PRO_INP->', txt)
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

def _findRefinedTextFeats( tmpKeyTxt, neigh_hor_main ):

    response_text_arr_ , response_hor_arr_ = [], []

    if ':' in tmpKeyTxt and len( ' '.join( tmpKeyTxt.split(':')[:-1] ) ) >= 5 and len( tmpKeyTxt.split(':')[0] ) >= 3:

      response_text_arr_.append( tmpKeyTxt.split(':')[0] )
      ## horizontal arr needs to be modified ..most neigh remains the same just for the key, the right neigh is replaced with newly splity val and for the val left neigh is replaced with newly split key
      ## FOR KEY
      rt_feat_, _ = findWdFeats( ' '.join( tmpKeyTxt.split(':')[1:] ).strip() )
      response_hor_arr_.append( [ neigh_hor_main[0], neigh_hor_main[1], neigh_hor_main[2], rt_feat_.tolist(),\
                                  neigh_hor_main[4], neigh_hor_main[5] ] )
      ## FOR VAL
      response_text_arr_.append( ' '.join( tmpKeyTxt.split(':')[1:] ).strip() )

      lt_feat_, _ = findWdFeats( tmpKeyTxt.split(':')[0] )
      response_hor_arr_.append( [ neigh_hor_main[0], neigh_hor_main[1], lt_feat_.tolist(), neigh_hor_main[3],\
                                  neigh_hor_main[4], neigh_hor_main[5] ] )

    elif len( tmpKeyTxt.split() ) > 1 and len( tmpKeyTxt.split()[-1] ) >= 3 and allNum( tmpKeyTxt.split()[-1] )\
        and len( ' '.join( tmpKeyTxt.split()[:-1] ) ) >= 3:

      response_text_arr_.append( tmpKeyTxt.split()[-1] )
      ## horizontal arr needs to be modified ..most neigh remains the same just for the key, the right neigh is replaced with newly splity val and for the val left neigh is replaced with newly split key
      ## FOR KEY
      rt_feat_, _ = findWdFeats( tmpKeyTxt.split()[-1] )
      response_hor_arr_.append( [ neigh_hor_main[0], neigh_hor_main[1], neigh_hor_main[2], rt_feat_.tolist(),\
                                  neigh_hor_main[4], neigh_hor_main[5] ] )

      ## FOR VAL
      response_text_arr_.append( ' '.join( tmpKeyTxt.split()[:-1] ) )
      lt_feat_, _ = findWdFeats( ' '.join( tmpKeyTxt.split()[:-1] ) )
      response_hor_arr_.append( [ neigh_hor_main[0], neigh_hor_main[1], lt_feat_.tolist(), neigh_hor_main[3],\
                                  neigh_hor_main[4], neigh_hor_main[5] ] )

    elif len( tmpKeyTxt.split() ) > 1 and len( tmpKeyTxt.split()[0] ) >= 1 and allNum( tmpKeyTxt.split()[0] ) and\
        not allNum( ' '.join( tmpKeyTxt.split()[1:] ), splitter_=True ):

      response_text_arr_.append( ' '.join( tmpKeyTxt.split()[1:] ) )
      ## horizontal arr needs to be modified ..most neigh remains the same just for the key, the right neigh is replaced with newly splity val and for the val left neigh is replaced with newly split key
      ## FOR KEY
      lt_feat_, _ = findWdFeats( ( tmpKeyTxt.split()[0] ) )
      response_hor_arr_.append( [ neigh_hor_main[0], neigh_hor_main[1], lt_feat_.tolist(), neigh_hor_main[3],\
                                  neigh_hor_main[4], neigh_hor_main[5] ] )

      ## FOR VAL
      response_text_arr_.append( ( tmpKeyTxt.split()[0] ) )
      rt_feat_, _ = findWdFeats( ' '.join( tmpKeyTxt.split()[1:] ) )
      response_hor_arr_.append( [ neigh_hor_main[0], neigh_hor_main[1], neigh_hor_main[2], rt_feat_.tolist(),\
                                  neigh_hor_main[4], neigh_hor_main[5] ] )

    return response_text_arr_ , response_hor_arr_

def loadLocal( folder_ ):

    ll_, vec_sz = os.listdir( folder_ ), 256
    inpCnt_ , inptCoOrds_ , labels_, actualInp = [], [], [], []
    master_ctr_idx_, master_coord_idx_, master_neigh_vert, master_neigh_hor = [], [], [], []
    fnList_ = []

    #for elem in ll_: 
    if True:
      elem = test_fnm + '.json'
      print('Processing file ->', elem)    

      with open( folder_ + elem, 'rb' ) as fp:
        np_arr_ = np.load( fp, allow_pickle=True )

      ## lets assume input vector size = 256
 
      locCnt_, locCoOrd_, locLabel_, neigh_axis0_size, neigh_axis1_size = [], [], [], 0, 0
      contour_index_ , coord_index_, actual_stuff, neigh_index_vert, neigh_index_hor = [], [], [], [], []

      for tuple_ in np_arr_:
        if len( tuple_ ) != 5: continue
        txt_main, coords_, labels, neigh_vert, neigh_hor_main = tuple_
        ## pre process 3 kinds of text
        '''
        Date: 23423 into "Date" and "234343"
        Date 27.8.23 in "Date" and "27.8.23"
        123 Premium paid into "123" and "Premium paid" 
        in all these cases, the "neigh_hor" is what will need to be manipulated    
        '''
        #txtArr_, nhorArr_ = _findRefinedTextFeats( txt_main, neigh_hor_main ) 
        if True:
        #if len( txtArr_ ) == 0:
          txtArr_, nhorArr_ = [ txt_main ], [ neigh_hor_main ]

        print('DGMI->Prev->', txt_main, labels, neigh_hor_main )
        print('POST->', txtArr_, nhorArr_ ) 
        
        for ctr in range( len(txtArr_) ): 
          txt_, neigh_hor = txtArr_[ ctr ], nhorArr_[ ctr ] 
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
          #normal_vert = np_mixed_vert.reshape( np_neigh_vert.shape[0]*np_neigh_vert.shape[1] )
          normal_vert = np_neigh_vert.reshape( np_neigh_vert.shape[0]*np_neigh_vert.shape[1] )

          np_neigh_hor = np.asarray( neigh_hor )
          np_mixed_hor = np.asarray( np.split( np_neigh_hor, np_neigh_hor.shape[0], axis=-1 ) )
          #normal_hor = np_mixed_hor.reshape( np_neigh_hor.shape[0]*np_neigh_hor.shape[1] )
          normal_hor = np_neigh_hor.reshape( np_neigh_hor.shape[0]*np_neigh_hor.shape[1] )
          print('Find wd feats->', txt_, coords_, feats_, labels, normal_hor, txt_.split(':'), txt_.split())


          print('POST->', normal_hor)
          neigh_index_vert.append( normal_vert )
          neigh_index_hor.append( normal_hor )

          coord_index_.append( co_ord_idx_ )
          master_coord_idx_.append( co_ord_idx_ )

          locCnt_.append( feats_ )
          locCoOrd_.append( coords_ + [ ( coords_[2] - coords_[0] ), ( coords_[3] - coords_[1] ) ] )
          locLabel_.append( label_maps_[ labels ] )    

      ## FOR LOOP BREAKS HERE 
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

    return inpCnt_, inptCoOrds_, labels_, \
                                        fnList_, actualInp, master_neigh_vert, master_neigh_hor


inpCnt_, inptCoOrds_, labels_, fnmList_, \
                            actualInp, master_neigh_vert, master_neigh_hor = loadLocal('./DATA_ENHANCED_KV/') 
#inpCnt_, inptCoOrds_, labels_, cntEmbInpDim_, coordEmbInpDim, fnmList_ = loadLocal('../DATA/') 
 
print( np.asarray( inpCnt_ ).shape, np.asarray( inptCoOrds_ ).shape, np.asarray( labels_ ).shape, \
                                                  np.asarray( master_neigh_vert ).shape   )
#exit()
num_samples_ = np.asarray( inpCnt_ ).shape[0]
val_dataset = tf.data.Dataset.from_tensor_slices(( (inpCnt_[-1*sampler_:], inptCoOrds_[-1*sampler_:], master_neigh_vert[-1*sampler_:], master_neigh_hor[-1*sampler_:] ), labels_[-1*sampler_:] ))

val_fnm_ = fnmList_[-1*sampler_:]
val_labels_ = labels_[-1*sampler_:]

val_acutal_ = actualInp[-1*sampler_:]

BATCH_SIZE, val_batch_sz = 32, sampler_
SHUFFLE_BUFFER_SIZE = 100

val_dataset = val_dataset.batch( val_batch_sz )

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


sample_input = val_dataset

loaded_model = keras.models.load_model( './NER_MODEL/model_three_embed_multi_neigh', \
        {'CustomNonPaddingTokenLoss': CustomNonPaddingTokenLoss }, compile=False )


def startWorkers( loaded_model, sample_input ):

    output = loaded_model.predict(sample_input)
    print('PREDN->', output.shape)

    file_ctr = 0 
    fnm = val_fnm_[ file_ctr ] 
    file_contours_ = output[ file_ctr ]

    print('Predns for filnm->', fnm)
    fp = open('./JSON_COMPLEX_NEIGH/_'+fnm, 'w+')

    matches_, tot, Keys_, Vals_, IRRs_ = 0, 0, [], [], []
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
      txtt, ptst = val_acutal_[ file_ctr ][contour_ctr]

      if 'KEY' in label_loc.split('-')[0]: Keys_.append( (txtt, ptst,label_loc)  )
      elif 'VALUE' in label_loc.split('-')[0]: Vals_.append( (txtt, ptst,label_loc) )
      else: IRRs_.append( (txtt, ptst,label_loc) )

    fp.write( 'MATCH CTR-> ||'+str(matches_)+'||'+str(tot) )
    fp.close()
    print('Match counter fr ', fnm, ' == ', matches_, tot)
    
    return Keys_, Vals_, IRRs_      


def unassigned( Keys_, Vals_, IRRs_, assigned_pairs_, ht_, wd_ ):
    ## fns ref - nothinInBetween, featNum

    for key_ctr in range( len(Keys_) ):
      tmpKeyTxt, tmpKeyCnt, tmpKeyLabel = Keys_[ key_ctr ]
      tmpKeyCnt = [ int( tmpKeyCnt[0]*wd_ ), int( tmpKeyCnt[1]*ht_ ), int( tmpKeyCnt[2]*wd_ ),\
                    int( tmpKeyCnt[-1]*ht_ ) ]
        
      assigned_ = False
      for ktup, vtup in assigned_pairs_:  
        if ktup[1] == tmpKeyCnt:
          assigned_ = True
          break

      if assigned_ is False:
        print('SEARCHING VAL FOR UNASS->', tmpKeyTxt, tmpKeyCnt, tmpKeyLabel)
        for tx, pt, _ in IRRs_:
          if type( pt ) is int: continue

          pt = [ int( pt[0]*wd_ ), int( pt[1]*ht_ ), int( pt[2]*wd_ ),int( pt[-1]*ht_ ) ]
          kf, irrf = featNum( tmpKeyTxt ), featNum( tx )
          print('#UNASS_CHK->', tx, pt, kf, irrf) 
          if irrf != kf and irrf in [1,2,3,5]:
            if xOverlap( tx, pt, tmpKeyTxt, tmpKeyCnt ) and nothinInBetween( tx, tmpKeyTxt, tmpKeyCnt, pt, IRRs_, 'VERTICAL', ht_, wd_, assigned_pairs_ ) and\
              nothinInBetween( tx, tmpKeyTxt, tmpKeyCnt, pt, Vals_, 'VERTICAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and\
              nothinInBetween( tx, tmpKeyTxt, tmpKeyCnt, pt, Keys_, 'VERTICAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and \
              pt[1] > tmpKeyCnt[1]:
         
              print('Assigning VERT UNASSIGNED ->', tx, ' To KEY ', tmpKeyTxt)
              paired_( ( ( tmpKeyTxt, tmpKeyCnt ), ( tx, pt ) ), assigned_pairs_  ) ## dont bother splitting co-ords

            elif abs( tmpKeyCnt[1] - pt[1] ) <= 10 and ( pt[0] >= tmpKeyCnt[2] or\
                                                       ( pt == tmpKeyCnt and tx != tmpKeyTxt ) ) and \
              nothinInBetween( tx, tmpKeyTxt, tmpKeyCnt, pt, IRRs_, 'HORIZONTAL', ht_, wd_, assigned_pairs_ ) and\
              nothinInBetween( tx, tmpKeyTxt, tmpKeyCnt, pt, Vals_, 'HORIZONTAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and\
              nothinInBetween( tx, tmpKeyTxt, tmpKeyCnt, pt, Keys_, 'HORIZONTAL', \
                                                                         ht_, wd_, assigned_pairs_ ):
         
              print('Assigning HORI UNASSIGNED ->', tx, ' To KEY ', tmpKeyTxt)
              paired_( ( ( tmpKeyTxt, tmpKeyCnt ), ( tx, pt ) ), assigned_pairs_  ) ## dont bother splitting co-ords

    for val_ctr in range( len(Vals_) ):
      tmpKeyTxt, tmpKeyCnt, tmpKeyLabel = Vals_[ val_ctr ]
      if type( tmpKeyCnt ) is int: continue
      tmpKeyCnt = [ int( tmpKeyCnt[0]*wd_ ), int( tmpKeyCnt[1]*ht_ ), int( tmpKeyCnt[2]*wd_ ),\
                    int( tmpKeyCnt[-1]*ht_ ) ]
      
      assigned_ = False
      for ktup, vtup in assigned_pairs_:  
        if vtup[1] == tmpKeyCnt:
          assigned_ = True
          break

      if assigned_ is False:
        for tx, pt, _ in IRRs_:
          if type( pt ) is int: continue

          pt = [ int( pt[0]*wd_ ), int( pt[1]*ht_ ), int( pt[2]*wd_ ),int( pt[-1]*ht_ ) ]
          kf, irrf = featNum( tmpKeyTxt ), featNum( tx )
          print('UNASSIGNED VALUE ->', tmpKeyTxt, tmpKeyCnt,' HUNTING FOR KEY IN IRR->', tx, pt, kf, irrf)
          if irrf != kf and irrf in [3,5]: ## only mixed and caps ..since its going to be KEY
            if xOverlap( tx, pt, tmpKeyTxt, tmpKeyCnt ) and nothinInBetween( tx, tmpKeyTxt, tmpKeyCnt, pt, IRRs_, 'VERTICAL', ht_, wd_, assigned_pairs_ ) and\
              nothinInBetween( tx, tmpKeyTxt, tmpKeyCnt, pt, Vals_, 'VERTICAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and\
              nothinInBetween( tx, tmpKeyTxt, tmpKeyCnt, pt, Keys_, 'VERTICAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and \
              pt[1] < tmpKeyCnt[1]:
         
              print('Assigning VERT UNASSIGNED ->', tx, ' To VAL ', tmpKeyTxt)
              paired_( ( ( tx, pt ), ( tmpKeyTxt, tmpKeyCnt ) ), assigned_pairs_  ) ## dont bother splitting co-ords

            elif abs( tmpKeyCnt[1] - pt[1] ) <= 10 and ( pt[2] < tmpKeyCnt[0] or \
                                                       ( pt == tmpKeyCnt and tx != tmpKeyTxt ) ) and \
              nothinInBetween( tx, tmpKeyTxt, tmpKeyCnt, pt, IRRs_, 'HORIZONTAL', ht_, wd_, assigned_pairs_ ) and\
              nothinInBetween( tx, tmpKeyTxt, tmpKeyCnt, pt, Vals_, 'HORIZONTAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and\
              nothinInBetween( tx, tmpKeyTxt, tmpKeyCnt, pt, Keys_, 'HORIZONTAL', \
                                                                         ht_, wd_, assigned_pairs_ ):
         
              print('Assigning HORI UNASSIGNED ->', tx, ' To VAL ', tmpKeyTxt)
              paired_( ( ( tx, pt ), ( tmpKeyTxt, tmpKeyCnt ) ), assigned_pairs_ ) ## dont bother splitting co-ords
      
def notPresent( txt, pts, df_ ):

    ## assumption that df_ is always going to be a list of 3 element tuples 
    for ref, rfcnt, _ in df_:
      if txt == ref and pts == rfcnt: return False

    return True

def _splitter( source, targetKeys, targetVals ):

    for tmpKeyTxt, tmpKeyCnt, _ in source:

      print('Trying to split->', tmpKeyTxt, ' & checks are ',allNum( tmpKeyTxt.split()[0] ), \
            allNum( ' '.join( tmpKeyTxt.split()[1:] ) ) , notPresent( tmpKeyTxt.split()[0], tmpKeyCnt, targetVals ) )
      if len( tmpKeyTxt.split() ) > 1 and len( tmpKeyTxt.split()[0] ) >= 1 and allNum( tmpKeyTxt.split()[0] ) and\
        not allNum( ' '.join( tmpKeyTxt.split()[1:] ), splitter_=True ) \
        and notPresent( tmpKeyTxt.split()[0], tmpKeyCnt, targetVals ):

          targetVals.append( ( tmpKeyTxt.split()[0], tmpKeyCnt, 'VALUE-KEY-LEFT' ) )
          targetKeys.append( ( ' '.join( tmpKeyTxt.split()[1:] ), tmpKeyCnt, 'KEY-VALUE-RIGHT' ) )
          print( 'Split ->', tmpKeyTxt, ' Into ||', tmpKeyTxt.split()[0], ' && ', ' '.join( tmpKeyTxt.split()[1:] ) )
   
def paired_( tuple_ , ds ):

    for ktup, vtup in ds:
      if ktup[0] == tuple_[0][0] and ktup[1] == tuple_[0][1]: return

    print('BOJI->ADDING->',tuple_)
    ds.append( tuple_ )

def generateResultImg( img_path, assigned_pairs_ ):

    urllib.request.urlretrieve( img_path, './STORE_IMG/'+img_path.split('/')[-1] ) 
    fnm_ = img_path.split('/')[-1]
    img_ = cv2.imread( './STORE_IMG/'+fnm_ )
    
    for ktup, vtup in assigned_pairs_:
      kpts_, vpts_ = ktup[1], vtup[1]
      cv2.rectangle( img_, ( kpts_[0], kpts_[1] ), ( kpts_[2], kpts_[3] ), ( 255, 0, 255 ), 3 ) 
      cv2.rectangle( img_, ( vpts_[0], vpts_[1] ), ( vpts_[2], vpts_[3] ), ( 0, 255, 0 ), 2 )

    cv2.imwrite( './STORE_IMG/_'+ fnm_, img_ )   

def finalResults( Keys_, Vals_, IRRs_, _fnm ):

    assigned_pairs_, unassigned_keys, unassigned_vals = [], [], []
    assi_key, assi_val = [], []

    with open( src_0 + _fnm, 'r' ) as fp:
      jsn_ = json.load( fp )

    ht_, wd_, img_path = jsn_['height'], jsn_['width'], jsn_['path']

    ## 
    ## need to separate KEY n VALs for ex insurance / close contours
    ## 123 Premium liability ..
    _splitter( Keys_, Keys_, Vals_ )
    _splitter( Vals_, Keys_, Vals_ )
    _splitter( IRRs_, Keys_, Vals_ )

    for key_ctr in range( len(Keys_) ):
      tmpKeyTxt, tmpKeyCnt, tmpKeyLabel = Keys_[ key_ctr ]
      tmpKeyCnt = [ int( tmpKeyCnt[0]*wd_ ), int( tmpKeyCnt[1]*ht_ ), int( tmpKeyCnt[2]*wd_ ),\
                    int( tmpKeyCnt[-1]*ht_ ) ]

      ## first check for conjoined KEY_V ( Date: 12/3/22 ; Invoice No. 12312 )
      if ':' in tmpKeyTxt and len( tmpKeyTxt.split(':')[-1] ) >= 3 and len( tmpKeyTxt.split(':')[0] ) >= 3:#\
        #and allNum( tmpKeyTxt.split(':')[-1] ):
        kt, vt = ' '.join( tmpKeyTxt.split(':')[:-1] ), tmpKeyTxt.split(':')[-1]
        print('MODEL Assigning CONJOINED ->', vt, ' To ', kt)

        paired_( ( ( kt, tmpKeyCnt ), ( vt, tmpKeyCnt ) ), assigned_pairs_  ) ## dont bother splitting co-ords
        continue

      elif len( tmpKeyTxt.split() ) > 1 and len( tmpKeyTxt.split()[-1] ) >= 3 and allNum( tmpKeyTxt.split()[-1] )\
        and len( ' '.join( tmpKeyTxt.split()[:-1] ) ) >= 3:
        kt, vt = ' '.join( tmpKeyTxt.split()[:-1] ), tmpKeyTxt.split(':')[-1]
        print('MODEL Assigning CONJOINED ->', vt, ' To ', kt)

        paired_( ( ( kt, tmpKeyCnt ), ( vt, tmpKeyCnt ) ), assigned_pairs_  ) ## dont bother splitting co-ords
        continue

 
      for val_ctr in range( len(Vals_) ):
        tmpValTxt, tmpValCnt , tmpValLabel = Vals_[ val_ctr ]
        if type( tmpValCnt ) == int: continue
        tmpValCnt = [ int( tmpValCnt[0]*wd_ ), int( tmpValCnt[1]*ht_ ), int( tmpValCnt[2]*wd_ ),\
                    int( tmpValCnt[-1]*ht_ ) ]

        if 'RIGHT' in tmpKeyLabel and 'LEFT' in tmpValLabel and \
          abs( tmpKeyCnt[1] - tmpValCnt[1] ) < 20 and tmpValCnt[0] > tmpKeyCnt[2] and\
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, Vals_, 'HORIZONTAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and\
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, Keys_, 'HORIZONTAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and \
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, IRRs_, 'HORIZONTAL', \
                                                                         ht_, wd_, assigned_pairs_ ):
          print('MODEL Assigning->', tmpValTxt, ' To ', tmpKeyTxt)

          paired_( ( ( tmpKeyTxt, tmpKeyCnt ), ( tmpValTxt, tmpValCnt ) ), assigned_pairs_  )
          continue

        if 'BELOW' in tmpKeyLabel and 'TOP' in tmpValLabel and \
          xOverlap( tmpValTxt, tmpValCnt, tmpKeyTxt, tmpKeyCnt ) and abs( tmpValCnt[1] - tmpKeyCnt[-1] ) <= 100\
          and tmpValCnt[1] > tmpKeyCnt[1] and\
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, Vals_, 'VERTICAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and\
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, Keys_, 'VERTICAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and \
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, IRRs_, 'VERTICAL', \
                                                                         ht_, wd_, assigned_pairs_ ):
          print('MODEL Assigning->', tmpValTxt, ' To ', tmpKeyTxt)

          paired_( ( ( tmpKeyTxt, tmpKeyCnt ), ( tmpValTxt, tmpValCnt ) ), assigned_pairs_  )
          continue
 
      for val_ctr in range( len(Keys_) ):
        tmpValTxt, tmpValCnt , tmpValLabel = Keys_[ val_ctr ]
        keyfeat, valfeat = featNum( tmpKeyTxt ), featNum( tmpValTxt )

        if type( tmpValCnt ) == int: continue
        tmpValCnt = [ int( tmpValCnt[0]*wd_ ), int( tmpValCnt[1]*ht_ ), int( tmpValCnt[2]*wd_ ),\
                    int( tmpValCnt[-1]*ht_ ) ]

        if keyfeat != valfeat and \
          abs( tmpKeyCnt[1] - tmpValCnt[1] ) < 20 and tmpValCnt[0] > tmpKeyCnt[2] and\
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, Vals_, 'HORIZONTAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and\
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, Keys_, 'HORIZONTAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and \
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, IRRs_, 'HORIZONTAL', \
                                                                         ht_, wd_, assigned_pairs_ ):
          print('MODEL Assigning->', tmpValTxt, ' To ', tmpKeyTxt)

          paired_( ( ( tmpKeyTxt, tmpKeyCnt ), ( tmpValTxt, tmpValCnt ) ), assigned_pairs_  )
          continue

        if keyfeat != valfeat and \
          xOverlap( tmpValTxt, tmpValCnt, tmpKeyTxt, tmpKeyCnt ) and abs( tmpValCnt[1] - tmpKeyCnt[-1] ) <= 100\
          and tmpValCnt[1] > tmpKeyCnt[1] and\
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, Vals_, 'VERTICAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and\
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, Keys_, 'VERTICAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and \
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, IRRs_, 'VERTICAL', \
                                                                         ht_, wd_, assigned_pairs_ ):
          print('MODEL Assigning->', tmpValTxt, ' To ', tmpKeyTxt)

          paired_( ( ( tmpKeyTxt, tmpKeyCnt ), ( tmpValTxt, tmpValCnt ) ), assigned_pairs_  )
          continue

      if True: # laziness to indent the below
        print('Trying to find a natural match within the TAGGED community for ->', tmpKeyTxt)
 
        for val_ctr in range( len(Vals_) ):
          tmpValTxt, tmpValCnt , tmpValLabel = Vals_[ val_ctr ]
          if type( tmpValCnt ) == int: continue
          tmpValCnt = [ int( tmpValCnt[0]*wd_ ), int( tmpValCnt[1]*ht_ ), int( tmpValCnt[2]*wd_ ),\
                    int( tmpValCnt[-1]*ht_ ) ]

          print('EVALUIN->', tmpValTxt, tmpValCnt , tmpKeyCnt, tmpValLabel) 
          #if 'RIGHT' in tmpKeyLabel and 'TOP' in tmpValLabel and \
          if True and \
            abs( tmpKeyCnt[1] - tmpValCnt[1] ) <= 10 and ( tmpValCnt[0] > tmpKeyCnt[0] or tmpValCnt == tmpKeyCnt ) and \
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, Vals_, 'HORIZONTAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and\
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, Keys_, 'HORIZONTAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and \
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, IRRs_, 'HORIZONTAL', \
                                                                         ht_, wd_, assigned_pairs_ ):
            print('MODEL Assigning->', tmpValTxt, ' To ', tmpKeyTxt)

            paired_( ( ( tmpKeyTxt, tmpKeyCnt ), ( tmpValTxt, tmpValCnt ) ), assigned_pairs_  )
            found_partner_ = True

          #if 'BELOW' in tmpKeyLabel and 'LEFT' in tmpValLabel and \
          if True and \
            xOverlap( tmpValTxt, tmpValCnt, tmpKeyTxt, tmpKeyCnt ) and abs( tmpValCnt[1] - tmpKeyCnt[-1] ) <= 100\
            and tmpValCnt[1] > tmpKeyCnt[1] and\
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, Vals_, 'VERTICAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and\
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, Keys_, 'VERTICAL', \
                                                                         ht_, wd_, assigned_pairs_ ) and \
            nothinInBetween( tmpKeyTxt, tmpValTxt, tmpValCnt, tmpKeyCnt, IRRs_, 'VERTICAL', \
                                                                         ht_, wd_, assigned_pairs_ ):

            print('MODEL Assigning->', tmpValTxt, ' To ', tmpKeyTxt)

            paired_( ( ( tmpKeyTxt, tmpKeyCnt ), ( tmpValTxt, tmpValCnt ) ), assigned_pairs_  )
            found_partner_ = True

    ## now handle unassigned KEYS and then unassigned VALS using IRR 
    print('Just before heading to backup->')
    print('KEYS')
    for elem in Keys_: print( elem )
    print('VALS')
    for elem in Vals_: print( elem )
    print('IRRS')
    for elem in IRRs_: print( elem )

    unassigned( Keys_, Vals_, IRRs_, assigned_pairs_, ht_, wd_ )
    ## finally fill in blanks by finding sandwiched KEY-VAL pairs
    
    for keytup, valtup in assigned_pairs_:
      print('KEY->', keytup, ' || VAL->', valtup) 

    generateResultImg( img_path, assigned_pairs_ )
 
if __name__ == "__main__":

    Keys_, Vals_, IRRs_ = startWorkers( loaded_model, sample_input )  
    print('DODO KEY->', Keys_)
    print('DODO VAL->', Vals_)
    print('DODO IRR->', IRRs_)
    finalResults( Keys_, Vals_, IRRs_, test_fnm+'.json' ) 
