# UnsupervisedKVExtraction
Unsupervised entity extraction from documents of all categories

File descriptions
------------
findKeys.py - generate training data ( some details provided in the file comments )

NERPassThreeEmbed_MULTI_NEIGH.py - NER training model (with transformer encoders )

singlePass.py - inference ( along with some additional pre processing for the contours ..for e.g. words like Invoice : 12312 are better split into 2 separate contours using the ":" delim and feature vectors are created appropriately

Theory
-----------

Labelling entities like KEYS and their VALUES along with drawing contours etc is a massively expensive exercise because of the manual labour involved along with higher chances of errors in the labelling process itself. So the idea here is to let the data tell you whats important. We used a simple hack wherein we picked up contours from the OCR extracted 
- go through every text extracted and isolate those that are pure text contours
- once these are found, filter them using a simple rule. The contour to its RIGHT or BELOW must be of a different data type and preferably a NUMERIC / ALPHANUMERIC value
- if you have gone through documents you will realize that , irrespective of the type of doc, KEYs and VALUEs more or less follow the above pattern
- once we find these, we create a simple feature vector for these contours and also create a neighbourhood embeddding using the very same features but choosing contours around the the entity of interest 
- we also run a round of KMEANS clustering ( not covered in the code shared here ) to ensure that most of these KEYs ( as defined by the feature vectors ) end up in the same cluster.
- now that we have the feature vector for every contour ( along with its neighbourhood ) we label every contour with 5 types of labels , key-value-right, key-value-below, value-key-top, value-key-left and IRR ..we label both potential KEYs and VALUEs
- we convert this into an NER ( name entity recognition ) problem and expose an entire document to a simple transformer encoder (with MHA ) .. please feel free to play around with the hyper params to see if u get better precision but our val loss , after 40 epochs was around 0.3
- post that we ran the inference on about 500 files and the F1 score ( accounting for FPs and FNs ) was around 90% and this included bank statements, insurance copies, invoices etc
- we are also sharing sample input files and sample output files ( pink indicates KEY and green indicates VALS ) 
- as u can see in the results its not bullet proof BUT we can a) improve the model with better feature engg and b) add some scaffolding around the results to ensure the results are more less 100% complete
- sadly most of the data we use is under NDA from our customers so i can only share what i can ..but once u gather ur data u ll know what to do 
