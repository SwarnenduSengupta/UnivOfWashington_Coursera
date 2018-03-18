import operator
import sframe                            # see below for install instruction
import matplotlib.pyplot as plt          # plotting
import numpy as np                       # dense matrices
from scipy.sparse import csr_matrix      # sparse matrices
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances

wiki = sframe.SFrame('people_wiki.gl/')
wiki = wiki.add_row_number()             # add row number, starting at 0
#print "Wiki head"
#print wiki.head(2)
#print "next"
#s1,s2,s3,s4 = zip(*wiki)
#print np.size(np.unique(wiki['name']))
def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    
    return csr_matrix( (data, indices, indptr), shape)

word_count = load_sparse_csr('people_wiki_word_count.npz')
map_index_to_word = sframe.SFrame('people_wiki_map_index_to_word.gl/')

model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)
#print wiki[wiki['name'] == 'Barack Obama']
distances, indices = model.kneighbors(word_count[35817], n_neighbors=10) # 1st arg: word count vector
#print type(indices)

neighbors = sframe.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
#print wiki.join(neighbors, on='id').sort('distance')[['id','name','distance']]

def unpack_dict(matrix, map_index_to_word):
    table = list(map_index_to_word.sort('index')['category'])
    # if you're not using SFrame, replace this line with
    ##      table = sorted(map_index_to_word, key=map_index_to_word.get)
    
    
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr
    
    num_doc = matrix.shape[0]

    return [{k:v for k,v in zip([table[word_id] for word_id in indices[indptr[i]:indptr[i+1]] ],
                                 data[indptr[i]:indptr[i+1]].tolist())} \
               for i in xrange(num_doc) ]

wiki['word_count'] = unpack_dict(word_count, map_index_to_word)
#print wiki.head(10)
#print (wiki["word_count"][1])
# for printing the words according to frequency in reverse
name='Barack Obama'
row = wiki[wiki['name'] == name]
#print type(row['word_count'][0])
#for key, value in sorted(row['word_count'][0].iteritems(), key=lambda (k,v): (v,k),reverse=True):
#    print "%s: %s" % (key, value)
# to print the top 5 key value pair
wrd = sorted(row['word_count'][0].items(), key=operator.itemgetter(1),reverse=True)
word_for_comparison = wrd[:5]
#print wrd[0:10]
#print type(word_for_comparison)
#print word_for_comparison[0][0:5]
#print type(word_for_comparison[0][0:5])


def top_words(name):
    """
    Get a table of the most frequent words in the given person's wikipedia page.
    """
    row = wiki[wiki['name'] == name]
    word_count_table = row[['word_count']].stack('word_count', new_column_name=['word','count'])
    return word_count_table.sort('count', ascending=False)

obama_words = top_words('Barack Obama')
#print obama_words

barrio_words = top_words('Francisco Barrio')
#print type(barrio_words)

combined_words = obama_words.join(barrio_words, on='word')
combined_words = combined_words.rename({'count':'Obama', 'count.1':'Barrio'})
combined_words.sort('Obama', ascending=False)
#print combined_words
#print combined_words["word"][0:5]

common_words=combined_words["word"][0:5]
common_words = set(common_words)
#print type(common_words)

def has_top_words(word_count_vector):
    kys=set(word_count_vector.keys())
    return common_words.issubset(kys)

def num_uniq_wrds(word_count_vector):
    kys=set(word_count_vector.keys())
    return len(kys)

wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)
wiki['len_uniq_words'] = wiki['word_count'].apply(num_uniq_wrds)
wiki_true = wiki[wiki['has_top_words'] == 1]
wiki_false = wiki[wiki['has_top_words'] == 0]
print "have top words",wiki_true.num_rows()
print "no top words present",wiki_false.num_rows()

#print wiki.head(10)
#print 'Output from your function:', has_top_words(wiki[32]['word_count'])
#print 'Output from your function:', num_uniq_wrds(wiki[32]['word_count'])
brk_obama = wiki[wiki['name'] == "Barack Obama"]
bush = wiki[wiki['name'] == "George W. Bush"]
biden = wiki[wiki['name'] == "Joe Biden"]
#print brk_obama
#print brk_obama["id"][0]
#print bush
#print type(bush["id"][0])
#print biden
#print biden["id"][0]
#print brk_obama['word_count'][0]
dist_obama_bush = euclidean_distances(word_count[brk_obama["id"][0]], word_count[bush["id"][0]])
dist_bush_biden = euclidean_distances(word_count[bush['id'][0]], word_count[biden['id'][0]])
dist_biden_obama = euclidean_distances(word_count[biden['id'][0]], word_count[brk_obama['id'][0]])
print "bush obama",dist_obama_bush
print "bush biden",dist_bush_biden
print "biden obama",dist_biden_obama

obama_words = top_words('Barack Obama')
#print obama_words
bush_words = top_words('George W. Bush')
#print bush_words

combined_words = obama_words.join(bush_words, on='word')
combined_words = combined_words.rename({'count':'Obama', 'count.1':'Bush'})
combined_words =combined_words.sort('Obama', ascending=False)#print_rows(num_rows = 10)
print combined_words["word"][0:10]

# next is tf-idf
tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')
wiki['tf_idf'] = unpack_dict(tf_idf, map_index_to_word)
#print tf_idf[0:1][1:10]
model_tf_idf = NearestNeighbors(metric='euclidean', algorithm='brute')
model_tf_idf.fit(tf_idf)
distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=10)
neighbors = sframe.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
print wiki.join(neighbors, on='id').sort('distance')[['id', 'name', 'distance']]

#top words from tf_idf
def top_words_tf_idf(name):
    row = wiki[wiki['name'] == name]
    word_count_table = row[['tf_idf']].stack('tf_idf', new_column_name=['word','weight'])
    return word_count_table.sort('weight', ascending=False)

obama_tf_idf = top_words_tf_idf('Barack Obama')
#name='Barack Obama'
#row = wiki[wiki['name'] == name]
#print row
#print obama_tf_idf

schiliro_tf_idf = top_words_tf_idf('Phil Schiliro')
#print schiliro_tf_idf

combined_words = obama_tf_idf.join(schiliro_tf_idf, on='word')
combined_words = combined_words.rename({'weight':'Obama', 'weight.1':'Schiliro'})
combined_words =combined_words.sort('Obama', ascending=False)#print_rows(num_rows = 10)
#print combined_words["word"][0:10]

common_if_idf = set(combined_words["word"][0:5])
def has_top_words_if_idf(word_count_vector):
    kys=set(word_count_vector.keys())
    return common_if_idf.issubset(kys)

def num_uniq_wrds_if_idf(word_count_vector):
    kys=set(word_count_vector.keys())
    return len(kys)

wiki['has_top_words_if_idf'] = wiki['word_count'].apply(has_top_words_if_idf)
#wiki['len_uniq_words_if_idf'] = wiki['word_count'].apply(num_uniq_wrds_if_idf)
#print wiki

wiki_true = wiki[wiki['has_top_words_if_idf'] == 1]
wiki_false = wiki[wiki['has_top_words_if_idf'] == 0]
print "have top words if-idf",wiki_true.num_rows()
print "no top words present if-idf",wiki_false.num_rows()

dist_obama_bush_tf_idf = euclidean_distances(tf_idf[brk_obama["id"][0]], tf_idf[bush["id"][0]])
dist_bush_biden_tf_idf = euclidean_distances(tf_idf[bush['id'][0]], tf_idf[biden['id'][0]])
dist_biden_obama_tf_idf = euclidean_distances(tf_idf[biden['id'][0]], tf_idf[brk_obama['id'][0]])
print "bush obama if-idf",dist_obama_bush_tf_idf
print "bush biden if-idf",dist_bush_biden_tf_idf
print "obama biden if-idf",dist_biden_obama_tf_idf

# Comptue length of all documents
def compute_length(row):
    return len(row['text'].split(' '))
wiki['length'] = wiki.apply(compute_length)

# Compute 100 nearest neighbors and display their lengths
distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=100)
neighbors = sframe.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
nearest_neighbors_euclidean = wiki.join(neighbors, on='id')[['id', 'name', 'length', 'distance']].sort('distance')
print nearest_neighbors_euclidean

plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nearest_neighbors_euclidean['length'], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.axvline(x=wiki['length'][wiki['name'] == 'Barack Obama'][0], color='k', linestyle='--', linewidth=4,
           label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki['length'][wiki['name'] == 'Joe Biden'][0], color='g', linestyle='--', linewidth=4,
           label='Length of Joe Biden', zorder=1)
plt.axis([0, 1000, 0, 0.04])

plt.legend(loc='best', prop={'size':15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size':16})
plt.tight_layout()
#plt.show()

model2_tf_idf = NearestNeighbors(algorithm='brute', metric='cosine')
model2_tf_idf.fit(tf_idf)
distances, indices = model2_tf_idf.kneighbors(tf_idf[35817], n_neighbors=100)
neighbors = sframe.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
nearest_neighbors_cosine = wiki.join(neighbors, on='id')[['id', 'name', 'length', 'distance']].sort('distance')
print nearest_neighbors_cosine

plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nearest_neighbors_euclidean['length'], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.hist(nearest_neighbors_cosine['length'], 50, color='b', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (cosine)', zorder=11, alpha=0.8)
plt.axvline(x=wiki['length'][wiki['name'] == 'Barack Obama'][0], color='k', linestyle='--', linewidth=4,
           label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki['length'][wiki['name'] == 'Joe Biden'][0], color='g', linestyle='--', linewidth=4,
           label='Length of Joe Biden', zorder=1)
plt.axis([0, 1000, 0, 0.04])
plt.legend(loc='best', prop={'size':15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
#plt.show()

# lets compare with tweet data
tweet = {'act': 3.4597778278724887,
 'control': 3.721765211295327,
 'democratic': 3.1026721743330414,
 'governments': 4.167571323949673,
 'in': 0.0009654063501214492,
 'law': 2.4538226269605703,
 'popular': 2.764478952022998,
 'response': 4.261461747058352,
 'to': 0.04694493768179923}

word_indices = [map_index_to_word[map_index_to_word['category']==word][0]['index'] for word in tweet.keys()]
tweet_tf_idf = csr_matrix( (list(tweet.values()), ([0]*len(word_indices), word_indices)),shape=(1, tf_idf.shape[1]) )

#cosine distance calculation
obama_tf_idf = tf_idf[35817]
print cosine_distances(obama_tf_idf, tweet_tf_idf)
distances, indices = model2_tf_idf.kneighbors(obama_tf_idf, n_neighbors=10)
print distances,indices

