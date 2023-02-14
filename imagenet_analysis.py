import os


imagenet_dir = "/home/krchinta/research/dataset/ImageNet/raw_data/train/"
classes = os.listdir(imagenet_dir)

synset_labels = open("/home/krchinta/research/dataset/ImageNet/raw_data/synset_labels.txt", "r")


# load the is-a file as a dictionary
is_a = open("/home/krchinta/research/dataset/ImageNet/raw_data/wordnet.is_a.txt","r")
parent_child = dict()
for mapping in is_a:
    fields = mapping.split(' ')
    parent = fields[0].strip()
    child = fields[1].strip()
    # print(parent)
    # print(child)
    parent_child[child] = parent
# print(parent_child)

# load the words and synset label mappping
words = open("/home/krchinta/research/dataset/ImageNet/raw_data/words.txt","r")

words_dic = dict()
for w in words:
    fields = w.split('\t')
    id = fields[0].strip()
    label = fields[1].strip()
    words_dic[id] = label


superclass = dict()

# for c in synset_labels:
for c in classes:
    c = c.strip()
    intermediate_parent = [c]
    while intermediate_parent[-1] in parent_child:
        intermediate_parent.append(parent_child[intermediate_parent[-1]])
    
    # there is at least 2 levels, since every class of the 1000 belongs to n00001740 (entity)
    superclass_level = max(-6,-len(intermediate_parent)) 
    if intermediate_parent[superclass_level] in superclass:
        superclass[intermediate_parent[superclass_level]].append(c)
    else:
        superclass[intermediate_parent[superclass_level]] = [c]

log_sc = open("/home/krchinta/research/dataset/ImageNet/raw_data/superclasses.txt","w")
log_c = open("/home/krchinta/research/dataset/ImageNet/raw_data/subclasses.txt","w")

for sc in superclass:
    log_sc.write(sc + "\t" + words_dic[sc] + "({} subclasses);\n".format(len(list(superclass[sc]))))
    log_c.write(str(superclass[sc]) + "\n")
log_sc.close()
log_c.close()

is_a.close()
synset_labels.close()
