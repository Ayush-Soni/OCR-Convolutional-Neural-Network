import cPickle as pickle

my_dict = pickle.load(open("dict.p","rb"))
my_set = pickle.load(open("list.p","rb"))
a = pickle.load(open("var.p","rb"))

print my_dict
print my_set
print a
