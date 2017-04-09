import cPickle as pickle

my_dict = {"tyro":"beginner", "cacophony":"harsh/unpleasant sounds"}
my_set = {"ayush","soni","akshat","vora"}

pickle.dump(my_dict, open("dict.p","wb"))
pickle.dump(my_set, open("set.p","wb"))
pickle.dump(a, open("var.p","wb"))

#cPickle for any Python 2.x while just Pickle for 3.x
#objects are stores as unicode.
#Not only does it store the data, but also the data structure along with it
#to test, run the following command on terminal, when current directory is ../pickle:
#"python makesPickle.py && python takesPickle.py"
