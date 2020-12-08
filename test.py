import sys, getopt
input = sys.argv[1:]
x = getopt.getopt(input,"",["epochs=","name="])
print(x)