import ctypes
NUM = 16
# libfun loaded to the python file
# using fun.myFunction(),
# C function can be accessed
# but type of argument is the problem.

# Or full path to file
fun = ctypes.CDLL(
    "C:\\Users\\Matt\\Documents\\Cambridge\\Part III\\Project\\Part-III-Project\\libfun.so")
# Now whenever argument
# will be passed to the function
# ctypes will check it.

fun.myFunction.argtypes = [ctypes.c_int]

# now we can call this
# function using instant (fun)
# returnValue is the value
# return by function written in C
# code
returnVale = fun.myFunction(NUM)
print(returnVale)
