
''' Conditional filtering of a list: in simple words, 
the filter() method filters the given iterable with the help of 
a function that tests each element in the iterable to be true or not.'''

numbers = [1, 6, 3, 8, 4, 9]

list(filter(lambda x: x<5, numbers))

#get first index {match in R} 
numbers.index['3']

#Comprehensive list of indices which matches a value
[ i for i in range(len(numbers)) if numbers[i] == 1 ]


#Get the list items based on custom indices
[numbers[p[i]] for i in range(len(p))]

----------------------------

#Reverse a list 

nums[::-1]

------------------------------

