#Weaks - dictionaries ,iterators, and zips(Generatrs),scope, lambda , map,filter,reduce, importing local scripts and 
# name == --main--

## if name == __main__ is put as a practice to executeif it is called directly as a scirpt
# it will not run when importing from other code



## Pandas

'''
A Pandas series is a one-dimensional array-like object that can hold many data types, such as numbers or strings. 
One of the main differences between Pandas Series and NumPy ndarrays is that you can assign an index label to each element in the Pandas Series.
 Another big difference between Pandas Series and NumPy ndarrays is that Pandas Series can hold data of different data types.
'''

#e.g
gorceries = pd.Series(data = (30,6,['Yes','No'],index = ['eggs','apples','milk','bread'])

#Series methods
groceries.shape
groceries.size
groceries.ndim
groceries.index
groceries.values

'banana' in groceries

## Accessing Series 

groceries['eggs']
groceries[['milk','eggs']]
groceries[0],groceries[[0,1]]

groceries.loc[['label1','label2']]
groceries.iloc[[1,2]]

#change 
groceries['eggs'] = 2
groceries.drop('eggs',inplace = True)

#note that we can also apply the numpy functions over pandas

np.sqrt(groceries)
np.min(),max,etc...

#Pandas dataframe , 2-d object with 
pd.DataFrame()
df.index
df.columns
df.values
df.ndim
df.shape
df.size # gives total number of items in the dataframe

#Load selected columns from dataframe
df2 = pd.DataFrame(items, index = ['flower1','flower2'],columns = ['Bob'])

#creating dictionary

#List of dictionary

# We create a list of Python dictionaries
items2 = [{'bikes': 20, 'pants': 30, 'watches': 35}, 
          {'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5}]

# We create a DataFrame 
store_items = pd.DataFrame(items2)

## Accessing rows

df.loc[['label']]
df['labelrow']['labelcolumn']
df[['label1','label2']]

# New column 

df['new_row'] = df['row1'] + df['row2']
df['new_column'] = df['existing_column'][1:] ## Doesn't shifts, puts Nas in the missing 0th row
df.insert(5,'label',[8,5,0])   #index , values

#New row

df = df.append(new_store)


#Remove columns
df = df.pop('new_watches')
df = df.drop(['watches','shoes'],axis = 1)
df = df.drop(['label','label2'],axis = 0) #drops rows

#Rename rows and columns

df = df.rename(columns = {'old':'new'})
df = df.rename(index = {'old':'new'})

#Set index

df = df.set_index('pants')

df.count()
df.isnull().sum().sum()  # first sum counts vs columns, second sum sums all the column nulls as well


## drop name

df.dropna(axis=0,inplace = True)
df.dropna(axis=1,inplace = True)

## fill values

df.fillna(0)
df.fillna(method = 'ffill', axis = 0) # fill from previous value
df.fillna(method = 'backfill', axis = 0) #fill from back 
df.interpolate(method = 'linear', axis = 0)  # along rows

# extracting values
best_rated = book_ratings[(book_ratings == 5).any(axis = 1)]['Book Title'].values


#read data

google = pd.readcsv('')
google.head()
google.tail()
google.isnull().any() # Gives True false for columns if
google.describe()
google['label'].describe()
google.corr()

 
#Grouping data

google.groupby(['Year'])['Salary'].sum() 
google.groupby(['Year','Department'])['Salary'].sum() 

 
#Dictionary of lists

items = {'Bob' : pd.Series(data = [245, 25, 55], index = ['bike', 'pants', 'watch']),
         'Alice' : pd.Series(data = [40, 110, 500, 45], index = ['book', 'glasses', 'bike', 'pants'])}

shopping_carts = pd.DataFrame(items)

## NUMPY

#USeful functions of generating numpy arraays

X = np.zeroes((3,4),dtype = int)
X = np.ones((3,4),dtype = int)
X = np.full((3,4),5) # creates a 3X4 array with constant as 5
X = np.eye(5)
X = np.diag([10,20,30])
X = np.arange(start,stop,step) #Creates a 1d array
X = np.delete(X,2,3) #deletes first and 3rd item
X = np.delete(X,[0,2],axis = 1) #deletes first and 3rd column
X = np.linspace(start,stop,step,endpoint = False)
X = np.reshape(nparray,specified shape), X = np.reshape(nparray,(2,3))

#appending a matrix
X = np.append(X,[[0,1,2]],axis = 0) #add a new row  
X = np.append(X,[[0],[1],[2]], axis = 1)# add new column

X = np.unique(X)

'''When slicing elemnts of a matrix, we never make a copy of them'''

z = X[:,2]
z[1,2] = 1 # Changes X as well.
 
#Remedy? Use numpy copy
z = np.copy(X[:,2])


#Insert

Y = np.array([[1,2,3],[7,8,9]])
W = np.insert(Y,1,[4,5,6],axis = 0)


#appending matrices

X = np.hstack((X,Y))
Y = np.vstack((X,Y))

#chaining methods

Y = np.arange(20).reshape((10,2))

#np.Random module

X = np.random.random((3,3))
X = np.random.randint(4,15,(3,3)) # lower, upper , shape
#Creates 3X3 matrix with values between 0 and 1 

X = np.random.normal(0,0.1,size = (1000,1000)) # mean = 0 , std = 0.1


np.diag(X) #extracts diagonals from X
np.diag(X,k=1) #extracts the diagonal +1 above 


#boolean indexing 
X[(X>2) &(x<7)]


#Numpy union intersection , 

np.intersect1d(x,y)
np.setdiff1d(x,y)
np.union1d(x,y)




'''#Numpy Sort'''

#When sort is used as a function, it doesn't changes array , 
np.sort(x)
#doesn't changes x


#When sort is used as a method, it changes array
x.sort()

np.sort(X,axis = 0) # Note, this sorts columns
np.sort(X,axis = 1) # Note, this sorts rows
np.sqrt(x)
np.exp(x)
np.power(x,2)


#Average of all elements in a matrix
X.mean()
X.mean(axis = 0)
X.sum(), max,min, median, 
## Broadcasting, how numpy handles elementwise operations with arrays of different shapes.
## Arrays to be operated on must have same shape or broadcastable
##e.g. X is a 2X2 array
X/3, X*2 is numpy doing things behind

#another eg X is 3X3, y is 1X3
X+y is expanding y to 3X3 and adding. That means adding each element to each column









''' start from bash and code python '''



###

# function that creates a flower_dictionary from filename
def create_flowerdict(filename):
    flower_dict = {}
    with open(filename) as f:
        for line in f:
            letter = line.split(": ")[0].lower() 
            flower = line.split(": ")[1].strip()
            flower_dict[letter] = flower
    return flower_dict

# Main function that prompts for user input, parses out the first letter
# includes function call for create_flowerdict to create dictionary
def main(): 
    flower_d = create_flowerdict('flowers.txt')
    full_name = input("Enter your First [space] Last name only: ")
    first_name = full_name[0].lower()
    first_letter = first_name[0]
# print command that prints final input with value from corresponding key in dictionary
    print("Unique flower name with the first letter: {}".format(flower_d[first_letter]))

main()

### REquirements.txt


'''

Using a requirements.txt File
Larger Python programs might depend on dozens of third party packages. 
To make it easier to share these programs, programmers often list a project's 
dependencies in a file called requirements.txt. This is an example of a requirements.txt file.


'''


## Run scriptin spyder
runfile('demo.py',args='one two three')

'''
Using a main block
To avoid running executable statements in a script when it's
 imported as a module in another script, include these lines in an if __name__ == "__main__" block. Or alternatively, 
 include them in a function called main() and call this in the if main block.

Whenever we run a script like this, Python actually sets a special built-in variable
 called __name__ for any module. When we run a script, Python recognizes this module as the main program, 
 and sets the __name__ variable for this module to the string "__main__". For any modules that are imported in this script, this built-in __name__ variable is just set to the name of that module. Therefore, the condition if __name__ == "__main__"is just checking whether this module is the main program.

'''

### Read a file

f = open('Git commands.txt','r')
file_data = f.read()
f.close()

print(file_data)

''' The object Reads one character at a time '''

### Write a file

f = open('new_file.txt','w')
f.write('Hello World!')
f.close()



##  automatic close

with open('new_file.txt','r') as f:
    file_data = f.read()    
print(file_data)

## Fetch lines of a file to a list

git_lines = []
with open("Git commands.txt") as f:
    for line in f:
        git_lines.append(line.strip())

print(git_lines)

#Scripting Example 

names = input("Enter names separated by commas: ").title().split(",")
assignments = input("Enter assignment counts separated by commas: ").split(",")
grades = input("Enter grades separated by commas: ").split(",")

message = "Hi {},\n\nThis is a reminder that you have {} assignments left to \
submit before you can graduate. You're current grade is {} and can increase \
to {} if you submit all assignments before the due date.\n\n"

for name, assignment, grade in zip(names, assignments, grades):
    print(message.format(name, assignment, grade, int(grade) + int(assignment)*2))


#Scripting Example 2 - Taking input

while True:
    try:
        x = int(input('Enter a number: '))
        break
    except ValueError:
        print('That\'s not a valid number!')
    except KeyboardInterrupt:
        print('\n No input taken')
        break
    finally: # prints every time irrespective of error or not
        print('\n Attempted Input \n')
        
        
#Access error message

try:
    # some code
except Exception as e:
   # some code
   print("Exception occurred: {}".format(e))
   




### Map

#### map() is a higher-order built-in function that takes a function and iterable
# as inputs, and returns an iterator that applies the function to each element of the iterable.


numbers = [
              [34, 63, 88, 71, 29],
              [90, 78, 51, 27, 45],
              [63, 37, 85, 46, 22],
              [51, 22, 34, 11, 18]
           ]

def mean(num_list):
    return sum(num_list) / len(num_list)

averages = list(map(mean, numbers))
print(averages)



###Filter

##filter() is a higher-order built-in function that takes a function and iterable as inputs and returns an iterator with the
### elements from the iterable for which the function returns True

cities = ["New York City", "Los Angeles", "Chicago", "Mountain View", "Denver", "Boston"]

short_cities = list(filter(lambda name: len(name)<10, cities))
print(short_cities)


# Generators are simple functions which return an iterable set of items, one at a time, in a special way.   functions that return it

e.g. yield returns one element at a time and continuing from where it was called

def my_range(x):
    i = 0
    while i<x:
        yield i
        i+=1
        
for a in my_range(4):
    print(n)

# Generate an iterator in list comprehension  
sq_iterator = (x**2 for x in range(10))  # this produces an iterator of squares

#Iterator is an object that represents stream of data
# an iterable returns one object at a time , eg. list 

import random

def lottery():
    # returns 6 numbers between 1 and 40
    for i in range(6):
        yield random.randint(1, 40)

    # returns a 7th number between 1 and 15
    yield random.randint(1,15)

for random_number in lottery():
       print("And the next number is... %d!" %(random_number))
       
      


#3//2 gives integer

#.format

animal = "dog"
action = "bite"
print("Does your {} {}?".format(animal, action))


# Join components of list through de-limiter mentioned

names = ["Carol", "Albert", "Ben", "Donna"]
print(" & ".join(sorted(names)))


# Sets are unordered , dict are unordered

a = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
b = set(a)
b.add(5)
b.pop()

# dictionary 

>>> elements.get('dilithium')
None
>>> elements['dilithium']
KeyError: 'dilithium'
>>> elements.get('kryptonite', 'There\'s no such element!')
"There's no such element!"



### Nested Dictionaries

elements = {'hydrogen': {'number': 1, 'weight': 1.00794, 'symbol': 'H'},
            'helium': {'number': 2, 'weight': 4.002602, 'symbol': 'He'}}

# todo: Add an 'is_noble_gas' entry to the hydrogen and helium dictionaries
# hint: helium is a noble gas, hydrogen isn't
elements['hydrogen']['is_noble_gas'] = False
elements['helium']['is_noble_gas'] = True





### Get key with maximum value of dictionary

Keymax = max(verse_dict, key=verse_dict.get) 

print(Keymax)


#### Make counters of dictionary

for word in book_title:
    if word not in word_counter:
        word_counter[word] = 1
    else:
        word_counter[word] += 1


for word in book_title:
    word_counter[word] = word_counter.get(word, 0) + 1


####iterate through dictionary

for key, value in cast.items():
    print("Actor: {}    Role: {}".format(key, value))
    
    
   
# Program to check prime number

check_prime = [26, 39, 51, 53, 57, 79, 85]

# iterate through the check_prime list
for num in check_prime:

# search for factors, iterating through numbers ranging from 2 to the number itself
    for i in range(2, num):

# number is not prime if modulo is 0
        if (num % i) == 0:
            print("{} is NOT a prime number, because {} is a factor of {}".format(num, i, num))
            break

# otherwise keep checking until we've searched all possible factors, and then declare it prime
        if i == num -1:    
            print("{} IS a prime number".format(num))
            
            
            
''' Zip 

#iterator
#Combines multiple iterators into one sequence of tuples. each tuple consists of elements in that position from all iterables.

'''

#####

x_coord = [23, 53, 2, -12, 95, 103, 14, -5]
y_coord = [677, 233, 405, 433, 905, 376, 432, 445]
z_coord = [4, 16, -6, -42, 3, -6, 23, -1]
labels = ["F", "J", "A", "Q", "Y", "B", "W", "X"]

points = []
for point in zip(labels, x_coord, y_coord, z_coord):
    points.append("{}: {}, {}, {}".format(*point))

for point in points:
    print(point)
    
    
#Some "ZIP" codes

x_coord = [23, 53, 2, -12, 95, 103, 14, -5]
y_coord = [677, 233, 405, 433, 905, 376, 432, 445]
z_coord = [4, 16, -6, -42, 3, -6, 23, -1]
labels = ["F", "J", "A", "Q", "Y", "B", "W", "X"]

points = []
for point in zip(labels, x_coord, y_coord, z_coord):
    points.append("{}: {}, {}, {}".format(*point))

for point in points:
    print(point)
    
    
 #Transpose with Zip 
 
 data = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))

data_transpose = tuple(zip(*data))
print(data_transpose)


#####


scores = {
             "Rick Sanchez": 70,
             "Morty Smith": 35,
             "Summer Smith": 82,
             "Jerry Smith": 23,
             "Beth Smith": 98
          }

passed = [name for name, score in scores.items() if score >= 65]
print(passed)
