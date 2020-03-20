# Create function that returns the average of an integer list
def average_numbers(num_list): 
avg = sum(num_list)/float(len(num_list)) # divide by length of list
return avg

# Take the average of a list: my_avg
my_avg = average_numbers([1, 2, 3, 4, 5, 6])

# Print out my_avg
print(my_avg)

-----------------------------------

# Import numpy as np
import numpy as np

# List input: my_matrix
my_matrix = [[1,2,3,4], [5,6,7,8]] 

# Function that converts lists to arrays: return_array
def return_array(matrix):
array = np.array(matrix, dtype = float)
return array

# Call return_array on my_matrix, and print the output
print(return_array(my_matrix))

------------
# Create a class: DataShell
class DataShell: 
pass
------------------------------


# Create empty class: DataShell
class DataShell:

# Pass statement
pass

# Instantiate DataShell: my_data_shell
my_data_shell = DataShell

# Print my_data_shell
print(my_data_shell)

---------------------------------

# Create class: DataShell
class DataShell:

# Initialize class with self argument
def __init__(self):

# Pass statement
pass

# Instantiate DataShell: my_data_shell
my_data_shell = DataShell()

# Print my_data_shell
print(my_data_shell)
-------------------------------------

# Create class: DataShell
class DataShell:

# Initialize class with self and integerInput arguments
def __init__(self, integerInput):

# Set data as instance variable, and assign the value of integerInput
self.data = integerInput

# Declare variable x with value of 10
x = 10 

# Instantiate DataShell passing x as argument: my_data_shell
my_data_shell = DataShell(x)

# Print my_data_shell
print(my_data_shell.data)

-----------------------------------------------------


# Create class: DataShell
class DataShell:

# Initialize class with self, identifier and data arguments
def __init__(self, identifier, data):

# Set identifier and data as instance variables, assigning value of input arguments
self.identifier = identifier
self.data = data

# Declare variable x with value of 100, and y with list of integers from 1 to 5
x = 100
y = [1, 2, 3, 4, 5]

# Instantiate DataShell passing x and y as arguments: my_data_shell
my_data_shell = DataShell(x, y)

# Print my_data_shell.identifier
print(my_data_shell.identifier)

# Print my_data_shell.data
print(my_data_shell.data)

---------------------------------------------------------------
# Create class: DataShell
class DataShell:

# Declare a class variable family, and assign value of "DataShell"
family = "DataShell"

# Initialize class with self, identifier arguments
def __init__(self, identifier):

# Set identifier as instance variable of input argument
self.identifier = identifier

# Declare variable x with value of 100
x = 100

# Instantiate DataShell passing x as argument: my_data_shell
my_data_shell = DataShell(x)

# Print my_data_shell class variable family

--------------------------------------------------------

print(my_data_shell.family)

-----------------------------

# Create class: DataShell
class DataShell:

# Declare a class variable family, and assign value of "DataShell"
family = "DataShell"

# Initialize class with self, identifier arguments
def __init__(self, identifier):

# Set identifier as instance variables, assigning value of input arguments
self.identifier = identifier

# Declare variable x with value of 100
x = 100

# Instantiate DataShell passing x as the argument: my_data_shell
my_data_shell = DataShell(x)

# Print my_data_shell class variable family
print(my_data_shell.family)

# Override the my_data_shell.family value with "NotDataShell"
my_data_shell.family = "NotDataShell"

# Print my_data_shell class variable family once again
print(my_data_shell.family)

--------------------------------------------------------------


# Create class: DataShell
class DataShell:

# Initialize class with self argument
def __init__(self):
pass

# Define class method which takes self argument: print_static
def print_static(self):
# Print string
print("You just executed a class method!")

# Instantiate DataShell taking no arguments: my_data_shell
my_data_shell = DataShell()

# Call the print_static method of your newly created object
my_data_shell.print_static()

-----------------------------------------

# Create class: DataShell
class DataShell:

# Initialize class with self and dataList as arguments
def __init__(self, dataList):
# Set data as instance variable, and assign it the value of dataList
self.data = dataList

# Define class method which takes self argument: show
def show(self):
# Print the instance variable data
print(self.data)

# Declare variable with list of integers from 1 to 10: integer_list 
integer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Instantiate DataShell taking integer_list as argument: my_data_shell
my_data_shell = DataShell(integer_list)

# Call the show method of your newly created object
my_data_shell.show()
--------------------------------------------------

# Create class: DataShell
class DataShell:

# Initialize class with self and dataList as arguments
def __init__(self, dataList):
# Set data as instance variable, and assign it the value of dataList
self.data = dataList

# Define method that prints data: show
def show(self):
print(self.data)

# Define method that prints average of data: avg 
def avg(self):
# Declare avg and assign it the average of data
avg = sum(self.data)/float(len(self.data))
# Print avg
print(avg)

# Instantiate DataShell taking integer_list as argument: my_data_shell
my_data_shell = DataShell(integer_list)

# Call the show and avg methods of your newly created object
my_data_shell.show()
my_data_shell.avg()
---------------------------------------------

# Create class: DataShell
class DataShell:

# Initialize class with self and dataList as arguments
def __init__(self, dataList):
# Set data as instance variable, and assign it the value of dataList
self.data = dataList

# Define method that returns data: show
def show(self):
return self.data

# Define method that prints average of data: avg 
def avg(self):
# Declare avg and assign it the average of data
avg = sum(self.data)/float(len(self.data))
# Return avg
return avg

# Instantiate DataShell taking integer_list as argument: my_data_shell
my_data_shell = DataShell(integer_list)

# Print output of your object's show method
print(my_data_shell.show())

# Print output of your object's avg method
print(my_data_shell.avg())

--------------------------------

# Load numpy as np and pandas as pd
import numpy as np
import pandas as pd

# Create class: DataShell
class DataShell:

# Initialize class with self and inputFile
def __init__(self, inputFile):
self.file = inputFile

# Define generate_csv method, with self argument
def generate_csv(self):
self.data_as_csv = pd.read_csv(self.file)
return self.data_as_csv

# Instantiate DataShell with us_life_expectancy as input argument
data_shell = DataShell(us_life_expectancy)

# Call data_shell's generate_csv method, assign it to df
df = data_shell.generate_csv()

# Print df
print(df)

-------------------------------------------

# Import numpy as np, pandas as pd
import numpy as np
import pandas as pd

# Create class: DataShell
class DataShell:

# Define initialization method
def __init__(self, filepath):
# Set filepath as instance variable 
self.filepath = filepath
# Set data_as_csv as instance variable
self.data_as_csv = pd.read_csv(filepath)

# Instantiate DataShell as us_data_shell
us_data_shell = DataShell(us_life_expectancy)

# Print your object's data_as_csv attribute
print(us_data_shell.data_as_csv)

--------------------------------

# Create class DataShell
class DataShell:

# Define initialization method
def __init__(self, filepath):
self.filepath = filepath
self.data_as_csv = pd.read_csv(filepath)

# Define method rename_column, with arguments self, column_name, and new_column_name
def rename_column(self, column_name, new_column_name):
self.data_as_csv.columns = self.data_as_csv.columns.str.replace(column_name, new_column_name)

# Instantiate DataShell as us_data_shell with argument us_life_expectancy
us_data_shell = DataShell(us_life_expectancy)

# Print the datatype of your object's data_as_csv attribute
print(us_data_shell.data_as_csv.dtypes)

# Rename your objects column 'code' to 'country_code'
us_data_shell.rename_column('code', 'country_code')

# Again, print the datatype of your object's data_as_csv attribute
print(us_data_shell.data_as_csv.dtypes)
------------------------------------------------------------

# Create class DataShell
class DataShell:

# Define initialization method
def __init__(self, filepath):
self.filepath = filepath
self.data_as_csv = pd.read_csv(filepath)

# Define method rename_column, with arguments self, column_name, and new_column_name
def rename_column(self, column_name, new_column_name):
self.data_as_csv.columns = self.data_as_csv.columns.str.replace(column_name, new_column_name)

# Define get_stats method, with argument self
def get_stats(self):
# Return a description data_as_csv
return self.data_as_csv.describe()

# Instantiate DataShell as us_data_shell
us_data_shell = DataShell(us_life_expectancy)

# Print the output of your objects get_stats method
print(us_data_shell.get_stats())
-----------------------------------------------

# Create class DataShell
class DataShell:

# Define initialization method
def __init__(self, filepath):
self.filepath = filepath
self.data_as_csv = pd.read_csv(filepath)

# Define method rename_column, with arguments self, column_name, and new_column_name
def rename_column(self, column_name, new_column_name):
self.data_as_csv.columns = self.data_as_csv.columns.str.replace(column_name, new_column_name)

# Define get_stats method, with argument self
def get_stats(self):
# Return a description data_as_csv
return self.data_as_csv.describe()

# Instantiate DataShell as us_data_shell
us_data_shell = DataShell(us_life_expectancy)

# Print the output of your objects get_stats method
print(us_data_shell.get_stats())

-------------------------------------------------------------------

# Create a class Animal
class Animal:
	def __init__(self, name):
		self.name = name

# Create a class Mammal, which inherits from Animal
class Mammal(Animal):
	def __init__(self, name, animal_type):
		self.animal_type = animal_type

# Create a class Reptile, which also inherits from Animal
class Reptile(Animal):
	def __init__(self, name, animal_type):
		self.animal_type = animal_type

# Instantiate a mammal with name 'Daisy' and animal_type 'dog': daisy
daisy = Mammal('Daisy', 'dog')

# Instantiate a reptile with name 'Stella' and animal_type 'alligator': stella
stella = Reptile('Stella', 'alligator')

# Print both objects
print(daisy)
print(stella)

------------------------------------


# Create a class Vertebrate
class Vertebrate:
    spinal_cord = True
    def __init__(self, name):
        self.name = name

# Create a class Mammal, which inherits from Vertebrate
class Mammal(Vertebrate):
    def __init__(self, name, animal_type):
        self.animal_type = animal_type
        self.temperature_regulation = True

# Create a class Reptile, which also inherits from Vertebrate
class Reptile(Vertebrate):
    def __init__(self, name, animal_type):
        self.animal_type = animal_type
        self.temperature_regulation = False

# Instantiate a mammal with name 'Daisy' and animal_type 'dog': daisy
daisy = Mammal('Daisy', 'dog')

# Instantiate a reptile with name 'Stella' and animal_type 'alligator': stella
stella = Reptile('Stella', 'alligator')

# Print stella's attributes spinal_cord and temperature_regulation
print("Stella Spinal cord: " + str(stella.spinal_cord))
print("Stella temperature regulation: " + str(stella.temperature_regulation))

# Print daisy's attributes spinal_cord and temperature_regulation
print("Daisy Spinal cord: " + str(daisy.spinal_cord))
print("Daisy temperature regulation: " + str(daisy.temperature_regulation))

------------------------------------

# Load numpy as np and pandas as pd
import numpy as np
import pandas as pd

# Create class: DataShell
class DataShell:
    def __init__(self, inputFile):
        self.file = inputFile

# Create class CsvDataShell, which inherits from DataShell
class CsvDataShell(DataShell):
    # Initialization method with arguments self, inputFile
    def __init__(self, inputFile):
        # Instance variable data
        self.data = pd.read_csv(inputFile)

# Instantiate CsvDataShell as us_data_shell, passing us_life_expectancy as argument
us_data_shell = CsvDataShell(us_life_expectancy)

# Print us_data_shell.data
print(us_data_shell.data)

------------------------------------
# Define abstract class DataShell
class DataShell:
    # Class variable family
    family = 'DataShell'
    # Initialization method with arguments, and instance variables
    def __init__(self, name, filepath): 
        self.name = name
        self.filepath = filepath

# Define class CsvDataShell      
class CsvDataShell(DataShell):
    # Initialization method with arguments self, name, filepath
    def __init__(self, name, filepath):
        # Instance variable data
        self.data = pd.read_csv(filepath)
        # Instance variable stats
        self.stats = self.data.describe()

# Instantiate CsvDataShell as us_data_shell
us_data_shell = CsvDataShell("US", us_life_expectancy)

# Print us_data_shell.stats
print(us_data_shell.stats)

------------------------------------

# Define abstract class DataShell
class DataShell:
    family = 'DataShell'
    def __init__(self, name, filepath): 
        self.name = name
        self.filepath = filepath

# Define class CsvDataShell
class CsvDataShell(DataShell):
    def __init__(self, name, filepath):
        self.data = pd.read_csv(filepath)
        self.stats = self.data.describe()

# Define class TsvDataShell
class TsvDataShell(DataShell):
    # Initialization method with arguments self, name, filepath
    def __init__(self, name, filepath):
        # Instance variable data
        self.data = pd.read_table(filepath)
        # Instance variable stats
        self.stats = self.data.describe()

# Instantiate CsvDataShell as us_data_shell, print us_data_shell.stats
us_data_shell = CsvDataShell("US", us_life_expectancy)
print(us_data_shell.stats)

# Instantiate TsvDataShell as france_data_shell, print france_data_shell.stats
france_data_shell = TsvDataShell("France", france_life_expectancy)
print(france_data_shell.stats)

-----------------------------------------
