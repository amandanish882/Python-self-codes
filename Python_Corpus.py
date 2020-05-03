# Example, do not modify!
print(5 / 8)

# Print the sum of 7 and 10
print(7+10)

-------------------------------------

# Division
print(5 / 8)

#Addition
print(7 + 10)

--------------------------------------
# Addition, subtraction
print(5 + 5)
print(5 - 5)

# Multiplication, division, modulo, and exponentiation
print(3 * 5)
print(10 / 2)
print(18 % 7)
print(4 ** 2)

# How much is your $100 worth after 7 years?
print(100*1.1**7)

----------------------------------------

# Create a variable savings
savings = 100

# Print out savings
print (savings)
-----------------------------------------

# Create a variable savings
savings = 100

# Create a variable growth_multiplier
growth_multiplier = 1.1

# Calculate result
result = 100 * 1.1 ** 7

# Print out result
print(result)
------------------------------------------

# Create a variable desc
desc = "compound interest"

# Create a variable profitable
profitable = True

-------------------------------------------

savings = 100
growth_multiplier = 1.1
desc = "compound interest"

# Assign product of growth_multiplier and savings to year1
year1 = savings * growth_multiplier

# Print the type of year1
print(type(year1))

# Assign sum of desc and desc to doubledesc
doubledesc = desc + desc

# Print out doubledesc
print (doubledesc)

-------------------------------------------------

# Definition of savings and result
savings = 100
result = 100 * 1.10 ** 7

# Fix the printout
print("I started with $" + str(savings) + " and now have $" + str(result) + ". Awesome!")

# Definition of pi_string
pi_string = "3.1415926"

# Convert pi_string into float: pi_float
pi_float = float(pi_string)
---------------------------------------------------

# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Create list areas
areas = [hall,kit,liv,bed,bath]

# Print areas
print(areas)


---------------------------------------------------------

# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Adapt list areas
areas = ["hallway",hall,"kitchen", kit, "living room", liv, "bedroom" ,bed, "bathroom", bath]

# Print areas
print(areas)
--------------------------------------------------------------

# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# house information as list of lists
house = [["hallway", hall],
         ["kitchen", kit],
         ["living room", liv],
         ["bedroom", bed],
         ["bathroom",bath]]

# Print out house
print(house)

# Print out the type of house
print(type(house))
------------------------------------------------------------------

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Print out second element from areas
print(areas[1])

# Print out last element from areas
print(areas[-1])

# Print out the area of the living room
print(areas[:])

----------------------------------------------------------------------

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Sum of kitchen and bedroom area: eat_sleep_area
eat_sleep_area = (areas[3] + areas[7])

# Print the variable eat_sleep_area
print(eat_sleep_area)

---------------------------------------------------------------------

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Use slicing to create downstairs
downstairs = areas[0:6]

# Use slicing to create upstairs
upstairs = areas[6:10]

# Print out downstairs and upstairs
print(upstairs)
print(downstairs)


-------------------------------------------------------------------------

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Alternative slicing to create downstairs
downstairs = areas[:6]

# Alternative slicing to create upstairs
upstairs = areas[6:]

--------------------------------------------------------------------------

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Correct the bathroom area
areas[-1] = 10.5

# Change "living room" to "chill zone"
areas[4] = "chill zone"

-----------------------------------------------------------------------------

# Create the areas list and make some changes
areas = ["hallway", 11.25, "kitchen", 18.0, "chill zone", 20.0,
         "bedroom", 10.75,"bathroom", 10.50]

# Add poolhouse data to areas, new list is areas_1
areas_1 = areas + ["poolhouse",24.5]

# Add garage data to areas_1, new list is areas_2
areas_2 = areas_1 +["garage",15.45]

------------------------------------------------------------------------------

# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Create areas_copy
areas_copy = list(areas)

# Change areas_copy
areas_copy[0] = 5.0

# Print areas
print(areas)

---------------------------------------------------------------------------------

# Create variables var1 and var2
var1 = [1, 2, 3, 4]
var2 = True

# Print out type of var1
print(type(var1))

# Print out length of var1
print(len(var1))

# Convert var2 to an integer: out2
out2 = int(var2)

------------------------------------------------------------------------------------

# Create lists first and second
first = [11.25, 18.0, 20.0]
second = [10.75, 9.50]

# Paste together first and second: full
full = first + second

# Sort full in descending order: full_sorted
full_sorted = sorted(full, reverse = True)

# Print out full_sorted
print(full_sorted)

---------------------------------------------------------------------------------------

# string to experiment with: place
place = "poolhouse"

# Use upper() on place: place_up
place_up = place.upper()

# Print out place and place_up
print(place);print(place_up)

# Print out the number of o's in place
print(place.count("o"))

--------------------------------------------------------------------------------------

# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Print out the index of the element 20.0
print(areas.index(20.0))

# Print out how often 9.50 appears in areas
print(areas.count(9.50))


----------------------------------------------------------------------------------


# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Use append twice to add poolhouse and garage size
areas.append(24.5)
areas.append(15.45)

# Print out areas
print(areas)

# Reverse the orders of the elements in areas
areas.reverse()

# Print out areas
print(areas)

---------------------------------------------------------------------------------


# Definition of radius
r = 0.43

# Import the math package
import math

# Calculate C
C = 2*math.pi*r

# Calculate A
A = math.pi*r**2

# Build printout
print("Circumference: " + str(C))
print("Area: " + str(A))


------------------------------------------------------------------------------------

# Definition of radius
r = 192500

# Import radians function of math package
from math import radians

# Travel distance of Moon over 12 degrees. Store in dist.
phi = radians(12)
dist = r * phi

# Print out dist
print(dist)

---------------------------------------------------------------------------------

'''Numpy '''

# Create list baseball
baseball = [180, 215, 210, 210, 188, 176, 209, 200]

# Import the numpy package as np
import numpy as np

# Create a numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out type of np_baseball
print(type(np_baseball))


--------------------------------------------------------------------------------

# height is available as a regular list

# Import numpy
import numpy as np

# Create a numpy array from height_in: np_height_in
np_height_in = np.array(height_in)

# Print out np_height_in
print(np_height_in)

# Convert np_height_in to m: np_height_m
np_height_m = np_height_in *0.0254

# Print np_height_m
print(np_height_m)

-----------------------------------------------------------------------------------


# height and weight are available as regular lists

# Import numpy
import numpy as np

# Create array from height_in with metric units: np_height_m
np_height_m = np.array(height_in) * 0.0254

# Create array from weight_lb with metric units: np_weight_kg
np_weight_kg = np.array(weight_lb) * 0.453592

# Calculate the BMI: bmi
bmi = np_weight_kg/np_height_m**2

# Print out bmi
print(bmi)

--------------------------------------------------------------------------------------

# height and weight are available as a regular lists

# Import numpy
import numpy as np

# Calculate the BMI: bmi
np_height_m = np.array(height_in) * 0.0254
np_weight_kg = np.array(weight_lb) * 0.453592
bmi = np_weight_kg / np_height_m ** 2

# Create the light array
light = bmi<21

# Print out light
print(light)

# Print out BMIs of all baseball players whose BMI is below 21
print(bmi[light])

-------------------------------------------------------------------------------------------

# height and weight are available as a regular lists

# Import numpy
import numpy as np

# Store weight and height lists as numpy arrays
np_weight_lb = np.array(weight_lb)
np_height_in = np.array(height_in)

# Print out the weight at index 50
print(np_weight_lb[50])

# Print out sub-array of np_height_in: index 100 up to and including index 110
print(np_height_in[100:111])

---------------------------------------------------------------------------------------------


# Create baseball, a list of lists
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]

# Import numpy
import numpy as np

# Create a 2D numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out the type of np_baseball
print(type(np_baseball))

# Print out the shape of np_baseball
print(np_baseball.shape)

---------------------------------------------------------------------------------------------------

# baseball is available as a regular list of lists

# Import numpy package
import numpy as np

# Create a 2D numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out the shape of np_baseball
print(np_baseball.shape)

---------------------------------------------------------------------------------------------------

# baseball is available as a regular list of lists

# Import numpy package
import numpy as np

# Create np_baseball (2 cols)
np_baseball = np.array(baseball)

# Print out the 50th row of np_baseball
print(np_baseball[49,:])

# Select the entire second column of np_baseball: np_weight_lb
np_weight_lb = np_baseball[:,1]

# Print out height of 124th player
print(np_baseball[123,0])

-----------------------------------------------------------------------------------------------------

# baseball is available as a regular list of lists
# updated is available as 2D numpy array

# Import numpy package
import numpy as np

# Create np_baseball (3 cols)
np_baseball = np.array(baseball)

# Print out addition of np_baseball and updated
print(np_baseball+updated)

# Create numpy array: conversion
conversion = np.array([0.0254,0.453592,1])

# Print out product of np_baseball and conversion
print(np_baseball * conversion)

------------------------------------------------------------------------------------------------------


# np_baseball is available

# Import numpy
import numpy as np

# Create np_height_in from np_baseball
np_height_in = np_baseball[:,0]

# Print out the mean of np_height_in
print(np.mean(np_height_in))

# Print out the median of np_height_in
print(np.median(np_height_in))

-----------------------------------------------------------------------------------------------------

# np_baseball is available

# Import numpy
import numpy as np

# Print mean height (first column)
avg = np.mean(np_baseball[:,0])
print("Average: " + str(avg))

# Print median height. Replace 'None'
med = np.median(np_baseball[:,0])
print("Median: " + str(med))

# Print out the standard deviation on height. Replace 'None'
stddev = np.std(np_baseball[:,0])
print("Standard Deviation: " + str(stddev))

# Print out correlation between first and second column. Replace 'None'
corr = np.corrcoef(np_baseball[:,0],np_baseball[:,1])
print("Correlation: " + str(corr))

------------------------------------------------------------------------------------------------------

# heights and positions are available as lists

# Import numpy
import numpy as np

# Convert positions and heights to numpy arrays: np_positions, np_heights
np_positions = np.array(positions)
np_heights = np.array(heights)

# Heights of the goalkeepers: gk_heights
gk_heights = np_heights[np_positions == 'GK']

# Heights of the other players: other_heights
other_heights = np_heights[np_positions != 'GK']

# Print out the median height of goalkeepers. Replace 'None'
print("Median height of goalkeepers: " + str(np.median(gk_heights)))

# Print out the median height of other players. Replace 'None'
print("Median height of other players: " + str(np.median(other_heights)))

--------------------------------------------------------------------------------------------------------

"""MATplotlib"""

# Print the last item from year and pop
print(year[-1])
print(pop[-1])

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(year,pop)


# Display the plot with plt.show()
plt.show()

-------------------------------------------

# Change the line plot below to a scatter plot
plt.scatter(gdp_cap, life_exp)

# Put the x-axis on a logarithmic scale
plt.xscale('log')

# Show plot
plt.show()

--------------------------------------------------

# Import package
import matplotlib.pyplot as plt

# Build Scatter plot
plt.scatter(pop,life_exp)

# Show plot
plt.show()

--------------------------------------------------

# Create histogram of life_exp data
plt.hist(life_exp)

# Display histogram
plt.show()

-------------------------------------------------


# Build histogram with 5 bins
plt.hist(life_exp,bins = 5)

# Show and clean up plot
plt.show()
plt.clf()

# Build histogram with 20 bins
plt.hist(life_exp,bins = 20)

# Show and clean up again
plt.show()
plt.clf()

--------------------------------------------

# Basic scatter plot, log scale
plt.scatter(gdp_cap, life_exp)
plt.xscale('log') 

# Strings
xlab = 'GDP per Capita [in USD]'
ylab = 'Life Expectancy [in years]'
title = 'World Development in 2007'

# Add axis labels
plt.xlabel(xlab)
plt.ylabel(ylab)



# Add title
plt.title(title)

# After customizing, display the plot
plt.show()

----------------------------------------------

# Scatter plot
plt.scatter(gdp_cap, life_exp)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')

# Definition of tick_val and tick_lab
tick_val = [1000, 10000, 100000]
tick_lab = ['1k', '10k', '100k']

# Adapt the ticks on the x-axis
plt.xticks(tick_val,tick_lab)

# After customizing, display the plot
plt.show()

------------------------------------------------------

# Import numpy as np
import numpy as np

# Store pop as a numpy array: np_pop
np_pop = np.array(pop)

# Double np_pop
np_pop = np_pop*2

# Update: set s argument to np_pop
plt.scatter(gdp_cap, life_exp, s = np_pop)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])

# Display the plot
plt.show()

---------------------------------------------------------

dict = {
    'Asia':'red',
    'Europe':'green',
    'Africa':'blue',
    'Americas':'yellow',
    'Oceania':'black'
}


# Specify c and alpha inside plt.scatter()
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c = col, alpha = 0.8)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])

# Show the plot
plt.show()


-----------------------------------------------------



# Scatter plot
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c = col, alpha = 0.8)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])

# Additional customizations
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')

# Add grid() call
plt.grid(True)

# Show the plot
plt.show()


-------------------------------------------------------------
"""dictionary"""

# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# Get index of 'germany': ind_ger
ind_ger = countries.index('germany')

# Use ind_ger to print out capital of Germany
print(capitals[ind_ger])


----------------------------------------------------------------
# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# From string in countries and capitals, create dictionary europe
europe = { 'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Print europe
print(europe)

------------------------------------------------------------------


# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Add italy to europe
europe['italy'] = 'rome'

# Print out italy in europe
print('italy' in europe)

# Add poland to europe
europe['poland'] = 'warsaw'

# Print europe
print(europe)


------------------------------------------------------------------

# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw',
          'australia':'vienna' }

# Update capital of germany
europe['germany'] = 'berlin'

# Remove australia
del europe['australia']

# Print europe
print(europe)

----------------------------------------------------------------


# Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }


# Print out the capital of France
print(europe['france']['capital'])

# Create sub-dictionary data
data = { 'capital': 'rome', 'population' : 59.83}

# Add data to europe under key 'italy'
europe['italy'] = data

# Print europe
print(europe)


------------------------------------------------------------------

"""PANDAS"""

# Pre-defined lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

# Import pandas as pd
import pandas as pd

# Create dictionary my_dict with three key:value pairs: my_dict
my_dict = {'country' : names, 'drives_right' : dr, 'cars_per_cap' : cpc}

# Build a DataFrame cars from my_dict: cars
cars = pd.DataFrame(my_dict)

# Print cars
print(cars)


--------------------------------------------------------------------

import pandas as pd

# Build cars DataFrame
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
cars_dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
cars = pd.DataFrame(cars_dict)
print(cars)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index = row_labels

# Print cars again
print(cars)


--------------------------------------------------------------------

# Import pandas as pd
import pandas as pd

# Fix import by including index_col
cars = pd.read_csv('cars.csv',index_col= 0)

# Print out cars
print(cars)

--------------------------------------------------------------------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out country column as Pandas Series
print(cars['country'])

# Print out country column as Pandas DataFrame
print(cars[['country']])

# Print out DataFrame with country and drives_right columns
print(cars[['country','drives_right']])

--------------------------------------------------------------------


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out first 3 observations
print(cars[0:4])

# Print out fourth, fifth and sixth observation
print(cars[3:6])

-------------------------------------------------------------------

"""SERIES VS DATAFRAME"""

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out observation for Japan
print(cars.loc['JPN'])
print(cars.iloc[2])
# Print out observations for Australia and Egypt
print(cars.loc[['AUS','EG']])
print(cars.iloc[[1,6]])

--------------------------------------------------------------------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out drives_right value of Morocco
print(cars.loc[['MOR'],['drives_right']])

# Print sub-DataFrame
print(cars.loc[['RU','MOR'],['country','drives_right']])


-----------------------------------------------------------------------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out drives_right column as Series
print(cars['drives_right'])

# Print out drives_right column as DataFrame
print(cars[['drives_right']])

# Print out cars_per_cap and drives_right as DataFrame
print(cars.loc[:,['cars_per_cap','drives_right']])



------------------------------------------------------------------------

#################### BOOOLEANS

# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than or equal to 18
print(my_house >= 18)

# my_house less than your_house
print(my_house<your_house)


-----------------------------------------------------------------

# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than 18.5 or smaller than 10
print(np.logical_or(my_house>18.5,my_house<10))

# Both my_house and your_house smaller than 11
print(np.logical_and(my_house<11,your_house<11))

-------------------------------------------------------------------

## IFELSE

# Define variables
room = "kit"
area = 14.0

# if statement for room
if room == "kit" :
    print("looking around in the kitchen.")

# if statement for area
if area > 15:
    print("big place!")


-------------------------------------------------------------------

# Define variables
room = "kit"
area = 14.0

# if-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
else :
    print("looking around elsewhere.")

# if-else construct for area
if area > 15 :
    print("big place!")
else : 
    print("pretty small.")
    
-------------------------------------------------------------------

# Define variables
room = "bed"
area = 14.0

# if-elif-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
elif room == "bed":
    print("looking around in the bedroom.")
else :
    print("looking around elsewhere.")

# if-elif-else construct for area
if area > 15 :
    print("big place!")
elif area >10 :
    print("medium size, nice!")
else :
    print("pretty small.")

--------------------------------------------------------------------
 '''PANDAS DATAFRAME FILTERING'''
 
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Convert code to a one-liner
sel = cars[cars['drives_right']]

# Print sel
print(sel)

---------------------------------------------------------------------


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Create car_maniac: observations that have a cars_per_cap over 500
cpc = cars['cars_per_cap']

many_cars = cpc > 500

car_maniac = cars[many_cars]
# Print car_maniac
print(car_maniac)

----------------------------------------------------------------------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Import numpy, you'll need this
import numpy as np

# Create medium: observations with cars_per_cap between 100 and 500
cpc = cars['cars_per_cap']
medium = cars[np.logical_and(cpc > 100,cpc < 500)]

print(medium)


---------------------------------------------------------------------

"""WHILE LOOPS """

# Initialize offset
offset = 8

# Code the while loop

while offset !=0 : 
    print("correcting...")
    offset = offset - 1
    print(offset)

------------------------------------------------------------------------------


# Initialize offset
offset = -6

# Code the while loop
while offset != 0 :
    print("correcting...")
    if offset > 0 :
      offset = offset - 1
    else : 
      offset = offset + 1    
    print(offset)


---------------------------------------------------------------------------------


# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop

for ctr in areas:
    print(ctr)
    
-------------------------------------------------------------------------------

#### Pritnting using formats


print("I love {} for \"{}!\"".format('Geeks', 'Geeks')) 
  
# using format() method and refering  
# a position of the object 
print('{0} and {1}'.format('Geeks', 'Portal')) 
  
print('{1} and {0}'.format('Geeks', 'Portal')) 

-----------------------------------------------------------------------------

# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Change for loop to use enumerate() and update print()
for idx,a in enumerate(areas) :
    print("room {}: {}".format(idx,a)) 


or
print("room " + str(index) + ": " + str(area))
-----------------------------------------------------------------------

###Looping in lists
# house list of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
         
# Build a for loop from scratch

for sublist in house:
    print('the {} is {} sqm'.format(sublist[0],sublist[1]))

-----------------------------------------------------------------------

#### LOOPING over dictionary

# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
          
# Iterate over europe

for key,value in europe.items():
    print('the capital of {} is {}'.format(key,value))

------------------------------------------------------------------------

####looping over individual elements of numpy  2d array

# Import numpy as np
import numpy as np

# For loop over np_height
for el in np_height:
    print('{} inches'.format(el))

# For loop over np_baseball
for el in np.nditer(np_baseball):
    print(el)
    
    
for el in np_baseball:
    print(el)

------------------------------------------------------------------------------

#### Iteration over Pandas

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars
for rlab,row in cars.iterrows():
    print(rlab)
    print(row)

--------------------------------------------------------------------------------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Adapt for loop
for lab, row in cars.iterrows() :
    print('{}: {}'.format(lab,row['cars_per_cap']))

-------------------------------------------------------------------------------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Code for loop that adds COUNTRY column
for lab,row in cars.iterrows() : 
    cars.loc[lab,"COUNTRY"] = row["country"].upper()


# Print cars
print(cars)

--------------------------------------------------------------------------------


# Import cars data, apply
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Use .apply(str.upper)
#for lab, row in cars.iterrows() :
#    cars.loc[lab, "COUNTRY"] = row["country"].upper()

cars["COUNTRY"] = cars["country"].apply(str.upper)

----------------------------------------------------------------------------------


#### Mini Random walk and statistics


# Import numpy as np
import numpy as np

# Set the seed
np.random.seed(123)

# Generate and print random float
print(np.random.rand())



------------------------------------------------------------------------------
'''note - second argument to randint is not printed'''

# Import numpy and set seed
import numpy as np
np.random.seed(123)

# Use randint() to simulate a dice
print(np.random.randint(1,7))

# Use randint() again
print(np.random.randint(1,7))

----------------------------------------------------------------------------


# Numpy is imported, seed is set

# Starting step
step = 50

# Roll the dice
dice = np.random.randint(1,7)

# Finish the control construct
if dice <= 2 :
    step = step - 1
elif dice>2 and dice<6 :
    step = step + 1
else :
    step = step + np.random.randint(1,7)

# Print out dice and step
print(dice);print(step)

-----------------------------------------------------------------------------

# Numpy is imported, seed is set


# Initialize random_walk
random_walk = [0]

# Complete the random_walk
for x in range(100) :
    # Set step: last element in random_walk
    step = random_walk[-1]

    # Roll the dice
    dice = np.random.randint(1,7)

    # Determine next step
    if dice <= 2:
        step = step - 1
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    # append next_step to random_walk
    random_walk.append(step)

# Print random_walk
print(random_walk)


--------------------------------------------------------------------------------

# Numpy is imported, seed is set


# Initialize random_walk
random_walk = [0]

# Complete the random_walk
for x in range(100) :
    # Set step: last element in random_walk
    step = random_walk[-1]

    # Roll the dice
    dice = np.random.randint(1,7)

    # Determine next step
    if dice <= 2:
        step = step = max(0,step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    # append next_step to random_walk
    random_walk.append(step)

# Print random_walk
print(random_walk)


-----------------------------------------------------------------------------------
#Plotting the random walk

# Numpy is imported, seed is set

# Initialization
random_walk = [0]

for x in range(100) :
    step = random_walk[-1]
    dice = np.random.randint(1,7)

    if dice <= 2:
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    random_walk.append(step)

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


# Plot random_walk
plt.plot(random_walk)

# Show the plot
plt.show()

----------------------------------------------------------------------------
# numpy and matplotlib imported, seed set.

# initialize and populate all_walks
all_walks = []
for i in range(10) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)
    all_walks.append(random_walk)

# Convert all_walks to Numpy array: np_aw
np_aw = np.array(all_walks)

# Plot np_aw and show
plt.plot(np_aw)
plt.show()

# Clear the figure
plt.clf()

# Transpose np_aw: np_aw_t
np_aw_t = np.transpose(np_aw)

# Plot np_aw_t and show
plt.plot(np_aw_t)
plt.show()


-------------------------------------------------------------

#implementing the probability that a run will be <0.1%


# numpy and matplotlib imported, seed set

# Simulate random walk 250 times
all_walks = []
for i in range(250) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)

        # Implement clumsiness
        if np.random.rand()< 0.001 :
            step = 0

        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))
plt.plot(np_aw_t)
plt.show()

-------------------------------------------------------------


#Writing your own functions

# Define the function shout
def shout():
"""Print a string with three exclamation marks"""
# Concatenate the strings: shout_word
shout_word = 'congratulations' +'!!!'

# Print shout_word
print(shout_word)

# Call shout
shout()

------------

# Define shout with the parameter, word
def shout(word):
"""Print a string with three exclamation marks"""
# Concatenate the strings: shout_word
shout_word = word + '!!!'

# Print shout_word
print(shout_word)

# Call shout with the string 'congratulations'
shout('congratulations')

------------------------

# Define shout with the parameter, word
def shout(word):
"""Return a string with three exclamation marks"""
# Concatenate the strings: shout_word
shout_word = word + '!!!'

# Replace print with return
return(shout_word)

# Pass 'congratulations' to shout: yell
yell = shout('congratulations')

# Print yell
print(yell)


# Define shout with parameters word1 and word2
def shout(word1, word2):
    """Concatenate strings with three exclamation marks"""
    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + '!!!'
   
    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + '!!!'
   
    # Concatenate shout1 with shout2: new_shout
    new_shout = shout1+shout2

    # Return new_shout
    return new_shout

# Pass 'congratulations' and 'you' to shout(): yell
yell = shout('congratulations','you')

# Print yell
print(yell)


# Unpack nums into num1, num2, and num3
num1, num2, num3 = nums

# Construct even_nums
even_nums = (2, num2, num3)


# Define shout_all with parameters word1 and word2
def shout_all(word1, word2):
   
    # Concatenate word1 with '!!!':
   
    shout1 = word1 + '!!!'
    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + '!!!'
   
    # Construct a tuple with shout1 and shout2: shout_words
    shout_words = (shout1,shout2)

    # Return shout_words
    return Shout_words

# Pass 'congratulations' and 'you' to shout_all(): yell1, yell2
yell1, yell2 = shout_all('congratulations','you')

# Print yell1 and yell2
print(yell1)
print(yell2)

--------------------------------------------------------------

# Import pandas
import pandas as pd

# Import Twitter data as DataFrame: df
df = pd.read_csv('tweets.csv')

# Initialize an empty dictionary: langs_count
langs_count = {}

# Extract column from DataFrame: col
col = df['lang']

# Iterate over lang column in DataFrame
for entry in col:

# If the language is in langs_count, add 1
if entry in langs_count.keys():
langs_count[entry]+=1
# Else add the language to langs_count, set the value to 1
else:
langs_count[entry] = 1

# Print the populated dictionary
print(langs_count)

----------------------------------------------------

# Define count_entries()
def count_entries(df, col_name):
"""Return a dictionary with counts of
occurrences as value for each key."""

# Initialize an empty dictionary: langs_count
langs_count = {}

# Extract column from DataFrame: col
col = df[col_name]

# Iterate over lang column in DataFrame
for entry in col:

# If the language is in langs_count, add 1
if entry in langs_count.keys():
langs_count[entry]+=1
# Else add the language to langs_count, set the value to 1
else:
langs_count[entry] = 1

# Return the langs_count dictionary
return langs_count

# Call count_entries(): result
result = count_entries(tweets_df,'lang')

# Print the result
print(result)

--------------------------------------------------------------------------------

# Create a string: team
team = "teen titans"

# Define change_team()
def change_team():
"""Change the value of the global variable team."""

# Use team in global scope
global team

# Change the value of team in global: team
team = "justice league"
# Print team
print(team)

# Call change_team()
change_team()

# Print team
print(team)

-----------------------------------------------------------------------------

'sum' in dir(builtins)
--------------------------------------------------------------------

# Define three_shouts
def three_shouts(word1, word2, word3):
"""Returns a tuple of strings
concatenated with '!!!'."""

# Define inner
def inner(word):
"""Returns a string concatenated with '!!!'."""
return word + '!!!'

# Return a tuple of strings
return (inner(word1), inner(word2), inner(word3))

# Call three_shouts() and print
print(three_shouts('a', 'b', 'c'))

--------------------------------------------------------------------


# Define echo
def echo(n):
"""Return the inner_echo function."""

# Define inner_echo
def inner_echo(word1):
"""Concatenate n copies of word1."""
echo_word = word1 * n
return echo_word

# Return inner_echo
return inner_echo

# Call echo: twice
twice = echo(2)

# Call echo: thrice
thrice = echo(3)

# Call twice() and thrice() then print
print(twice('hello'), thrice('hello'))


---------------------------------------------------------------------------
# Define echo_shout()
def echo_shout(word):
"""Change the value of a nonlocal variable"""

# Concatenate word with itself: echo_word
echo_word = word + word

# Print echo_word
print(echo_word)

# Define inner function shout()
def shout():
"""Alter a variable in the enclosing scope"""
# Use echo_word in nonlocal scope
nonlocal echo_word

# Change echo_word to echo_word concatenated with '!!!'
echo_word = echo_word + '!!!'

# Call function shout()
shout()

# Print echo_word
print(echo_word)

# Call function echo_shout() with argument 'hello'
echo_shout('hello')

-------------------------------------------------------------------------

# Define shout_echo
def shout_echo(word1, echo = 1):
"""Concatenate echo copies of word1 and three
exclamation marks at the end of the string."""

# Concatenate echo copies of word1 using *: echo_word
echo_word = word1*echo

# Concatenate '!!!' to echo_word: shout_word
shout_word = echo_word + '!!!'

# Return shout_word
return shout_word

# Call shout_echo() with "Hey": no_echo
no_echo = shout_echo("Hey")

# Call shout_echo() with "Hey" and echo=5: with_echo
with_echo = shout_echo("Hey",5)

# Print no_echo and with_echo
print(no_echo)
print(with_echo)

------------------------------------------------------------------------------------------------------------------------------------
# Define shout_echo
def shout_echo(word1, echo = 1, intense = False):
"""Concatenate echo copies of word1 and three
exclamation marks at the end of the string."""

# Concatenate echo copies of word1 using *: echo_word
echo_word = word1 * echo

# Make echo_word uppercase if intense is True
if intense is True:
# Make uppercase and concatenate '!!!': echo_word_new
echo_word_new = echo_word.upper() + '!!!'
else:
# Concatenate '!!!' to echo_word: echo_word_new
echo_word_new = echo_word + '!!!'

# Return echo_word_new
return echo_word_new

# Call shout_echo() with "Hey", echo=5 and intense=True: with_big_echo
with_big_echo = shout_echo("Hey",echo = 5 , intense = True)

# Call shout_echo() with "Hey" and intense=True: big_no_echo
big_no_echo = shout_echo("Hey", intense = True)

# Print values
print(with_big_echo)
print(big_no_echo)

---------------------------------------------------------------------------------------

# Define gibberish
def gibberish(*args):
"""Concatenate strings in *args together."""

# Initialize an empty string: hodgepodge
hodgepodge = ""

# Concatenate the strings in args
for word in args:
hodgepodge += word

# Return hodgepodge
return hodgepodge

# Call gibberish() with one string: one_word
one_word = gibberish("luke")

# Call gibberish() with five strings: many_words
many_words = gibberish("luke", "leia", "han", "obi", "darth")

# Print one_word and many_words
print(one_word)
print(many_words)

--------------------------------------------------------------------------------------------

# Define report_status
def report_status(**kwargs):
"""Print out the status of a movie character."""

print("\nBEGIN: REPORT\n")

# Iterate over the key-value pairs of kwargs
for keys, values in kwargs.items():
# Print out the keys and values, separated by a colon ':'
print(keys + ": " + values)

print("\nEND REPORT")

# First call to report_status()
report_status(name = "luke", affiliation = "jedi", status = "missing")

# Second call to report_status()
report_status(name="anakin", affiliation="sith lord", status="deceased")

---------------------------------------------------------------------------------------------

# Define count_entries()
def count_entries(df, col_name = 'lang'):
"""Return a dictionary with counts of
occurrences as value for each key."""

# Initialize an empty dictionary: cols_count
cols_count = {}

# Extract column from DataFrame: col
col = df[col_name]

# Iterate over the column in DataFrame
for entry in col:

# If entry is in cols_count, add 1
if entry in cols_count.keys():
cols_count[entry] += 1

# Else add the entry to cols_count, set the value to 1
else:
cols_count[entry] = 1

# Return the cols_count dictionary
return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df,'lang')

# Call count_entries(): result2
result2 = count_entries(tweets_df,'source')

# Print result1 and result2
print(result1)
print(result2)

----------------------------------------------------------------------------------------------------------------------------

# Define count_entries()
def count_entries(df, *args):
"""Return a dictionary with counts of
occurrences as value for each key."""

#Initialize an empty dictionary: cols_count
cols_count = {}

# Iterate over column names in args
for col_name in args:

# Extract column from DataFrame: col
col = df[col_name]

# Iterate over the column in DataFrame
for entry in col:

# If entry is in cols_count, add 1
if entry in cols_count.keys():
cols_count[entry] += 1

# Else add the entry to cols_count, set the value to 1
else:
cols_count[entry] = 1

# Return the cols_count dictionary
return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'lang', 'source')

# Print result1 and result2
print(result1)
print(result2)


--------------------------------------------------------------\

Lambda Function

# Define echo_word as a lambda function: echo_word
echo_word = (lambda word1,echo: word1 * echo)

# Call echo_word: result
result = echo_word('hey',5)

# Print result
print(result)

-----------------------

# Create a list of strings: spells
spells = ["protego", "accio", "expecto patronum", "legilimens"]

# Use map() to apply a lambda function over spells: shout_spells
shout_spells = map(lambda item : item + '!!!', spells)

# Convert shout_spells to a list: shout_spells_list
shout_spells_list = list(shout_spells)

# Print the result
print(shout_spells_list)

--------------------------------------------------------------------

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'pippin', 'aragorn', 'boromir', 'legolas', 'gimli', 'gandalf']

# Use filter() to apply a lambda function over fellowship: result
result = filter(lambda member: len(member)>6, fellowship)

# Convert result to a list: result_list
result_list = list(result)

# Print result_list
print(result_list)

# Import reduce from functools
from functools import reduce

# Create a list of strings: stark
stark = ['robb', 'sansa', 'arya', 'brandon', 'rickon']

# Use reduce() to apply a lambda function over stark: result
result = reduce(lambda item1,item2: item1+item2, stark)

# Print the result
print(result)


---------------------------------

# Define shout_echo
def shout_echo(word1, echo=1):
"""Concatenate echo copies of word1 and three
exclamation marks at the end of the string."""

# Initialize empty strings: echo_word, shout_words
echo_word="";shout_words = ""


# Add exception handling with try-except
try:
# Concatenate echo copies of word1 using *: echo_word
echo_word = word1*echo

# Concatenate '!!!' to echo_word: shout_words
shout_words = echo_word+"!!!" 
except:
# Print error message
print("word1 must be a string and echo must be an integer.")

# Return shout_words
return shout_words

# Call shout_echo
shout_echo("particle", echo="accelerator")
-------------------------------
# Define shout_echo
def shout_echo(word1, echo=1):
"""Concatenate echo copies of word1 and three
exclamation marks at the end of the string."""

# Raise an error with raise
if echo<0:
raise ValueError('echo must be greater than or equal to 0')

# Concatenate echo copies of word1 using *: echo_word
echo_word = word1 * echo

# Concatenate '!!!' to echo_word: shout_word
shout_word = echo_word + '!!!'

# Return shout_word
return shout_word

# Call shout_echo
shout_echo("particle", echo=5)

----------------------------------------------------

# Select retweets from the Twitter DataFrame: result
result = filter(lambda x:x[0:2] == 'RT', tweets_df['text'])

# Create list from filter object result: res_list
res_list = list(result)

# Print all retweets in res_list
for tweet in res_list:
print(tweet)
-------------------------------------
# Define count_entries()
def count_entries(df, col_name='lang'):
"""Return a dictionary with counts of
occurrences as value for each key."""

# Initialize an empty dictionary: cols_count
cols_count = {}

# Add try block
try:
# Extract column from DataFrame: col
col = df[col_name]

# Iterate over the column in dataframe
for entry in col:

# If entry is in cols_count, add 1
if entry in cols_count.keys():
cols_count[entry] += 1
# Else add the entry to cols_count, set the value to 1
else:
cols_count[entry] = 1

# Return the cols_count dictionary
return cols_count

# Add except block
except:
print('The DataFrame does not have a ' + col_name + ' column.')

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Print result1
print(result1)

----------------------------------------------
# Define count_entries()
def count_entries(df, col_name='lang'):
"""Return a dictionary with counts of
occurrences as value for each key."""

# Raise a ValueError if col_name is NOT in DataFrame
if col_name not in df.columns:
raise ValueError('The DataFrame does not have a ' + col_name + ' column.')

# Initialize an empty dictionary: cols_count
cols_count = {}

# Extract column from DataFrame: col
col = df[col_name]

# Iterate over the column in DataFrame
for entry in col:

# If entry is in cols_count, add 1
if entry in cols_count.keys():
cols_count[entry] += 1
# Else add the entry to cols_count, set the value to 1
else:
cols_count[entry] = 1

# Return the cols_count dictionary
return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df,'lang')

# Print result1
print(result1)

------------------------------------------------------------------------------------------------------------

'''Python data science Toolbox'''

#ITerators 
# Create a list of strings: flash
flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']

# Print each list item in flash using a for loop
for a in flash:
    print(a)


# Create an iterator for flash: superhero
superhero = iter(flash)

# Print each item from the iterator
print(next(superhero))
print(next(superhero))
print(next(superhero))
print(next(superhero))

------------------------------------------------------------------------------

# Create a range object: values
values = range(10,21)

# Print the range object
print(values)

# Create a list of integers: values_list
values_list = list(values)

# Print values_list
print(values_list)

# Get the sum of values: values_sum
values_sum = sum(values)

# Print values_sum
print(values_sum)


--------------------------------------------------------------------------------

###enumerate

# Create a list of strings: mutants
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pryde']

# Create a list of tuples: mutant_list
mutant_list = list(enumerate(mutants))

# Print the list of tuples
print(mutant_list)

# Unpack and print the tuple pairs
for index1,value1 in enumerate(mutants):
    print(index1, value1)

# Change the start index
for index2,value2 in enumerate(mutants,start = 1):
    print(index2, value2)


------------------------------------------------------------------------------------


#iterator for tuples

# Create a list of tuples: mutant_data
mutant_data = list(zip(mutants,aliases,powers))

# Print the list of tuples
print(mutant_data)

# Create a zip object using the three lists: mutant_zip
mutant_zip = zip(mutants,aliases,powers)

# Print the zip object
print(mutant_zip)

# Unpack the zip object and print the tuple values
for value1, value2, value3 in mutant_zip:
    print(value1, value2, value3)
    
    
--------------------------------------------------------------------------------------

#read twittwer and create dictionary

# Initialize an empty dictionary: counts_dict
counts_dict = {}

# Iterate over the file chunk by chunk
for chunk in pd.read_csv('tweets.csv',chunksize = 10):

    # Iterate over the column in DataFrame
    for entry in chunk['lang']:
        if entry in counts_dict.keys():
            counts_dict[entry] += 1
        else:
            counts_dict[entry] = 1

# Print the populated dictionary
print(counts_dict)

---------------------------------------------------------------------------------------

## in function


# Define count_entries()
def count_entries(csv_file,c_size,colname):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Iterate over the file chunk by chunk
    for chunk in pd.read_csv(csv_file,chunksize = c_size):

        # Iterate over the column in DataFrame
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1

    # Return counts_dict
    return counts_dict

# Call count_entries(): result_counts
result_counts = count_entries('tweets.csv',10,'lang')

# Print result_counts
print(result_counts)


------------------------------------------------------------------------------------------------

'''##List comprehensions'''


print([i**2 for i in range(10)])

--------------------------------------------------------------------------------------------------


''' Nested list comprehensions '''


# Create a 5 x 5 matrix using a list of lists: matrix
matrix = [[col for col in range(5)] for row in range(5)]

# Print the matrix
for row in matrix:
    print(row)


-----------------------------------------------------------------------------------------

'''List comprehension with if'''

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
new_fellowship = [member for member in fellowship if len(member)>=7]

# Print the new list
print(new_fellowship)

--------------------------------------------------------------------------------------------

'''List comprehension with if-else'''

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
new_fellowship = [member if len(member)>=7 else "" for member in fellowship]

# Print the new list
print(new_fellowship)

----------------------------------------------------------------------------------------------

'''List Comprehension Dictionaries '''


# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create dict comprehension: new_fellowship
new_fellowship = { member : len(member) for member in fellowship}

# Print the new dictionary
print(new_fellowship)

---------------------------------------------------------------------------------------------

'''List Comprehension for Generators {they are just iterators}'''

# Create generator object: result
result = (num for num in range(31))

# Print the first 5 values
print(next(result))
print(next(result))
print(next(result))
print(next(result))
print(next(result))

# Print the rest of the values
for value in result:
    print(value)


----------------------------------------------------------------------------------------------

'''Create and print a generator'''


# Create a list of strings: lannister
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Create a generator object: lengths
lengths = (len(person) for person in lannister)

# Iterate over and print the values in lengths
for value in lengths:
    print(value)


-------------------------------------------------------------------------------------------------

'''definea Generator Function '''


# Create a list of strings
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Define generator function get_lengths
def get_lengths(input_list):
    """Generator function that yields the
    length of the strings in input_list."""

    # Yield the length of a string
    for person in input_list:
        yield len(person)

# Print the values generated by get_lengths()
for value in get_lengths(lannister):
    print(value)
# Create a list of strings
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

    
-------------------------------------------------------------------------------


''' Extract time from a tweets dataframe '''

# Extract the created_at column from df: tweet_time
tweet_time = df['created_at']

# Extract the clock time: tweet_clock_time
tweet_clock_time = [entry[11:19] for entry in tweet_time]

# Print the extracted times
print(tweet_clock_time)


----------------------------------------------------------------------------------

''' Conditional list comprehension on the tweet '''

# Extract the created_at column from df: tweet_time
tweet_time = df['created_at']

# Extract the clock time: tweet_clock_time
tweet_clock_time = [entry[11:19] for entry in tweet_time if entry[17:19] == '19']

# Print the extracted times
print(tweet_clock_time)


----------------------------------------------------------------------------------

'''Project'''

# Zip lists: zipped_lists
zipped_lists = zip(feature_names,row_vals)

# Create a dictionary: rs_dict
rs_dict = dict(zipped_lists)

# Print the dictionary
print(rs_dict)


-----------------------------------------------------------------------------------

'''Project'''

# Define lists2dict()
def lists2dict(list1, list2):
    """Return a dictionary where list1 provides
    the keys and list2 provides the values."""

    # Zip lists: zipped_lists
    zipped_lists = zip(list1, list2)

    # Create a dictionary: rs_dict
    rs_dict = dict(zipped_lists)

    # Return the dictionary
    return rs_dict

# Call lists2dict: rs_fxn
rs_fxn = lists2dict(feature_names,row_vals)

# Print rs_fxn
print(rs_fxn)

----------------------------------------------------------------------------------

'''Create a list of dictionaries'''


# Print the first two lists in row_lists
print(row_lists[0])
print(row_lists[1])

# Turn list of lists into list of dicts: list_of_dicts
list_of_dicts = [lists2dict(feature_names,sublist) for sublist in row_lists]

# Print the first two dictionaries in list_of_dicts
print(list_of_dicts[0])
print(list_of_dicts[1])


-----------------------------------------------------------------------------------

'''Creating Dataframe from Dictionary'''

# Import the pandas package
import pandas as pd

# Turn list of lists into list of dicts: list_of_dicts
list_of_dicts = [lists2dict(feature_names, sublist) for sublist in row_lists]

# Turn list of dicts into a DataFrame: df
df = pd.DataFrame(list_of_dicts)

# Print the head of the DataFrame
print(df.head())

----------------------------------------------------------------------------------------

''' Read initial values of file '''

# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Skip the column names
    file.readline()

    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Process only the first 1000 rows
    for j in range(0,1000):

        # Split the current line into a list: line
        line = file.readline().split(',')

        # Get the value for the first column: first_col
        first_col = line[0]

        # If the column value is in the dict, increment its value
        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1

        # Else, add to the dict and set value to 1
        else:
            counts_dict[first_col] = 1

# Print the resulting dictionary
print(counts_dict)

------------------------------------------------------------------------------------------


'''Generator for a very large file'''


# Define read_large_file()
def read_large_file(file_object):
    """A generator function to read a large file lazily."""

    # Loop indefinitely until the end of the file
    while True:

        # Read a line from the file: data
        data = file_object.readline()

        # Break if this is the end of the file
        if not data:
            break

        # Yield the line of data
        yield data
        
# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Create a generator object for the file: gen_file
    gen_file = read_large_file(file)

    # Print the first three lines of the file
    print(next(gen_file))
    print(next(gen_file))
    print(next(gen_file))
    
----------------------------------------------------------------------------------------


'''Continue....'''


# Initialize an empty dictionary: counts_dict
counts_dict = {}

# Open a connection to the file
with open('world_dev_ind.csv') as file :

    # Iterate over the generator from read_large_file()
    for line in read_large_file(file):

        row = line.split(',')
        first_col = row[0]

        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1
        else:
            counts_dict[first_col] = 1

# Print            
print(counts_dict)


--------------------------------------------------------------------------------------

''' Pandas read file in chunks'''

# Import the pandas package
import pandas as pd

# Initialize reader object: df_reader
df_reader = pd.read_csv('ind_pop.csv', chunksize=10)

# Print two chunks
print(next(df_reader))
print(next(df_reader))


----------------------------------------------------------------------------------


''' REading file in chunks for only specific filter '''


# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize= 1000)

# Get the first DataFrame chunk: df_urb_pop
df_urb_pop = next(urb_pop_reader)

# Check out the head of the DataFrame
print(df_urb_pop.head())

# Check out specific country: df_pop_ceb
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

# Zip DataFrame columns of interest: pops
pops = zip(df_pop_ceb['Total Population'], df_pop_ceb['Urban population (% of total)'])

# Turn zip object into list: pops_list
pops_list = list(pops)

# Print pops_list
print(pops_list)

--------------------------------------------------------------------------------

'''Creating plot on it'''


# Code from previous exercise
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)
df_urb_pop = next(urb_pop_reader)
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']
pops = zip(df_pop_ceb['Total Population'], 
           df_pop_ceb['Urban population (% of total)'])
pops_list = list(pops)

# Use list comprehension to create new DataFrame column 'Total Urban Population'
df_pop_ceb['Total Urban Population'] = [int(ctr[0]*ctr[1]/100) for ctr in pops_list]

# Plot urban population data
df_pop_ceb.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()

------------------------------------------------------------------------
'''-----continued'''


# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)

# Initialize empty DataFrame: data
data = pd.DataFrame()

# Iterate over each DataFrame chunk
for df_urb_pop in urb_pop_reader:

    # Check out specific country: df_pop_ceb
    df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

    # Zip DataFrame columns of interest: pops
    pops = zip(df_pop_ceb['Total Population'],
                df_pop_ceb['Urban population (% of total)'])

    # Turn zip object into list: pops_list
    pops_list = list(pops)

    # Use list comprehension to create new DataFrame column 'Total Urban Population'
    df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]
    
    # Append DataFrame chunk to data: data
    data = data.append(df_pop_ceb)

# Plot urban population data
data.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()


----------------------------------------------------------------

''' function of the same implementation Definition'''


# Define plot_pop()
def plot_pop(filename, country_code):

    # Initialize reader object: urb_pop_reader
    urb_pop_reader = pd.read_csv(filename, chunksize=1000)

    # Initialize empty DataFrame: data
    data = pd.DataFrame()
    
    # Iterate over each DataFrame chunk
    for df_urb_pop in urb_pop_reader:
        # Check out specific country: df_pop_ceb
        df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == country_code]

        # Zip DataFrame columns of interest: pops
        pops = zip(df_pop_ceb['Total Population'],
                    df_pop_ceb['Urban population (% of total)'])

        # Turn zip object into list: pops_list
        pops_list = list(pops)

        # Use list comprehension to create new DataFrame column 'Total Urban Population'
        df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]
    
        # Append DataFrame chunk to data: data
        data = data.append(df_pop_ceb)

    # Plot urban population data
    data.plot(kind='scatter', x='Year', y='Total Urban Population')
    plt.show()

# Set the filename: fn
fn = 'ind_pop_data.csv'

# Call plot_pop for country code 'CEB'
plot_pop(fn,'CEB')

# Call plot_pop for country code 'ARB'
plot_pop(fn,'ARB')

-----------------------------------------------------------------------------

''' Letsts begin with Pandas!'''


'''bracket functions

df.head()
df.info()
df.describe()
df[df["col"].isin(["a","b"])]
df.sort_values("")
df["col"].mean()
df["col"].median()
df["col"].max()
df["col"].min()
plt.show(
'''


# Print the head of the homelessness data
print(homelessness.head())

# Print information about homelessness
print(homelessness.info())

# Print the shape of homelessness
print(homelessness.shape)

# Print a description of homelessness
print(homelessness.describe())

------------------------------------------------------------------


# Import pandas using the alias pd
import pandas as pd

# Print the values of homelessness , converts into a 2d numpy array
homelessness.values

# Print the column index of homelessness
homelessness.columns

# Print the row index of homelessness
homelessness.index

-------------------------------------------------------------------

# Sort homelessness by descending family members
homelessness_fam = homelessness.sort_values('family_members',ascending = False)

# Print the top few rows
print(homelessness_fam.head())


---------------------------------------------------------------------

# Sort homelessness by region, then descending family members
homelessness_reg_fam = homelessness.sort_values(["region","family_members"],ascending = [True,False])

# Print the top few rows
print(homelessness_reg_fam.head())

-------------------------------------------------------------------


# Select the state and family_members columns
state_fam = homelessness[["state","family_members"]]

# Print the head of the result
print(state_fam.head())


-------------------------------------------------------------------


# Filter for rows where individuals is greater than 10000
ind_gt_10k = homelessness[homelessness['individuals']>10000]

# See the result
print(ind_gt_10k)

-------------------------------------------------------------------

# Filter for rows where region is Mountain
mountain_reg = homelessness[homelessness["region"]== "Mountain"]

# See the result
print(mountain_reg)

-------------------------------------------------------------------

# Filter for rows where family_members is less than 1000 
# and region is Pacific
fam_lt_1k_pac = homelessness[(homelessness['family_members']<1000) & (homelessness['region'] == "Pacific") ]

# See the result
print(fam_lt_1k_pac)


---------------------------------------------------------------------

# The Mojave Desert states
canu = ["California", "Arizona", "Nevada", "Utah"]

# Filter for rows in the Mojave Desert states
mojave_homelessness = homelessness[homelessness["state"].isin(canu)]

# See the result
print(mojave_homelessness)

--------------------------------------------------------------------

# Add total col as sum of individuals and family_members
homelessness["total"] = homelessness["individuals"]+homelessness["family_members"]

# Add p_individuals col as proportion of individuals
homelessness["p_individuals"] = homelessness["individuals"]/homelessness["total"]

# See the result
print(homelessness)

----------------------------------------------------------------------------


# Create indiv_per_10k col as homeless individuals per 10k state pop
homelessness["indiv_per_10k"] = 10000 * homelessness["individuals"] / homelessness["state_pop"] 

# Subset rows for indiv_per_10k greater than 20
high_homelessness = homelessness[homelessness["indiv_per_10k"]>20]

# Sort high_homelessness by descending indiv_per_10k
high_homelessness_srt = high_homelessness.sort_values("indiv_per_10k",ascending = False)

# From high_homelessness_srt, select the state and indiv_per_10k cols
result = high_homelessness_srt[["state","indiv_per_10k"]]

# See the result
print(result)

---------------------------------------------------------------------------------------

# Print the head of the sales DataFrame
print(sales.head())

# Print the info about the sales DataFrame
print(sales.info())

# Print the mean of weekly_sales
print(sales["weekly_sales"].mean())

# Print the median of weekly_sales
print(sales["weekly_sales"].median())


------------------------------------------------------------------------------------------

# Print the maximum of the date column
print(sales["date"].max())

# Print the minimum of the date column
print(sales["date"].min())

-----------------------------------------------------------------------------------------
'''.agg()'''

# A custom IQR function
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)
    
# Print IQR of the temperature_c column
print(sales["temperature_c"].agg(iqr))

-----------------------------------------------------------------------------------------


# A custom IQR function
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)

# Update to print IQR of temperature_c, fuel_price_usd_per_l, & unemployment
print(sales[["temperature_c","fuel_price_usd_per_l","unemployment"]].agg(iqr))


--------------------------------------------------------------------------------------------

# Import NumPy and create custom IQR function
import numpy as np
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)

# Update to print IQR and median of temperature_c, fuel_price_usd_per_l, & unemployment
print(sales[["temperature_c", "fuel_price_usd_per_l", "unemployment"]].agg([iqr,np.median]))

--------------------------------------------------------------------------------------------

# Sort sales_1_1 by date
sales_1_1 = sales_1_1.sort_values("date")

# Get the cumulative sum of weekly_sales, add as cum_weekly_sales col
sales_1_1["cum_weekly_sales"] = sales["weekly_sales"].cumsum()

# Get the cumulative max of weekly_sales, add as cum_max_sales col
sales_1_1["cum_max_sales"] = sales["weekly_sales"].cummax()

# See the columns you calculated
print(sales_1_1[["date", "weekly_sales", "cum_weekly_sales", "cum_max_sales"]])

---------------------------------------------------------------------------------------------
'''.apply() , use lambda as .apply( lamda x : x/2) ***************************** 

NOTE - .apply can't use multiplefunctions as list as arguments, use .agg in that case'''

# Write a function to convert degrees Fahrenheit to degrees Celsius: to_celsius
def to_celsius(F):
    return 5/9*(F - 32)

# Apply the function over 'Mean TemperatureF' and 'Mean Dew PointF': df_celsius
df_celsius = weather[['Mean TemperatureF','Mean Dew PointF']].apply(to_celsius)

# Reassign the column labels of df_celsius
df_celsius.columns = ['Mean TemperatureC', 'Mean Dew PointC']

# Print the output of df_celsius.head()
print(df_celsius.head())

---------------------------------------------------------------------------------------------

'''.map()'''

# Create the dictionary: red_vs_blue
red_vs_blue = {'Obama':'blue', 'Romney':'red'}

# Use the dictionary to map the 'winner' column to the new column: election['color']
election['color'] = election["winner"].map(red_vs_blue)

# Print the output of election.head()
print(election.head())


--------------------------------------------------------------------------------------------


# Import zscore from scipy.stats
from scipy.stats import zscore

# Call zscore with election['turnout'] as input: turnout_zscore
turnout_zscore = zscore(election['turnout'])

# Print the type of turnout_zscore
print(type(turnout_zscore))

# Assign turnout_zscore to a new column: election['turnout_zscore']
election['turnout_zscore'] = turnout_zscore

# Print the output of election.head()
print(election.head())

-----------------------------------------------------------------------------------------------

#remove a column

df.drop(['B', 'C'], axis=1)

#remove a row based on conditions

SNP_data = SNP_data.drop(SNP_data[ SNP_data['Volume'] == 0 ].index, axis=0)

----------------------------------------------------------------------------------------------


# Drop duplicate store/type combinations
store_types = sales.drop_duplicates(subset = ["store","type"])
print(store_types.head())

# Drop duplicate store/department combinations
store_depts = sales.drop_duplicates(subset = ["store","department"])
print(store_depts.head())

# Subset the rows that are holiday weeks and drop duplicate dates
holiday_dates = sales[sales["is_holiday"]].drop_duplicates(subset = "date")

# Print date col of holiday_dates
print(holiday_dates[["date"]])

---------------------------------------------------------------------------------------------------
# Count the number of stores of each type
store_counts = stores.type.value_counts()
print(store_counts)

# Get the proportion of stores of each type
store_props = stores.type.value_counts(normalize = True)
print(store_props)

# Count the number of departments of each type and sort
dept_counts_sorted = departments.department.value_counts(sort = True)
print(dept_counts_sorted)

# Get the proportion of departments of each type and sort
dept_props_sorted = departments.department.value_counts(sort=True, normalize=True)
print(dept_props_sorted)

-------------------------------------------------------------------------------------------------

# Calc total weekly sales
sales_all = sales["weekly_sales"].sum()

# Subset for type A stores, calc total weekly sales
sales_A = sales[sales["type"] == "A"]["weekly_sales"].sum()

# Subset for type B stores, calc total weekly sales
sales_B = sales[sales["type"] == "B"]["weekly_sales"].sum()

# Subset for type C stores, calc total weekly sales
sales_C = sales[sales["type"] == "C"]["weekly_sales"].sum()

# Get proportion for each type
sales_propn_by_type = [sales_A, sales_B, sales_C] / sales_all
print(sales_propn_by_type)

-------------------------------------------------------------------------------------------------------


# Group by type; calc total weekly sales
sales_by_type = sales.groupby("type")["weekly_sales"].sum()

# Get proportion for each type
sales_propn_by_type = sales_by_type/sales_by_type.sum()
print(sales_propn_by_type)

-------------------------------------------------------------------------------------------


# Import NumPy with the alias np
import numpy as np

# For each store type, aggregate weekly_sales: get min, max, mean, and median
sales_stats = sales.groupby("type")["weekly_sales"].agg([np.min,np.max,np.mean,np.median])

# Print sales_stats
print(sales_stats)

# For each store type, aggregate unemployment and fuel_price_usd_per_l: get min, max, mean, and median
unemp_fuel_stats = sales.groupby("type")["unemployment","fuel_price_usd_per_l"].agg([np.min,np.max,np.mean,np.median])

# Print unemp_fuel_stats
print(unemp_fuel_stats)


-----------------------------------------------------------------------------------------

# Pivot for mean weekly_sales for each store type
mean_sales_by_type = sales.pivot_table(values = "weekly_sales",index = "type")

# Print mean_sales_by_type
print(mean_sales_by_type)

# Import NumPy as np
import numpy as np

# Pivot for mean and median weekly_sales for each store type
mean_med_sales_by_type = sales.pivot_table(values = "weekly_sales",index = "type", aggfunc= [np.mean,np.median] )

# Print mean_med_sales_by_type
print(mean_med_sales_by_type)

# Pivot for mean weekly_sales by store type and holiday 
mean_sales_by_type_holiday = sales.pivot_table(values = "weekly_sales",index  = "type",columns = "is_holiday" )

# Print mean_sales_by_type_holiday
print(mean_sales_by_type_holiday)
------------------------------------------------------------------------------------------------

# Print mean weekly_sales by department and type; fill missing values with 0
print(sales.pivot_table(values = "weekly_sales",index = "type", columns = "department", fill_value = 0))

# Print the mean weekly_sales by department and type; fill missing values with 0s; sum all rows and cols
print(sales.pivot_table(values="weekly_sales", index="department", columns="type", margins = True))

---------------------------------------------------------------------------------------------
#isin
#dogs[dogs["name"].isin(["Bella","Stella"])

# Indexing

# Look at temperatures
print(temperatures)

# Index temperatures by city
temperatures_ind = temperatures.set_index("city")

# Look at temperatures_ind
print(temperatures_ind)

# Reset the index, keeping its contents
print(temperatures_ind.reset_index())

# Reset the index, dropping its contents
print(temperatures_ind.reset_index(drop = True))

---------------------------------------------------------------------------------------------


# Make a list of cities to subset on
cities = ["Moscow","Saint Petersburg"]

# Subset temperatures using square brackets
print(temperatures[temperatures["city"].isin(cities)])

# Subset temperatures_ind using .loc[]
print(temperatures_ind.loc[cities])

--------------------------------------------------------------------------------------------------
# Index temperatures by country & city
temperatures_ind = temperatures.set_index(["country","city"])

# List of tuples: Brazil, Rio De Janeiro & Pakistan, Lahore
rows_to_keep = [( "Brazil","Rio De Janeiro"),("Pakistan","Lahore")]

# Subset for rows to keep
print(temperatures_ind.loc[rows_to_keep])

----------------------------------------------------------------------------------------------


# Sort temperatures_ind by index values
print(temperatures_ind.sort_index())

# Sort temperatures_ind by index values at the city level
print(temperatures_ind.sort_index(level = "city"))

# Sort temperatures_ind by country then descending city
print(temperatures_ind.sort_index(level = ["country","city"],ascending= [True,False]))

---------------------------------------------------------------------------------------

#You can only slice an index if the index is sorted

# Sort the index of temperatures_ind
temperatures_srt = temperatures_ind.sort_index()

# Incorrectly subset rows from Pakistan to Russia
print(temperatures_srt.loc["Pakistan":"Russia"])

# Subset rows from Lahore to Moscow
print(temperatures_srt.loc["Lahore":"Moscow"])

# Subset rows from Pakistan, Lahore to Russia, Moscow
print(temperatures_srt.loc[("Pakistan", "Lahore"):("Russia","Moscow")])

---------------------------------------------------------------------------------


# Subset rows from India, Hyderabad to Iraq, Baghdad
print(temperatures_srt.loc[("India", "Hyderabad") : ("Iraq", "Baghdad")])

# Subset columns from date to avg_temp_c
print(temperatures_srt.loc[:,"date" :"avg_temp_c"])

# Subset in both directions at once
print(temperatures_srt.loc[("India", "Hyderabad") : ("Iraq", "Baghdad"),"date" :"avg_temp_c"])



----------------------------------------------------------------------------------

''' Most normal filter slicing'''
# Use Boolean conditions to subset temperatures for rows in 2010 and 2011
print(temperatures[((temperatures["date"] >= "2010")&(temperatures["date"] < "2012"))])

# Set date as an index
temperatures_ind = temperatures.set_index("date")

# Use .loc[] to subset temperatures_ind for rows in 2010 and 2011
print(temperatures_ind.loc["2010":"2011"])

# Use .loc[] to subset temperatures_ind for rows from Aug 2010 to Feb 2011
print(temperatures_ind.loc["2010-08":"2011-02"])

------------------------------------------------------------------------------------


# Get 23rd row, 2nd column (index 22, 1)
print(temperatures.iloc[22,1])

# Use slicing to get the first 5 rows
print(temperatures.iloc[0:5,:])

# Use slicing to get columns 2 to 3
print(temperatures.iloc[:,2:4])

# Use slicing in both directions at once
print(temperatures.iloc[0:5,2:4])


-----------------------------------------------------------------------------------

# Get 23rd row, 2nd column (index 22, 1)
print(temperatures.iloc[22,1])

# Use slicing to get the first 5 rows
print(temperatures.iloc[0:5,:])

# Use slicing to get columns 2 to 3
print(temperatures.iloc[:,2:4])

# Use slicing in both directions at once
print(temperatures.iloc[0:5,2:4])

------------------------------------------------------------------------------------

# Add a year column to temperatures
temperatures["year"] = temperatures.date.dt.year

# Pivot avg_temp_c by country and city vs year
temp_by_country_city_vs_year = temperatures.pivot_table(values = "avg_temp_c", index = ["country","city"], columns = "year")

# See the result
print(temp_by_country_city_vs_year)

--------------------------------------------------------------------------------------

# Subset for Egypt to India
print(temp_by_country_city_vs_year.loc["Egypt":"India"])

# Subset for Egypt, Cairo to India, Delhi
print(temp_by_country_city_vs_year.loc[("Egypt", "Cairo"):("India","Delhi")])

# Subset in both directions at once
print(temp_by_country_city_vs_year.loc[("Egypt", "Cairo"):("India","Delhi"),"2005":"2010"])


--------------------------------------------------------------------------------------------------


# Get the worldwide mean temp by year
mean_temp_by_year = temp_by_country_city_vs_year.mean()


''' Display the min of the df'''
# Filter for the year that had the highest mean temp
print(mean_temp_by_year[mean_temp_by_year == mean_temp_by_year.max()])

# Get the mean temp by city
mean_temp_by_city = temp_by_country_city_vs_year.mean(axis = "columns")

# Filter for the city that had the lowest mean temp
print(mean_temp_by_city[mean_temp_by_city == mean_temp_by_city.min()])

-----------------------------------------------------------------------------------------------

''' Matplotlib.plt'''

# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Look at the first few rows of data
print(avocados.head())

# Get the total number of avocados sold of each size
nb_sold_by_size = avocados.groupby("size")["nb_sold"].sum()


# Create a bar plot of the number of avocados sold by size
nb_sold_by_size.plot(kind = "bar")

# Show the plot
plt.show()



-----------------------------------------------------------------------------------------------
# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Get the total number of avocados sold on each date
nb_sold_by_date = avocados.groupby("date")["nb_sold"].sum()

# Create a line plot of the number of avocados sold by date
nb_sold_by_date.plot(kind = "line")

# Show the plot
plt.show()

-----------------------------------------------------------------------------------------------

# Scatter plot of nb_sold vs avg_price with title
avocados.plot(x = "nb_sold",y = "avg_price", title = "Number of avocados sold vs. average price", kind = "scatter")

# Show the plot
plt.show()

-------------------------------------------------------------------------------------------------

# Modify bins to 20
avocados[avocados["type"] == "conventional"]["avg_price"].hist(alpha=0.5,bins = 20)

# Modify bins to 20
avocados[avocados["type"] == "organic"]["avg_price"].hist(alpha=0.5,bins = 20)

# Add a legend
plt.legend(["conventional", "organic"])

# Show the plot
plt.show()


-------------------------------------------------------------------------------------------------

''' NA values'''

# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Check individual values for missing values
print(avocados_2016.isna())

# Check each column for missing values
print(avocados_2016.isna().any())

# Bar plot of missing values by variable
print(avocados_2016.isna().sum().plot(kind = "bar"))

# Show plot
plt.show()

---------------------------------------------------------------------------------------------------


# Remove rows with missing values
avocados_complete = avocados_2016.dropna()

# Check if any columns contain missing values
print(avocados_complete.isna().any())


---------------------------------------------------------------------------------------------------

# From previous step
cols_with_missing = ["small_sold", "large_sold", "xl_sold"]
avocados_2016[cols_with_missing].hist()
plt.show()

# Fill in missing values with 0
avocados_filled = avocados_2016.fillna(0)

# Create histograms of the filled columns
avocados_filled[cols_with_missing].hist()

# Show the plot
plt.show()

-----------------------------------------------------------------------------------------------------
''' Creating a dataframe'''

# Create a list of dictionaries with new data
avocados_list = [
    {"date": "2019-11-03", "small_sold": 10376832, "large_sold": 7835071},
    {"date": "2019-11-10", "small_sold": 10717154, "large_sold": 8561348},
]

# Convert list into DataFrame
avocados_2019 = pd.DataFrame(avocados_list)

# Print the new DataFrame
print(avocados_2019)
------------------------------------------------------------------------------------------------


# Create a dictionary of lists with new data
avocados_dict = {
  "date": ["2019-11-17","2019-12-01"],
  "small_sold": [10859987,9291631],
  "large_sold": [7674135,6238096]
}

# Convert dictionary into DataFrame
avocados_2019 = pd.DataFrame(avocados_dict)

# Print the new DataFrame
print(avocados_2019)

--------------------------------------------------------------------------------------------

'''Reading data'''


# From previous steps
airline_bumping = pd.read_csv("airline_bumping.csv")
print(airline_bumping.head())
airline_totals = airline_bumping.groupby("airline")[["nb_bumped", "total_passengers"]].sum()
airline_totals["bumps_per_10k"] = airline_totals["nb_bumped"] / airline_totals["total_passengers"] * 10000

# Print airline_totals
print(airline_totals)


-----------------------------------------------------------------------------------------------------


'''Writing file'''


# Create airline_totals_sorted
airline_totals_sorted = airline_totals.sort_values("bumps_per_10k",ascending=False)

# Print airline_totals_sorted
print(airline_totals_sorted)

# Save as airline_totals_sorted.csv
airline_totals_sorted.to_csv("airline_totals_sorted.csv")


-----------------------------------------------------------------------------------------------------

'''REad files '''

# Read in the file: df1
df1 = pd.read_csv(data_file)

# Create a list of the new column labels: new_labels
new_labels = ["year","population"]

# Read in the file, specifying the header and names parameters: df2
df2 = pd.read_csv(data_file, header=0, names=new_labels)

# Print both the DataFrames
print(df1)
print(df2)

----------------------------------------------------------------------------------------------------

'''SAve files with delim and cleaning '''


# Read the raw file as-is: df1
df1 = pd.read_csv(file_messy)

# Print the output of df1.head()
print(df1.head())

# Read in the file with the correct parameters: df2
df2 = pd.read_csv(file_messy, delimiter=' ', header=3, comment='#')

# Print the output of df2.head()
print(df2.head())

# Save the cleaned up DataFrame to a CSV file without the index
df2.to_csv(file_clean, index=False)

# Save the cleaned up DataFrame to an excel file without the index
df2.to_excel('file_clean.xlsx', index=False)

-------------------------------------------------------------------------------------------------

#you can apply a .transform() method after grouping to apply a function to groups of data independently.

# Import zscore
from scipy.stats import zscore

# Group gapminder_2010: standardized
standardized = gapminder_2010.groupby("region")['life','fertility'].transform(zscore)

# Construct a Boolean Series to identify outliers: outliers
outliers = (standardized['life'] < -3) | (standardized['fertility'] > 3)

# Filter gapminder_2010 by the outliers: gm_outliers
gm_outliers = gapminder_2010.loc[outliers]

# Print gm_outliers
print(gm_outliers)

-------------------------------------------------------------------------------------------


# Create a groupby object: by_sex_class
by_sex_class = titanic.groupby(['sex','pclass'])

# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.median())

# Impute age and assign to titanic['age']
titanic.age = by_sex_class["age"].transform(impute_median)

# Print the output of titanic.tail(10)
print(titanic.tail(10))

-------------------------------------------------------------------------------------------

# Group gapminder_2010 by 'region': regional
regional = gapminder_2010.groupby("region")

#returns an object

# Apply the disparity function on regional: reg_disp
reg_disp = regional.apply(disparity)

# Print the disparity of 'United States', 'United Kingdom', and 'China'
print(reg_disp.loc[['United States','United Kingdom','China']])

---------------------------------------------------------------------------------------------

def c_deck_survival(gr):

    c_passengers = gr['cabin'].str.startswith('C').fillna(False)

    return gr.loc[c_passengers, 'survived'].mean()

# Create a groupby object using titanic over the 'sex' column: by_sex
by_sex = titanic.groupby('sex')

# Call by_sex.apply with the function c_deck_survival
c_surv_by_sex = by_sex.apply(c_deck_survival)

# Print the survival rates
print(c_surv_by_sex)

-------------------------------------------------------------------------------------

### filtering with lambda

# Read the CSV file into a DataFrame: sales
sales = pd.read_csv('sales.csv', index_col='Date', parse_dates=True)

# Group sales by 'Company': by_company
by_company = sales.groupby('Company')

# Compute the sum of the 'Units' of by_company: by_com_sum
by_com_sum = by_company['Units'].sum()
print(by_com_sum)

# Filter 'Units' where the sum is > 35: by_com_filt
by_com_filt = by_company.filter(lambda g:g['Units'].sum()>35)
print(by_com_filt)

--------------------------------------------------------------------------------------

## filtering with .filter()


# Create the Boolean Series: under10
under10 = (titanic['age'] < 10).map({True:'under 10', False:'over 10'})

# Group by under10 and compute the survival rate
survived_mean_1 = titanic.groupby(under10)["survived"].mean()
print(survived_mean_1)

# Group by under10 and pclass and compute the survival rate
survived_mean_2 = titanic.groupby([under10,'pclass'])["survived"].mean()
print(survived_mean_2)

--------------------------------------------------------------------------------------

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


''' Conditional filtering of a list: in simple words, 
the filter() method filters the given iterable with the help of 
a function that tests each element in the iterable to be true or not.'''

numbers = [1, 6, 3, 8, 4, 9]

list(filter(lambda x: x<5, numbers))

#get first index {match in R} 
numbers.index['3']

#Comprehensive list of indices which matches a value
[ i for i in range(len(numbers)) if numbers[i] == 1 ]

#or
#s[s.index.isin([2, 4, 6])]
'''This is the best'''#SNP_data.columns.get_loc("Close")

#Get the list items based on custom indices
[numbers[p[i]] for i in range(len(p))]

----------------------------

#Reverse a list 

nums[::-1]

------------------------------

#Get present working directory 

import os
os.getcwd()

# Get files in current directory

os.listdir()


------------


#Merge a data frame

df_outer = pd.merge(df1, df2, on='id', how='outer')

---------


#Resample

df.resample('W').mean()


------------------

#Get working directory/ #change working directory

os.getcwd()
os.chdir('../')



----------------------------

#Shortcut editor in jupyterlab

{
       "shortcuts": [
        {
            "command": "notebook:run-all-above",
             "keys": [
             "Ctrl Alt B"
             ],
             "selector": ".jp-Notebook:focus",
             "title": "Run All Above",
             "category": "Notebook Cell Operations"
        }
    ]
}


--------------------------------------

#cbind in python

df3 = pd.concat([df1, df2], axis=1, ignore_index=True)



----------------------------------------------



#expand the no. of rows in display

pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 25)
pd.set_option('display.min_rows', 25)
pd.set_option('display.expand_frame_repr', True)


---------------------------------

#Most frequently occuring value in dataframe 

df['hour'].mode()