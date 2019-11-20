# Jupyter-Fuzzy-Logic-System

For our Solution Design we have implemented a fuzzy logic system to operate the auto-braking function in a car. To do this our system takes crisp inputs of the cars Current Speed and the Distance between cars/objects in the collision path. 
To prove our solution, we have used the Scikit-fuzzy library in Python to show case the logic and test our system through test cases and data visualizations. We have used Jupyter notebook, a popular Python environment to both generate the prescribed system into working code and to document the different steps and methods our system takes. To simplify the task of explaining our Fuzzy logic system solution we have divided the solution design into subheadings according to the diagram below to show the process of the system from beginning to completion. 

# Fuzzifier
Each input has a scoring mechanism of four different levels. 
Our inputs and score qualifications are as follows:
Speed: { Very Slow [0, 0, 33.33], Slow [0, 33.33, 66.66], Fast [33.33, 66.66, 100], Very Fast [66.66, 100, 100]} 
Distance: {Very Close [0, 0, 3.33], Close [0, 3.33, 6.66], Far [3.33, 6.66, 100], Very Far [6.66, 100, 100]} 

These Crisp inputs will be the basis of fuzzy logic system and will result in a crisp output regarding the breaking force to be applied by the car controller.
Our output and score qualifiers is as follows:
Braking Force: {Very Light [0, 0, 33.33], Light [0, 33.33, 66.66], Heavy [33.33, 66.66, 100], Very Heavy [66.66, 100, 100]}  
From these quantified fuzzy sets, we can graph the membership functions of each input and output variables, which in our systems case is Speed, Distance and Braking Force.

# Rules
The next step of the fuzzy rule-based system is creating a set of rules set to inform our inference engine based on fuzzified input memberships given to the car controller. 
Our memberships set of speed and distance have four membership functions each therefore we need to generate sixteen rules to provide knowledge to our fuzzy Logic System. [1] Our rule solution references the research paper ‘Automatic Braking System in Train using Fuzzy Logic’ for the basis of rule formation for our inference engine as we did not have the current knowledge required and needed to research previous solutions to inform the basis of our rule set. We have altered the statements and antecedents to better fit our unique solution for the car controllers auto-breaking function.
These rules include:
Rule 1: IF Distance is Very Close AND Speed is Very Slow THEN Braking Force is Very Light 
Rule 2: IF Distance is Very Close AND Speed is Slow THEN Braking Force is Light
Rule 3: IF Distance is Very Close AND Speed is Fast THEN Braking Force is Heavy
Rule 4: IF Distance is Very Close AND Speed is Very Fast THEN Braking Force is Very Heavy
Rule 5: IF Distance is Close AND Speed is Very Slow THEN Braking Force is Very Light
Rule 6: IF Distance is Close AND Speed is Slow THEN Braking Force is Light
Rule 7: IF Distance is Close AND Speed is Fast THEN Braking Force is Heavy
Rule 8: IF Distance is Close AND Speed is Very Fast THEN Braking Force is Very Heavy
Rule 9: IF Distance is Far AND Speed is Very Slow THEN Braking Force is Very Light
Rule 10: IF Distance is Far AND Speed is Slow THEN Braking Force is Light
Rule 11: IF Distance is Far AND Speed is Fast THEN Braking Force is Heavy
Rule 12: IF Distance is Far AND Speed is Very Fast THEN Braking Force is Heavy
Rule 13: IF Distance is Very Far AND Speed is Very Slow THEN Braking Force is Very Light
Rule 14: IF Distance is Very Far AND Speed is Slow THEN Braking Force is Very Light
Rule 15: IF Distance is Very Far AND Speed is Fast THEN Braking Force is Light
Rule 16: IF Distance is Very Far AND Speed is Very Fast THEN Braking Force is Light
The Rule Chart Below shows the relation between antecedents and resulting output membership.
	Very Far	Far	Close	Very Close
Very Slow	Very Light	Very Light	Very Light	Very Light
Slow	Very Light	Light	Light	Light
Fast	Light	Heavy	Heavy	Heavy
Very Fast	Light	Heavy	Very Heavy	Very Heavy

# Inference engine
In the application of our car controllers fuzzified inputs and rules, the inference engine will calculate fuzzy membership functions from the input values.
Consider our first rule, 
IF distance is very close AND speed is very slow THEN Braking force is very light.
The system will run the fuzzified input values through each rule and apply a minimum function to combine two antecedents (eg, distance very close and speed very slow). 

Using the minimum function result we use Mamdani’s method to compare our rule result to the output membership function (breaking force) according to the rule 1 of our rule set. This creates a fuzzified output set by taking the minimum value again.

In our function we add the rule data to a list that we return once all membership output functions are completed.

With all our output membership functions combined into a list we use a maximum operator to aggregate values for the fuzzy output set.

# De-fuzzifier
For our de-fuzzification step we have applied the centroid method to find the middle value of our fuzzy output set. This method will return a crisp number by taking the median or average value from our output set calculated in the aggregation step. This crisp output value is how much breaking force the car controller needs to apply in order to avoid a collision. In our system, the crisp output value is a percentage in the range of zero being no breaking force needed and one hundred, being maximum breaking force needed.
