PROMPT = """
Answer the following questions step by step.

Q: In which year, was the value of the line maximum?
A: Let's extract the data of Rural poverty headcount ratio.
The data is 59.6 in 1996, 52.1 in 2001, 55.0 in 2007.
The maximum value is 59.6 in 1996. So the answer is 1996.
 
Q: What is the difference between the under 5 mortality rate in Namibia in 1990 and the under 5 mortality rate in Yemen, Rep. in 2000?
A: Let's extract the data of Namibia BY 1990.
The data is 41.2.
Let's extract the data of Yemen, Rep. BY 2000.
The data is 50.2.
The difference is 41.2-50.2=-9.0. So the answer is -9.0.

Q: In how many countrys, is the amount in kilograms of the orange bar greater than 51?
A: Let's extract the data of 2003.
The data is 23.06 in Guyana, 54.9 in Honduras, 359.8 in Hong Kong, 95.5 in Hungary.
The values that are greater than 51 are [54.9, 359.8, 95.5]. So the answer is 3.

Q: What is the revenue of Taxes generated in Cameroon in the year of 2018?
A: Let's extract the data of Cameroon BY Taxes.
The data is 9.77.
The revenue of Taxes generated in Cameroon is 9.77. So the answer is 9.77.

Q: Is the % of children under 5 of Female dot in 2000 less than that in 2006?
A: Let's extract the data of Female BY 2000.
The data is 11.9.
Let's extract the data of Female BY 2006.
The data is 6.6.
11.9 is not less than 6.6. So the answer is no.

Q: {question}
A:
""".strip()


