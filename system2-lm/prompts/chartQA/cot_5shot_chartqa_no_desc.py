PROMPT = """
Answer the following questions step by step.

Q: In which year the private health expenditure per person in Oman is 210.69?
A: Let's extract the data of Oman.
The data is 183.88 in 2008, 233.80 in 2009, 210.69 in 2010, 195.26 in 2011, 196.32 in 2012, 154.21 in 2013, 153.22 in 2014.
The value 210.69 is in year 2010. So the answer is 2010.

Q: By how many points does NET Excellent/good surpass NET Only fair/poor in German in the year of 2018?
A: Let's extract the data of NET Excellent/ good BY German.
The data is 54.00.
Let's extract the data of NET Only fair/ poor BY German.
The data is 39.00.
54.00 surpasses 39.00 by 54.00-39.00=15.00. So the answer is 15.00.

Q: How many perceptions have a value below 40% in America?
A: Let's extract the data of Share of respondents.
The data is 4.00 in Very positive, 41.00 in Fairly positive, 50.00 in Fairly negative, 11.00 in Very negative.
The values that are below 40.00 are [4.00, 11.00]. So the answer is 2.

Q: In 2020, how much money did Activision Blizzard's console segment generate in annual revenues in Australia?
A: Let's extract the data of Consoles BY 2020.
The data is 2784.00.
The Console segment generated 2784.00 in annual revenues in 2020. So the answer is 2784.00.

Q: Is the sum of two smallest segments greater than the largest segment?
A: Let's extract the data of Value.
The data is 81.00 in Decreased, 16.00 in No impact, 3.00 in Increased.
Among [81.00, 16.00, 3.00], the two smallest values are 16.00 and 3.00 while the largest value is 81.00. 16.00+3.00=19.00, which is smaller than 81.00. So the answer is no.

Q: {question}
A:
""".strip()
