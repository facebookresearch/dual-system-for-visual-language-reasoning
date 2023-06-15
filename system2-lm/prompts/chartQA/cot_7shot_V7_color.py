PROMPT = """
Answer the following questions step by step.

Q: In which year the private health expenditure per person in Oman is 210.69?
A: Let's extract the legends of the chart.
The legends are: (1) 2008 (2) 2009 (3) 2010 (4) 2011 (5) 2012 (6) 2013 (7) 2014.
Let's extract the data of Oman.
The data is 183.88 in 2008, 233.80 in 2009, 210.69 in 2010, 195.26 in 2011, 196.32 in 2012, 154.21 in 2013, 153.22 in 2014.
The value 210.69 is in year 2010. So the answer is 2010.

Q: What is the ratio between green and grey bars in the chart?
A: Let's extract the legends of the chart.
The legends are: (1) Value.
Let's find out what the color green represents.
The color green represents Very/somewhat serious threat.
Let's extract the data of Very/somewhat serious threat.
The data is 76.0.
Let's find out what the color grey represents.
The color grey represents Minor/not threat.
Let's extract the data of Minor/not threat.
The data is 7.0.
The ratio between 76.0 and 7.0 is 76.0/7.0=10.86. So the answer is 10.86.

Q: By how many points does NET Excellent/good surpass NET Only fair/poor in the year of 2018?
A: Let's extract the legends of the chart.
The legends are: (1) NET Excellent/ good (2) NET Only fair/ poor.
Let's extract the data of NET Excellent/good.
The data is 54.00.
Let's extract the data of in NET Only fair/poor.
The data is 39.00.
54.00 surpasses 39.00 by 54.00-39.00=15.00. So the answer is 15.00.

Q: How many countries have a value below 40%?
A: Let's extract the legends of the chart.
The legends are: (1) Share of respondents who believe Coronavirus is a threat.
Let's extract the data of Share of respondents who believe Coronavirus is a threat.
The data is 60.00 in United Kingdom, 50.00 in Italy, 38.00 in Germany, 29.00 in Finland. 
The values that are below 40.00 are [38.00, 29.00]. So the answer is 2.

Q: How much money did Activision Blizzard's console segment generate in annual revenues in 2020?
A: Let's extract the legends of the chart.
The legends are: (1) Consoles (2) PC (3) Mobile and ancillary (4) Other.
Let's extract the data of Console in 2020.
The data is 2784.00.
The Console segment generated 2784.00 in annual revenues in 2020. So the answer is 2784.00.

Q: What is the value of highest blue bar?
A: Let's extract the legends of the chart.
The legends are: (1) Not at all (2) 1 time (3) 2-3 times (4) 4 times or more.
Let's find out what the color blue represents.
The color blue represents Not at all.
Let's extract the data of Not at all.
The data is 70 in Visit close friends or family, 67 in Go to work, 57 in Exercise, 20 in Shop for food.
Among [70, 67, 57, 20], 70 is the largest. So the answer is 70.

Q: Is the sum of two smallest segments greater than the largest segment?
A: Let's extract the legends of the chart.
The legends are: (1) Share of respondents.
Let's extract the data of Share of respondents.
The data is 81.00 in Decreased, 16.00 in No impact, 3.00 in Increased.
Among [81.00, 16.00, 3.00], the two smallest values are 16.00 and 3.00 while the largest value is 81.00. 16.00+3.00=19.00, which is smaller than 81.00. So the answer is no.

Q: {question}
A:
""".strip()

# two_col_101055, 00268016000832, 4549, two_col_42979, multi_col_1490, two_col_62894
# Q: What was the percentage of the Malaysian population that had access to improved sanitation in 2013?
# A: What is the value of Malaysian population that had access to improved sanitation in 2013 according to the figure?
# The value is 96.00.
# So the answer is 96.00.

