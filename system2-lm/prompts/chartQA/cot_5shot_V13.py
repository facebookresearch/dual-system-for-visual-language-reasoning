PROMPT = """
Answer the following questions step by step.

Q: In which year the private health expenditure per person in Oman is 210.69?
A: Let's describe the figure.
The figure shows the data of: Oman (brown) | Samoa (dark blue). The x-axis shows: 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014.
Let's extract the data of Oman.
The data is 183.88 in 2008, 233.80 in 2009, 210.69 in 2010, 195.26 in 2011, 196.32 in 2012, 154.21 in 2013, 153.22 in 2014.
The value 210.69 is in year 2010. So the answer is 2010.

Q: By how many points does NET Excellent/good surpass NET Only fair/poor in German in the year of 2018?
A: Let's describe the figure.
The figure shows the data of: NET Excellent/ good (blue) | NET Only fair/ poor (orange). The x-axis shows: Brazil, German, Russia, U.S., Japan.
Let's extract the data of NET Excellent/ good BY German.
The data is 54.00.
Let's extract the data of NET Only fair/ poor BY German.
The data is 39.00.
54.00 surpasses 39.00 by 54.00-39.00=15.00. So the answer is 15.00.

Q: How many perceptions have a value below 40% in America?
A: Let's describe the figure.
The figure shows the data of: Share of respondents (blue). The x-axis shows: Very positive, Fairly positive, Fairly negative, Very negative.
Let's extract the data of Share of respondents.
The data is 4.00 in Very positive, 41.00 in Fairly positive, 50.00 in Fairly negative, 11.00 in Very negative.
The values that are below 40.00 are [4.00, 11.00]. So the answer is 2.

Q: In 2020, how much money did Activision Blizzard's console segment generate in annual revenues in Australia?
A: Let's describe the figure.
The figure shows the data of: Consoles (blue) | PC* (dark blue) | Mobile and ancillary** (grey) | Other (dard red). The x-axis shows: 2019 | 2020 | 2021 | 2022.
Let's extract the data of Consoles BY 2020.
The data is 2784.00.
The Console segment generated 2784.00 in annual revenues in 2020. So the answer is 2784.00.

Q: Is the sum of two smallest segments greater than the largest segment?
A: Let's describe the figure.
The figure shows the data of: Value. The x-axis shows: Decreased | No impact | Increased.
Let's extract the data of Value.
The data is 81.00 in Decreased, 16.00 in No impact, 3.00 in Increased.
Among [81.00, 16.00, 3.00], the two smallest values are 16.00 and 3.00 while the largest value is 81.00. 16.00+3.00=19.00, which is smaller than 81.00. So the answer is no.

Q: {question}
A:
""".strip()

# two_col_101055, 00268016000832, 4549, two_col_42979, multi_col_1490, two_col_62894, multi_col_41001, 3730.png
# Q: What was the percentage of the Malaysian population that had access to improved sanitation in 2013?
# A: What is the value of Malaysian population that had access to improved sanitation in 2013 according to the figure?
# The value is 96.00.
# So the answer is 96.00.

# The legends are: 2008 (light blue) | 2009 (light blue) | 2010 (light blue) | 2011 (light blue) | 2012 (light blue) | 2013 (light blue) | 2014 (light blue).
