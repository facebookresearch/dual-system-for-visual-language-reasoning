PROMPT = """
Answer the following questions step by step.

Q: Which group of bars contains the smallest valued individual bar in the whole chart?
A: Let's extract the data of nobody.
The data is 2.0 in size, 3.0 in clerk.
Let's extract the data of today.
The data is 4.0 in size, 8.0 in clerk.
Let's extract the data of phrase.
The data is 8.0 in size, 7.0 in clerk.
Let's extract the data of heart.
The data is 8.0 in size, 7.0 in clerk.
The value 2.0 in the group of nobody is the smallest. So the answer is nobody.

Q: Which stack of bars has the smallest summed value?
A: Let's extract the data of supply.
The data is 9.0 in hay, 1.0 in number.
Let's extract the data of center.
The data is 1.0 in hay, 4.0 in number.
Let's extract the data of sweat.
The data is 2.5 in hay, 4.5 in number.
The summed value of supply is 9.0+1.0=10.0. The summed value of center is 1.0+4.0=5.0. The summed value of sweat is 2.5+4.5=7.0. The smallest summed value is 5.0 in center. So the answer is center.

Q: How many groups of bars contain at least one bar with value smaller than 9?
A: Let's extract the data of oxygen.
The data is 4.0 in route, 1.0 in cash, 0.0 in king, 4.0 in taste.
Let's extract the data of lamp.
The data is 4.0 in route, 4.0 in cash, 9.0 in king, 4.0 in taste.
Let's extract the data of clock.
The data is 6.0 in route, 4.0 in cash, 5.0 in king, 3.0 in taste.
The groups of oxygen, lamp and clock all contain at least one bar with value smaller than 9. So the answer is 3.

Q: Which algorithm has the largest accuracy summed across all the datasets?
A: Let's extract the data of supply.
The data is 1.0 in datasets, 1.0 in stand, 1.0 in title, 1.0 in crowd.
Let's extract the data of horse.
The data is 2.0 in datasets, 5.0 in stand, 1.0 in title, 2.0 in crowd.
The summed value of supply is 1.0+1.0+1.0+1.0=4.0. The summed value of horse is 2.0+5.0+1.0+2.0=10.0. 10.0 in horse is the largest accuracy summed across all the datasets. So the answer is horse.

Q: Is the object device in the category hung preferred by more people than the object turn in the category pot?
A: Let's extract the data of hung BY device.
The data is 4.0.
Let's extract the data of pot BY turn.
The data is 8.0.
4.0 is less than 8.0. So the answer is no.

Q: {question}
A:
""".strip()

