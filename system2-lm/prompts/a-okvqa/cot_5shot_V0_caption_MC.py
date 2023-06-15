PROMPT = """
Q: Why is the person reaching out to the giraffe?
Answer Choices:
(a) to bathe
(b) to feed
(c) to pet
(d) to comb
A: Let's find out what is shown in the image.
Person, giraffe.
Let's find out what is the person holding.
Food.
Let's find out is the person giving the food to the giraffe.
Yes.
People give animals food to feed them. So the answer is (b) to feed.

Q: What are the two people engaging in?
Answer Choices:
(a) fishing
(b) paddling
(c) swimming
(d) surfing
A: Let's find out what is shown in the image.
People, sea.
Let's find out where are the people.
Above sea.
Let's find out what are the people holding.
Paddles.
People hold paddles on the sea when paddling. So the answer is (b) paddling.

Q: What is the name of a band with this number of members?
Answer Choices:
(a) quartet
(b) cinqtet
(c) duet
(d) sextet
A: Let's find out what is shown in the image.
People playing music.
Let's find out how many people are performing on the stage.
4.
A band with 4 people playing music is called a quartet. So the answer is (a) quartet.

Q: How many items qualify as a berry botanically?
Answer Choices:
(a) one
(b) three
(c) two 
(d) four
A: Let's find out what is shown in the image.
Food on the table.
Let's find out what food items are there.
Salad, bread.
Let's find out what berries are on the bread.
Blueberries, raspberries.
There are two types of berries. So the answer is (c) two.

Q: What does the number 84 represent?
Answer Choices:
(a) table number
(b) store number
(c) order number
(d) price
A: Let's find out what is shown in the image.
A table in a restaurant.
Let's find out is the number 84 on the table.
No.
It's a restaurant and the number is not on the table. It's likely that 84 is the number of the order. So the answer is (c) order number.

Q: {question}
Answer Choices:
{choices}
A:
""".strip()

