PROMPT = """
Q: Why is the person reaching out to the giraffe?
A: Let's find out what is shown in the image.
Person, giraffe.
Let's find out what is the person holding.
Food.
Let's find out is the person giving the food to the giraffe.
Yes.
People give animals food to feed them. So the answer is feeding.

Q: What are the two people engaging in?
A: Let's find out what is shown in the image.
People, sea.
Let's find out where are the people.
Above sea.
Let's find out what are the people holding.
Paddles.
People hold paddles on the sea when paddle boarding. So the answer is paddle boarding.

Q: What is the name of a band with this number of members?
A: Let's find out what is shown in the image.
People playing music.
Let's find out how many people are performing on the stage.
4.
A band with 4 people playing music is called a quartet. So the answer is quartet.

Q: How many items qualify as a berry botanically?
A: Let's find out what is shown in the image.
Food on the table.
Let's find out what food items are there.
Salad, bread.
Let's find out what berries are on the bread.
Blueberries, raspberries.
There are two types of berries. So the answer is two.

Q: What does the number 84 represent?
A: Let's find out what is shown in the image.
A table in a restaurant.
Let's find out is the number 84 on the table.
No.
It's a restaurant and the number is not on the table. It's likely that 84 is the number of the order. So the answer is order number.

Q: {question}
A:
""".strip()

