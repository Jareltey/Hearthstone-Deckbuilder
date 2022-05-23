# Hearthstone Deckbuilder

This project aims to recreate the HearthArena card ranking algorithm using deep learning models built with Tensorflow. In arena mode of the card game Hearthstone, players draft a deck of 30 cards sequentially where they get a choice between 3 picks each time. Cards offered vary in base quality and fit in the deck being drafted (i.e. some cards work well in certain kinds of decks or synergize with other cards), making the drafting process critical to doing well in the arena games. The algorithm assigns scores to the cards offered by learning from historical data about card picks and winrates, allowing players to make informed decisions about which card to pick each time.

## Algorithm design

The algorithm is designed as a pipeline with 5 main stages:

1) Webscraping to obtain historical game data using BeautifulSoup
2) Using Hearthstone's API to obtain card data
3) Data preprocessing using Numpy library in Python
4) Constructing and training DNN regression and Linear regression models using Tensorflow
5) Performing prediction (assigning card scores) on test decks
