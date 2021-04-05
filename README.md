# Film Review Naïve Bayes Classifier
This is a Naïve Bayes classifier that categorizes film reviews as being either a positive review or a negative one.

A Naïve Bayes classifier is a probabilistic classifier based on [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
with strong (naïve) independence assumptions between the features.
## Training/Testing Data
The file, `alldata.txt`, contains about 13,000 film reviews, each on its own line. Each line is of the following format:
```
NUMBER OF STARS|ID|TEXT
```
- The number of stars is 1 or 5. 
- The text goes until a newline (`\n`).

I did a 90/10 split of the data, using 90% of the data for training my classifier and the other 10% for testing it.

## Try It Out
Clone the repository to your local machine. Make sure [Python](https://www.python.org/) is installed. In the local repository, run the following command:
```
pip install -r requirements.txt
```
Run the main.py file to run the provided tests. Edit the `alldata.txt` file and add your own film reviews (in the format specified above)
to see how my accurately my classifier with categorize them.
