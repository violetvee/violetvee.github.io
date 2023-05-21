# What is Deep Learning?

Deep learning is a subset of machine learning where artificial neural networks are trained to learn and make decisions or predictions about a given dataset. The neural networks consist of three or more layers of interconnected artificial neurons - known as hidden layers. The more hidden layers the network has, the deeper the network, and the "deeper" the learning. (IBM, n.d.) This multi-layered structure mimics the learning process of the brain.

![](/images/artificial_neural_network.jpeg "The multi-layered structure of artificial neural networks used in deep learning (Levity AI, n.d.)"

Deep learning differs from classical machine learning in the type of data that it works with and the methods in which it learns. In traditional machine learning algorithms, the feature extraction is often required to be performed by a domain expert. Whereas deep learning algorithms are able to learn these high-level features from data in an incremental manner and thus perform feature extraction automatically. (Nagdeve, 2020)
As such, deep learning algorithms require less human intervention than classical machine learning algorithms. However, this also means that deep learning requires a lot more data than traditional machine learning algorithms to perform properly.

References:
IBM. (n.d.). Deep learning. Retrieved from https://www.ibm.com/topics/deep-learning
Microsoft. (n.d.). Deep Learning vs. Machine Learning - Azure Machine Learning. Retrieved from https://learn.microsoft.com/en-us/azure/machine-learning/concept-deep-learning-vs-machine-learning?view=azureml-api-2
Nagdeve, M. (2020, June 15). Why Deep Learning is Needed Over Traditional Machine Learning. Towards Data Science. Retrieved from https://towardsdatascience.com/why-deep-learning-is-needed-over-traditional-machine-learning-1b6a99177063
Levity AI. (n.d.). The Difference Between Machine Learning and Deep Learning. Retrieved from https://levity.ai/blog/difference-machine-learning-deep-learning

Here's the table of contents:

1. TOC
{:toc}

## Basic setup

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-filename.md`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `filename` is whatever file name you choose, to remind yourself what this post is about. `.md` is the file extension for markdown files.

The first line of the file should start with a single hash character, then a space, then your title. This is how you create a "*level 1 heading*" in markdown. Then you can create level 2, 3, etc headings as you wish but repeating the hash character, such as you see in the line `## File names` above.

## Basic formatting

You can use *italics*, **bold**, `code font text`, and create [links](https://www.markdownguide.org/cheat-sheet/). Here's a footnote [^1]. Here's a horizontal rule:

---

## Lists

Here's a list:

- item 1
- item 2

And a numbered list:

1. item 1
1. item 2

## Boxes and stuff

> This is a quotation

{% include alert.html text="You can include alert boxes" %}

...and...

{% include info.html text="You can include info boxes" %}

## Images

![](/images/logo.png "fast.ai's logo")

## Code

General preformatted text:

    # Do a thing
    do_thing()

Python code and output:

```python
# Prints '2'
print(1+1)
```

    2

## Tables

| Column 1 | Column 2 |
|-|-|
| A thing | Another thing |

## Footnotes

[^1]: This is the footnote.

