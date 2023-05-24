# How to train and deploy a simple model using fastai.
In this blog post, I'll be showing you how to leverage the fastai library to train a simple deep learning model and deploy it as a web app using Gradio. This post is based on Lesson 2 of the fast.ai course - to watch the online lecture video that was recorded for this lesson at UQ in 2022, go [here](https://course.fast.ai/Lessons/lesson2.html). To check out the corresponding chapter in the *Practical Deep Learning for Coders with fastai & PyTorch*, go [here](https://github.com/fastai/fastbook/blob/master/02_production.ipynb). This was a really cool lesson in the fast.ai course where Jeremy walks you through the process of building your own image classification model, how to perform data augmentation and clean up your data after training your model to improve your model's accuracy, and how to deploy your model as an interactive web app. This lesson is super practical and fun - which is wasy I chose to document my experience playing around with my own model and to show you how simple the process can be using the fastai library!

## Step 0. Set-up your environment.
I built my image classifier model and app using the VSCode Jupyter extension in GitHub's CodeSpaces. However, you could you just create a notebook in Kaggle, Google Colab or Jupyter Notebook (via Anaconda).
Now, let's first install and import all the necessary libraries we'll need to train and deploy our deep learning model:

```python
!pip install -Uqq fastai
!pip install -Uqq fastbook
import gradio as gr
from fastbook import *
from fastai.vision.widgets import *
from fastai.vision.all import *
```

If you'd like to play around with multiple other deep learning architectures that don't already come with the fastai library, import timm as well:
```python
# If you want to try out other architectures:
!pip install -Uqq timm
!pip install git+https://github.com/rwightman/pytorch-image-models.git
import timm
```
If you are also using the VSCode Jupyter extension to run your Jupyter notebook, you may also be affected by a bug that will prevent your fine-tuning results to print. If that's the case, run this code snippet to fix that issue:
```python
# code to handle issue with VSCode Jupyter extension not printing epoch results:
from IPython.display import clear_output, DisplayHandle
def update_patch(self, obj):
    clear_output(wait=True)
    self.display(obj)
DisplayHandle.update = update_patch
```
Alright, in my experience trying to run Jeremy's "is it a bird" image classifier example from Lesson 1, the DuckDuckGo search API had a bug that prevented the notebook from running smoothly. Let's just run this snippet of code to make sure that our search images function is working:
```python
# Code to test if search_images_ddg is working:
from fastbook import *
urls = search_images_ddg('grizzly bear', max_images=100)
len(urls),urls[0]

download_url(urls[0], 'images/bear.jpg')
im = Image.open('images/bear.jpg')
im.thumbnail((256,256))
im
```
OK - if that's all working, then we can start building!

## Step 1. Create our testing dataset.
The first step is to download some images that you can load into your image classifier app later to test out your model. For my example, I'd like to build and train a model to classify different Iron Man suits. So I'll be downloading images of the four different Iron Man suits I'd like to classify.
```python
from fastdownload import download_url

class_1 = 'iron man mark 1'
save_class_1 = class_1 + '.jpg'
class_2 = 'iron man mark 2'
save_class_2 = class_2 + '.jpg'
class_3 = 'iron man mark 3'
save_class_3 = class_3 + '.jpg'
class_4 = 'iron man mark 4'
save_class_4 = class_4 + '.jpg'

download_url(search_images_ddg(class_1, max_images=1)[0], save_class_1, show_progress=False)
download_url(search_images_ddg(class_2, max_images=1)[0], save_class_2, show_progress=False)
download_url(search_images_ddg(class_3, max_images=1)[0], save_class_3, show_progress=False)
download_url(search_images_ddg(class_4, max_images=1)[0], save_class_4, show_progress=False)

# you may want to run the following lines in separate code cells in your notebook so that you can inspect each image individually:
Image.open(save_class_1).to_thumb(256,256)
Image.open(save_class_2).to_thumb(256,256)
Image.open(save_class_3).to_thumb(256,256)
Image.open(save_class_4).to_thumb(256,256)
```

## Step 2. Create our training and validation dataset.
Now, we are going to search for and download images for each of the classes that we want to train our model to classify. This block of code will also save the images in labelled folders in the given path directory based on their class name - which will give the images their corresponding labels.
```python
searches = 'iron man mark 1', 'iron man mark 2', 'iron man mark 3', 'iron man mark 4'
path = Path('iron_man_suit_classifier')
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images_ddg(f'{o}'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images_ddg(f'{o} photo'))
    sleep(10)
    # download_images(dest, urls=search_images_ddg(f'{o}'))
    # sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)
```
Before we start training our model, we'll want to remove any images that failed to download correctly.
```python
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)
```
Now, in order to load our dataset (the downloaded images) into our model, we'll first need to create a DataBlock. More information on what a DataBlock is can be found in Lesson 1's [Kaggle notebook](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data) and in fastai's [docs](https://docs.fast.ai/data.block.html).
```python
suits = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
)
dls = suits.dataloaders(path)
dls.train.show_batch(max_n=8)
```


## Code

General preformatted text:

    # Do a thing
    do_thing()

Python code and output:

```python
# Prints '2'
print(1+1)
```

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

