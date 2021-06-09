# is-a-news
This is a web application that automatically discover news URLs based on their content and a predefined URL database. The application is running at https://is-a-news.herokuapp.com/

##### Instructions to Run as an Standalone Application
1. To install required libraries (Note: The code is written in python 3).
```shell=
pip install -r requirements.txt
```

2. To run the whole pipeline:
```shell=
python run_example.py --url [URL]
```
e.g., python run_example.py --url https://www.bbc.com/news/world-us-canada-57408094


##### Optional Arguments
Here we describe the optional arguments of the model:

[--skip_crawling]
What is it: Boolean value to inform the pipeline to skip the content-based label prediction step, which is the bottleneck of this pipeline. If you skip this step (by setting True for this argument), it could substantially reduce the processing time by relying just on the existing databases to make predictions.
Default: False
