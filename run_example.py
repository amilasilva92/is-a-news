import argparse
import time
from url_classifier import URL_Classifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict news URLs')
    parser.add_argument('--url', type=str, help='url', required=True)
    parser.add_argument('--skip_crawling', type=bool, help='to skip the classfier-based prediciton step', default=False)
    args = parser.parse_args()


    classifier = URL_Classifier()
    start_time = time.time()
    result = classifier.classify_url(args.url, skip_crawling = args.skip_crawling)
    time_taken = time.time() - start_time
    print('Is %s a news URL? %s' % (args.url, str(result)))
    print('Time taken: %f (in seconds)' % time_taken )
