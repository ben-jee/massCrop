Specify python3 on launch

Padding paints an area around the region of interest and enlarges the ROI based on the pixel value. 
460 default padding.

Threads isn't actually threading yet, but increasing the value does improve processing time.
Average image resolution for testing was 4721.4 x 3177, and done on dataset of 80 different images

1 Thread:
5:20s, 100% accuracy, 99.99%+ confidence

4 Threads:
3:40s, 100% acc, 99.99%+ conf

8 Threads:
5:40s, 100% acc, 99.98%+ conf

Default thread value is 4