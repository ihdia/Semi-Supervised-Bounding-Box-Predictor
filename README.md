# Semi-Supervised-Bounding-Box-Predictor
An intelligent tool to predict fitting polygons given the bounding box around the region.

This system takes in an input image and Bounding Box coordinates around a region for annotation in document image and outputs a tight fitting bounding polygon around the region.

### Install prerequisites 
Install requirements stated in requirements.txt

### To run Inference on your own image

1. Download the model from this [link](https://drive.google.com/drive/folders/10yGtlXGTOFPuF4Wk_gurBhOmyUMqQz2j?usp=sharing) 
2. Place the `Final.pth` file in the folder (`Semi-Supervised-Bounding-Box-Predictor/ssbbox/`)
3. Run the script(`Semi-Supervised-Bounding-Box-Predictor/ssbbox/toolscrpt.py`) as ->
```bash
python3 toolscrpt.py --img path/to/your/image --bbox x,y,w,h
```
4. This outputs the final result in terms of X and Y (in new lines) coordinates respectively.
