# CSE 573: Computer Vision and Image Processing
<p align="center">
<img src="Project -1/data/ub.png" alt="ub_logo.jpg" width="100" height="100"> <br>
  <b> Course offered by Professor David Doermann in Spring 2020 </b>
</p>

### [Edge Detection](Project-1) :
<img src="Project -1/data/bar.jpg" alt="bar.jpg" width="1100" height="3"> <br>

**Problem:** 
`Edge Detection` :The goal of this task is to experiment with two commonly used edge detection operator, i.e.,Prewitt operator and Sobel operator.Specifically, the task is to detect edges in a given image. Do not use any API provided by opencv and numpy in the code except(“np.sqrt()”, “np.zeros()”, “np.ones()”, “np.multiply()”, “np.divide()”, “cv2.imread()”,“cv2.imshow()”, “cv2.imwrite()”, and “cv2.resize()”)

**Approach:**
- The project applied `Sobel` and `Prewitt` filters to detect edges in a given image
- Implemented common image processing tasks : 
  - Image Padding
  - Applied Convolution and Correlation
  - All Images were normalized before implementing convolution and correlation. 
  
**Sample input and output:** 

Results: <br>

Edge detection using `Prewitt` filter: 

<img src="Project -1/results/result 1.png" alt="result 1.png">

Edge detection using `Sobel` filter: 

<img src="Project -1/results/result 2.png" alt="result 2.png">

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

### [Character Detection](Project-1) :
<img src="Project -1/data/bar.jpg" alt="bar.jpg" width="1100" height="3"> <br>

**Problem:** 
`Character Detection`: The goal of this task is to experiment with template matching algorithms. Specifically, the task is to find a specific character (or set of characters) in a given image. 

**Approach:**
- The project applied **Template matching algorithm** to detect a specific character (ex. a/b/c) in a given image
- Created a templete each character "a", "b" and "c".
- Implemented `NCC (Normalized Cross Correlation)` for matching the template with the given image.


### [Panorama/Image Stitching](Project-2) :
<img src="Project -1/data/bar.jpg" alt="bar.jpg" width="1100" height="3"> <br>

**Problem:** 
`Image Stitching`: Create a panoramic image from at most 5 images. The goal of this task is to experiment with image stitching methods. Given a set of photos, your
program should be able to stitch them into a panoramic photo. Overlap of the given images will be at least 20%. Any API provided by OpenCV could be used, except “`cv2.findHomography()`” and APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “`cv2.BFMatcher()`” and “`cv2.Stitcher.create()`”.

**Approach:**
- Keypoints detection and 128 bit feature vector computation using `SIFT` descriptor. 
- Created an algorithm that can define the order of the images if given in randomized order.
- Homography matrix generation using `SVD` technique.
- Implemented `RANSAC` algorithm for finding the best Homography matrix
- Stitched all images


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

**Results:** 

Output image: <br>
<img src="Project -2/data/panorama.jpg" alt="panoroma.jpg">
<img src="Project -2/extra1/panorama.jpg" alt="panoroma.jpg">
<img src="Project -2/extra3/panorama.jpg" alt="panoroma.jpg">



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

### [Face Detection in the Wild](Project-3) :
<img src="Project -1/data/bar.jpg" alt="bar.jpg" width="1100" height="3"> <br>

**Problem:** 
Implement the `Viola-Jones` face detection algorithm which is capable of detecting frontal faces in real time and is regarded as a milestone in the development of computer vision. Any APIs provided by OpenCV that have “cascade”, “Cascade”, “haar” or “Haar” functionality can not be used. Using any APIs that implement part of Viola-Jones algorithm directly, e.g., an API that computes integral image, will result in a deduction of 10% − 100% of the maximum possible points of this project

**Approach:**
- Used `FDDB` dataset to train the model with 'face images and `CBCL` dataset to train with 'non-face images'
- Implemented `integral image` calculation
- `Adaboost` implementation
- Developed `CASCADING` to reject non-face region quickly


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Project Report](Project_03/Report.pdf)


**output:** <br>

<img src="Project_03/sample_output/1.png" width="300" height="250"><img src="Project_03/sample_output/2.png" width="300" height="250"><img src="Project_03/sample_output/3.png" width="250" height="250">


---
## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;