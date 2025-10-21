<h2>TensorFlow-FlexUNet-Image-Segmentation-Spontaneous-Intracerebral-Hemorrhage-CT (2025/10/21)</h2>

Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
<br>
This is the first experiment of Image Segmentation for <b>PHE-SICH-CT-IDS-Hemorrhage-CT (Spontaneous Intracerebral Hemorrhage)</b> based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
, and a 512x512 pixels PNG dataset <a href="https://drive.google.com/file/d/1jowXif0M2Y3aFHOptUaG4Lg4VVdjfo_d/view?usp=sharing">
Augmented-PHE-SICH-CT-IDS-Hemorrhage-ImageMask-Dataset.zip</a>, which was derived by us from 
<br><br>
<b>SubdatasetC_PNG</b> of  
<a href="https://www.kaggle.com/datasets/naumanalimurad/phe-sich-ct-ids"><b>PHE-SICH-CT-IDS: Hemorrhage CT Scan Dataset</b></a>
<br>
Please see also <a href="https://en.wikipedia.org/wiki/Intracerebral_hemorrhage">Intracerebral hemorrhage</a>.
<br>
<br>
As demonstrated in <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel">
TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel</a>,
 our Multiclass TensorFlowFlexUNet, which uses categorized masks, can also be applied to 
single-class image segmentation models. 
This is because it inherently treats the background as one category and your single-class mask data as 
a second category. In essence, your single-class segmentation model will operate with two categorized classes within our Multiclass UNet framework.
<br><br>
<b>Data Augmentation Strategy</b><br>
To address the limited size of PNG subset of <b>PHE-ICH-CT-IDS: Hemorrhage CT Scan Dataset</b> 
, we used our offline augmentation tool 
 <a href="https://github.com/sarah-antillia/Image-Distortion-Tool">
Image-Distortion-Tool</a> to augment the original subset.
<br>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks.<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/images/11210.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/masks/11210.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test_output/11210.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/images/11395.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/masks/11395.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test_output/11395.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/images/12791.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/masks/12791.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test_output/12791.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1 Dataset Citation</h3>
The dataset used here was obtained from 
<br><br>
<a href="https://www.kaggle.com/datasets/naumanalimurad/phe-sich-ct-ids"><b>PHE-SICH-CT-IDS: Hemorrhage CT Scan Dataset</b></a>
 on the kaggle web-site.
<br><br>
<b>About Dataset</b><br>
This publicly available dataset namely PHE-SICH-CT-IDS, which is constructed 120 CT scans of patients with 
spontaneous intracerebral hemorrhage. PHE-SICH-CT-IDS contains 3,511 CT images of SICH occurring in 
the basal ganglia region, with associated labels for the surrounding edematous zone around the hematoma. 
PHE-SICH-CT-IDS provides multiple functionalities including segmentation, detection, feature extraction, 
and more. It is divided into three sub-datasets representing different functionalities and formats: 
subdatasetA in NIfTI format, containing source CT data and edema zone labels, offering segmentation, 
feature extraction and prognosis prediction functionalities; subdatasetB and subdatasetC in JPG and PNG formats,
 containing sliced data with segmentation labels and detection annotations, providing segmentation and detection functionalities.

<br><br>
<b>LICENSE</b><br>
<a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>
<br>
<br>
<h3>
2 PHE-SICH-CT-IDS ImageMask Dataset
</h3>
If you would like to train this PHE-SICH-CT-IDS Segmentation model by yourself,
 please download our data <a href="https://drive.google.com/file/d/1jowXif0M2Y3aFHOptUaG4Lg4VVdjfo_d/view?usp=sharing">
Augmented-PHE-SICH-CT-IDS-Hemorrhage-ImageMask-Dataset.zip
 </a> on the google drive,
, expand the downloaded, and put it under dataset folder to be:
<pre>
./dataset
└─PHE-SICH-CT-IDS
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<b>PHE-SICH-CT-IDS Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/PHE-SICH-CT-IDS_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowUNet Model
</h3>
 We trained PHE-SICH-CT-IDS TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS and, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16</b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3

num_classes    = 2

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.03
dilation       = (1,1)

</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
You may specify other loss and metrics function names.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for PHE-SICH-CT-IDS 1+1 classes.
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"

; PHE-SICH-CT-IDS
; rgb color map dict for 2 classes.
;                 Hemorrhage:white
rgb_map = {(0,0,0):0,(255,255,255):1,}

</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 29,30,31)</b><br>
<img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 60,61,62)</b><br>
<img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was stopped at epoch 62 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/asset/train_console_output_at_epoch62.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/eval/train_losses.png" width="520" height="auto"><br>
<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for PHE-SICH-CT-IDS.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/asset/evaluate_console_output_at_epoch62.png" width="720" height="auto">
<br><br>Image-Segmentation-PHE-SICH-CT-IDS

<a href="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/evaluation.csv">evaluation.csv</a><br>

The loss (categorical_crossentropy) to this PHE-SICH-CT-IDS/test was very low, and dice_coef_multiclass very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0027
dice_coef_multiclass,0.9986
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowUNet model for PHE-SICH-CT-IDS.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
 
<img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Images of 512x512 pixels </b><br>
<table>
<tr>

<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/images/11395.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/masks/11395.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test_output/11395.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/images/11551.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/masks/11551.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test_output/11551.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/images/12791.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/masks/12791.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test_output/12791.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/images/12874.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/masks/12874.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test_output/12874.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/images/12975.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/masks/12975.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test_output/12975.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/images/13464.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test/masks/13464.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PHE-SICH-CT-IDS/mini_test_output/13464.png" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Artificial Intelligence in Healthcare Competition (TEKNOFEST-2021):Stroke Data Set</b><br>
Ural Koç,, Ebru Akçapınar Sezer, Yaşar Alper Özkaya, Yasin Yarbay4 , Onur Taydaş,<br>
Veysel Atilla Ayyıldız , Hüseyin Alper Kızıloğlu, Uğur Kesimal, İmran Çankaya, Muhammed Said Beşler,<br>
Emrah Karakaş, Fatih Karademir, Nihat Barış Sebik, Murat Bahadır , Özgür Sezer,<br>
Batuhan Yeşilyurt, Songul Varlı, Erhan Akdoğan, Mustafa Mahir Ülgü , Şuayip Birinci<br>

<a href="https://www.eajm.org/en/artificial-intelligence-in-healthcare-competition-teknofest-2021-stroke-data-set-1618971">
https://www.eajm.org/en/artificial-intelligence-in-healthcare-competition-teknofest-2021-stroke-data-set-1618971
</a>

<br><br>
<b>2. Hemorrhagic stroke lesion segmentation using a 3D U-Net with squeeze-and-excitation blocks</b><br>
Valeriia Abramova, Albert Clèrigues, Ana Quiles, Deysi Garcia Figueredo, Yolanda Silva, Salvador Pedraza,<br>
 Arnau Oliver, Xavier Lladó<br>
<a href="https://www.sciencedirect.com/science/article/pii/S0895611121000574">
https://www.sciencedirect.com/science/article/pii/S0895611121000574
</a>
<br>
<br>
<b>3. Segmentation of acute stroke infarct core using image-level labels on CT-angiography</b><br>
Luca Giancardo, Arash Niktabe, Laura Ocasio, Rania Abdelkhaleq, Sergio Salazar-Marioni, Sunil A Sheth<br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10011814/">
https://pmc.ncbi.nlm.nih.gov/articles/PMC10011814/
</a>
<br>
<br>
<b>4.Segmenting Small Stroke Lesions with Novel Labeling Strategies</b><br>
Liang Shang, Zhengyang Lou, Andrew L. Alexander, Vivek Prabhakaran,<br>
William A. Sethares, Veena A. Nair, and Nagesh Adluru<br>
<a href="https://arxiv.org/pdf/2408.02929">
https://arxiv.org/pdf/2408.02929
</a>
<br>
<br>
<b>5. Spontaneous Intracerebral Hemorrhage</b><br>
Kevin N. Sheth, M.D.<br>
<a href="https://www.nejm.org/doi/pdf/10.1056/NEJMra2201449">
https://www.nejm.org/doi/pdf/10.1056/NEJMra2201449
</a>
<br>
<br>
<b>6. Semantic Segmentation of Spontaneous Intracerebral Hemorrhage, <br>
Intraventricular Hemorrhage, and Associated Edema on CT Images Using Deep Learning</b><br>
Yong En Kok, Stefan Pszczolkowski, Zhe Kang Law, Azlinawati Ali, Kailash Krishnan, <br>
Philip M Bath, Nikola Sprigg, Robert A Dineen, Andrew P French<br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC9745441/">
https://pmc.ncbi.nlm.nih.gov/articles/PMC9745441/
</a>
<br>
<br>
<b>7. TensorFlow-FlexUNet-Image-Segmentation-Real-Time-Brain-Stroke-CT</b><br>
Toshiyuki Arai antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Real-Time-Brain-Stroke-CT">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Real-Time-Brain-Stroke-CT
</a>
<br>
<br>
<b>8. TensorFlow-FlexUNet-Image-Segmentation-TEKNOFEST-2021-Stroke-CT</b><br>
Toshiyuki Arai antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-TEKNOFEST-2021-Stroke-CT">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-TEKNOFEST-2021-Stroke-CT
</a>
<br>
<br>

