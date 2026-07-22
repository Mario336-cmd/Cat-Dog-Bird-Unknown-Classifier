# Cat, Dog, Bird and Unknown Image Classifier

This project is a standalone open-set image classification system.

The classifier is designed to classify images into four classes:

* Cat
* Dog
* Bird
* Unknown

The Unknown class allows the system to handle images that do not belong to the main target classes. This makes the project more realistic because users can upload images outside the Cat, Dog, and Bird categories.

The main model used in this project is EfficientNetV2B0. The model is prepared for browser deployment using TensorFlow.js, allowing predictions to run directly in the web app.

## Project Features

This project includes:

* Cat, Dog, Bird, and Unknown classification
* Advanced prediction details
* Class confidence scores
* Browser-side prediction using TensorFlow.js
* Live Grad-CAM explainability through a Model Focus Heatmap
* Browser performance metrics such as model load time, prediction time, TensorFlow.js backend, backend fallback support, input size, and model size
* Consent-based feedback ZIP export for dataset improvement


## Table of Contents

1. [Dataset Collection and Automatic Cleaning](#notebook-1)
2. [Manual Review and Dataset Splitting](#notebook-2)
3. [Model Training and Fine-Tuning](#notebook-3)
4. [Final Model Evaluation](#notebook-4)
5. [TensorFlow.js Export and Browser Verification](#notebook-5)
6. [Future Work and Limitations](#future-work-and-limitations)


# Notebooks

<a id="notebook-1"></a>

## `Model Building/Jupyter Notebooks/01_dataset_collection_and_auto_cleaning.ipynb`

This notebook is part of the model-building workflow for the project.

It handles the dataset collection and automatic-cleaning stage for the Cat, Dog, Bird, and Unknown image classifier. The goal of this notebook is to build automatically cleaned image pools before manual review and the final Train, Validation, and Test split are completed in the next notebook.

This notebook prepares automatically cleaned image pools for:

* Cat Train/Validation images
* Cat Test images
* Dog Train/Validation images
* Dog Test images
* Bird Train/Validation images
* Bird Test images
* Unknown Train/Validation images
* Unknown Test images

### Required Directories

The following structure must exist before the downloading and automatic-cleaning process starts:

```text
Image Classifier Project/
└── Model Building/
    └── Jupyter Notebooks/
        └── 01_dataset_collection_and_auto_cleaning.ipynb
```

The notebook finds the `Image Classifier Project` root and creates the required `Model Building/Datasets` directories automatically.

The `Datasets`, `Original`, `Auto Cleaned`, `Manually Cleaned`, `Rejected`, `Final Dataset`, class, source, and rejection-reason directories do not need to be created manually before the notebook is run for the first time.

### Dataset Download Requirement

The datasets can be downloaded automatically by running the dataset collection and cleaning notebook. Where automatic downloading is supported, the notebook downloads the required archives and metadata, extracts the supported archives, locates the source images, automatically cleans them, and saves the accepted images into the correct `Model Building/Datasets/Auto Cleaned` folders.

The notebook detects previously downloaded or extracted sources and avoids repeating completed work.

Datasets can also be downloaded manually. If a Train/Validation dataset is downloaded manually, place the downloaded archive or extracted dataset folder inside:

`Model Building/Datasets/Original/Train and Validation`

The Open Images V7 Test images and metadata are stored inside:

`Model Building/Datasets/Original/Test/Open Images V7`

Microsoft Cats and Dogs:
https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip

Oxford-IIIT Pet:
https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz

AFHQ-v2:
https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=1

CAT Dataset:
https://archive.org/download/CAT_DATASET/CAT_DATASET_01.zip
https://archive.org/download/CAT_DATASET/CAT_DATASET_02.zip

Stanford Dogs:
http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar

CUB-200-2011:
https://www.vision.caltech.edu/datasets/cub_200_2011/
https://data.caltech.edu/records/65de6-vp158
https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1

NABirds:
https://dl.allaboutbirds.org/nabirds

Birdsnap:
http://thomasberg.org/datasets/birdsnap/1.1/

iNaturalist non-target taxa:
https://api.inaturalist.org/v1/taxa/autocomplete
https://api.inaturalist.org/v1/observations
https://www.inaturalist.org/

NINCO:
https://zenodo.org/record/8013288/files/NINCO_all.tar.gz?download=1

ImageNet-O:
https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar

COCO 2017:
http://images.cocodataset.org/zips/train2017.zip
http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Places365:
http://data.csail.mit.edu/places/places365/val_256.tar

Food-101:
http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz

Open Images V7:
https://storage.googleapis.com/openimages/web/index.html

### Dataset Sources

The Train/Validation pools use the following sources:

| Class   | Dataset sources                                                                                                               |
| ------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Cat     | Microsoft Cats and Dogs, Oxford-IIIT Pet, AFHQ-v2 Cat, CAT Dataset                                                            |
| Dog     | Microsoft Cats and Dogs, Oxford-IIIT Pet, AFHQ-v2 Dog, Stanford Dogs                                                          |
| Bird    | CUB-200-2011, NABirds, Birdsnap                                                                                               |
| Unknown | iNaturalist non-target taxa, AFHQ-v2 Wild, NINCO, ImageNet-O, COCO 2017, Places365, Food-101, and Derived Unknown Corruptions |

The Unknown pool is intentionally broader than the known classes. It includes non-target animals, humans, objects, vehicles, food, household content, scenes, nature, and realistic low-quality images.

The official Test candidates are collected separately from Open Images V7. Open Images is not used in the rebuilt Train/Validation pools.

For Cat, Dog, and Bird, Open Images candidates are processed in the following order until the required number of automatically cleaned images is reached:

1. Validation split
2. Test split
3. Train split when additional images are required

The Unknown Test candidates are collected from the Open Images V7 Validation split using positive non-target labels. Images that also have a positive Cat, Dog, or Bird label are excluded from the Unknown candidate pool.

### Dataset Cleaning

After the source images are available in the `Model Building/Datasets/Original` directory, the cleaning code processes each dataset and saves accepted images into the correct `Model Building/Datasets/Auto Cleaned` folders.

The cleaning process includes:

* Reading images from the original source folders
* Creating automatically cleaned Train/Validation pools
* Creating automatically cleaned Test pools
* Removing broken or unreadable images
* Removing images with invalid dimensions
* Removing images where either dimension is below 64 pixels
* Removing completely uniform blank images
* Removing exact duplicate images
* Rejecting images that fail during hashing, processing, or copying
* Checking for exact duplicates across sources within the same class
* Checking Unknown images against accepted Cat, Dog, and Bird images
* Checking Test images against the complete Train/Validation pool
* Checking each new Test class against Test classes already collected
* Excluding Unknown Test candidates with positive Cat, Dog, or Bird labels
* Creating Derived Unknown Corruptions from accepted Unknown base images
* Keeping Unknown images separate from Cat, Dog, and Bird images
* Saving rejected images into the `Model Building/Datasets/Rejected/Automatic` directory by pool, class, source, and rejection reason
* Printing final automatic-cleaning metrics in the notebook output cells

Exact duplicate detection corrects EXIF orientation, temporarily converts the image to RGB, and hashes the decoded image dimensions and pixel values using SHA-256.

Completely blank images are rejected only when every pixel has the same value. Dark, bright, blurred, noisy, compressed, or low-contrast photographs are not rejected unless they are genuinely uniform or fail another automatic-cleaning rule.

Accepted images are copied into `Model Building/Datasets/Auto Cleaned` without resizing, recompressing, renaming, or converting them. The original filename, extension, and relative source folder structure are preserved.


Rejected images are not deleted immediately. They are copied into `Model Building/Datasets/Rejected/Automatic` so they can be inspected later if needed.

### Dataset Cleaning Results

The dataset collection and cleaning notebook created automatically cleaned Train/Validation pools and separate automatically cleaned Test candidate pools for Cat, Dog, Bird, and Unknown.

The notebook stops once the target number of automatically cleaned images is reached for each source or class. This means some candidate images were not processed because enough acceptable images had already been collected.

### Automatically Cleaned Dataset Size

| Class     | Train/Validation Pool | Test Candidate Pool | Total Automatically Cleaned Images |
| --------- | --------------------: | ------------------: | ---------------------------------: |
| Cat       |                30,339 |               3,000 |                             33,339 |
| Dog       |                31,000 |               3,000 |                             34,000 |
| Bird      |                31,000 |               3,000 |                             34,000 |
| Unknown   |                55,000 |               3,000 |                             58,000 |
| **Total** |           **147,339** |          **12,000** |                        **159,339** |

These are automatically cleaned candidate pools and are not the final dataset counts.

The extra images provide room for unsuitable images to be removed during manual review while retaining the planned final dataset:

| Class     | Planned Train | Planned Validation | Planned Test | Planned Total |
| --------- | ------------: | -----------------: | -----------: | ------------: |
| Cat       |        27,000 |              3,000 |        2,500 |        32,500 |
| Dog       |        27,000 |              3,000 |        2,500 |        32,500 |
| Bird      |        27,000 |              3,000 |        2,500 |        32,500 |
| Unknown   |        45,000 |              5,000 |        2,500 |        52,500 |
| **Total** |   **126,000** |         **14,000** |   **10,000** |   **150,000** |

### Train/Validation Automatic Cleaning Results

| Class     | Available Candidate Images | Processed Images | Automatically Cleaned Images | Automatically Rejected Images |
| --------- | -------------------------: | ---------------: | ---------------------------: | ----------------------------: |
| Cat       |                     30,455 |           30,455 |                       30,339 |                           116 |
| Dog       |                     43,239 |           31,081 |                       31,000 |                            81 |
| Bird      |                     72,823 |           31,014 |                       31,000 |                            14 |
| Unknown   |                    284,861 |           55,571 |                       55,000 |                           571 |
| **Total** |                **431,378** |      **148,121** |                  **147,339** |                       **782** |

Dog, Bird, and Unknown processing stopped after their automatically cleaned image targets were reached. Their available candidate totals are therefore larger than the number of images that were processed.

### Cat Dataset Train/Validation Results

| Dataset Source          | Candidate Images | Processed Images | Automatically Cleaned Images | Automatically Rejected Images | Rejection Reasons                                                |
| ----------------------- | ---------------: | ---------------: | ---------------------------: | ----------------------------: | ---------------------------------------------------------------- |
| Microsoft Cats and Dogs |           12,500 |           12,500 |                       12,465 |                            35 | broken_or_unreadable: 1, exact_duplicate: 17, low_resolution: 17 |
| Oxford-IIIT Pet         |            2,400 |            2,400 |                        2,380 |                            20 | exact_duplicate: 20                                              |
| AFHQ-v2 Cat             |            5,558 |            5,558 |                        5,558 |                             0 | none                                                             |
| CAT Dataset             |            9,997 |            9,997 |                        9,936 |                            61 | exact_duplicate: 61                                              |
| **Total**               |       **30,455** |       **30,455** |                   **30,339** |                       **116** | —                                                                |

### Cat Dataset Test Results

| Dataset Source | Open Images Split | Processed Images | Automatically Cleaned Images | Automatically Rejected Images | Rejection Reasons |
| -------------- | ----------------- | ---------------: | ---------------------------: | ----------------------------: | ----------------- |
| Open Images V7 | Validation        |              350 |                          350 |                             0 | none              |
| Open Images V7 | Test              |            1,012 |                        1,012 |                             0 | none              |
| Open Images V7 | Train             |            1,638 |                        1,638 |                             0 | none              |
| **Total**      | —                 |        **3,000** |                    **3,000** |                         **0** | —                 |

### Dog Dataset Train/Validation Results

| Dataset Source          | Available Candidate Images | Processed Images | Automatically Cleaned Images | Automatically Rejected Images | Rejection Reasons                                                |
| ----------------------- | -------------------------: | ---------------: | ---------------------------: | ----------------------------: | ---------------------------------------------------------------- |
| Microsoft Cats and Dogs |                     12,500 |           12,500 |                       12,462 |                            38 | broken_or_unreadable: 1, exact_duplicate: 11, low_resolution: 26 |
| Oxford-IIIT Pet         |                      4,990 |            4,990 |                        4,980 |                            10 | exact_duplicate: 10                                              |
| AFHQ-v2 Dog             |                      5,169 |            5,169 |                        5,169 |                             0 | none                                                             |
| Stanford Dogs           |                     20,580 |            8,422 |                        8,389 |                            33 | exact_duplicate: 33                                              |
| **Total**               |                 **43,239** |       **31,081** |                   **31,000** |                        **81** | —                                                                |

The 31,000-image Dog automatic-cleaning target was reached before every Stanford Dogs candidate needed to be processed.

### Dog Dataset Test Results

| Dataset Source | Open Images Split | Processed Images | Automatically Cleaned Images | Automatically Rejected Images | Rejection Reasons           |
| -------------- | ----------------- | ---------------: | ---------------------------: | ----------------------------: | --------------------------- |
| Open Images V7 | Validation        |            1,593 |                        1,590 |                             3 | exact_duplicate_cat_test: 3 |
| Open Images V7 | Test              |            1,411 |                        1,410 |                             1 | exact_duplicate_cat_test: 1 |
| **Total**      | —                 |        **3,004** |                    **3,000** |                         **4** | —                           |

All four automatically rejected Dog Test candidates were exact duplicates of images already accepted into the Cat Test candidate pool.

### Bird Dataset Train/Validation Results

| Dataset Source | Available Candidate Images | Processed Images | Automatically Cleaned Images | Automatically Rejected Images | Rejection Reasons   |
| -------------- | -------------------------: | ---------------: | ---------------------------: | ----------------------------: | ------------------- |
| CUB-200-2011   |                     11,788 |           11,788 |                       11,787 |                             1 | exact_duplicate: 1  |
| NABirds        |                     48,562 |            9,606 |                        9,606 |                             0 | none                |
| Birdsnap       |                     12,473 |            9,620 |                        9,607 |                            13 | exact_duplicate: 13 |
| **Total**      |                 **72,823** |       **31,014** |                   **31,000** |                        **14** | —                   |

CUB-200-2011 was retained almost completely. The remaining Bird target was divided approximately evenly between NABirds and Birdsnap to improve source diversity and prevent one large source from dominating the class.

### Bird Dataset Test Results

| Dataset Source | Open Images Split | Processed Images | Automatically Cleaned Images | Automatically Rejected Images | Rejection Reasons           |
| -------------- | ----------------- | ---------------: | ---------------------------: | ----------------------------: | --------------------------- |
| Open Images V7 | Validation        |              698 |                          697 |                             1 | exact_duplicate_cat_test: 1 |
| Open Images V7 | Test              |            1,895 |                        1,892 |                             3 | exact_duplicate_dog_test: 3 |
| Open Images V7 | Train             |              411 |                          411 |                             0 | none                        |
| **Total**      | —                 |        **3,004** |                    **3,000** |                         **4** | —                           |

One automatically rejected Bird Test candidate was an exact duplicate of an accepted Cat Test image. Three were exact duplicates of accepted Dog Test images.

### Unknown Dataset Train/Validation Results

| Dataset Source                | Candidate Images | Processed Images | Automatically Cleaned Images | Target Images | Automatically Rejected Images | Rejection Reasons                                       |
| ----------------------------- | ---------------: | ---------------: | ---------------------------: | ------------: | ----------------------------: | ------------------------------------------------------- |
| iNaturalist Non-Target Taxa   |           12,500 |           10,359 |                       10,350 |        10,350 |                             9 | exact_duplicate: 9                                      |
| AFHQ-v2 Wild                  |            5,076 |            4,501 |                        4,500 |         4,500 |                             1 | exact_duplicate: 1                                      |
| NINCO                         |           15,393 |            4,308 |                        3,850 |         3,850 |                           458 | blank_image: 450, exact_duplicate: 7, low_resolution: 1 |
| ImageNet-O                    |            2,000 |            1,742 |                        1,650 |         1,650 |                            92 | exact_duplicate: 88, low_resolution: 4                  |
| COCO Humans                   |           60,669 |            8,801 |                        8,800 |         8,800 |                             1 | low_resolution: 1                                       |
| COCO Other Filtered Negatives |           46,223 |            4,400 |                        4,400 |         4,400 |                             0 | none                                                    |
| Places365                     |           36,500 |            7,700 |                        7,700 |         7,700 |                             0 | none                                                    |
| Food-101                      |          101,000 |            8,260 |                        8,250 |         8,250 |                            10 | exact_duplicate: 10                                     |
| Derived Unknown Corruptions   |            5,500 |            5,500 |                        5,500 |         5,500 |                             0 | none                                                    |
| **Total**                     |      **284,861** |       **55,571** |                   **55,000** |    **55,000** |                       **571** | —                                                       |

The 450 NINCO files rejected as `blank_image` were completely uniform files rather than merely dark, bright, blurred, noisy, or low-contrast photographs. They were excluded because they contained no meaningful visual information.

The Unknown pool includes non-target animals such as fish, frogs, reptiles, insects, bats, rodents, livestock, primates, marine animals, and other wildlife. It also includes humans, objects, vehicles, food, household content, scenes, nature, and realistic low-quality images.

Derived Unknown Corruptions include blur, darkness, compression artifacts, noise, and crop-resize transformations created from accepted Unknown base images. Genuinely broken, unreadable, or completely blank files are not used as training data.

### Unknown Dataset Test Results

Unknown candidates were selected from Open Images V7 images with positive non-target labels and no positive Cat, Dog, or Bird image label.

| Dataset Source | Open Images Split | Eligible Candidate Images | Processed Images | Automatically Cleaned Images | Automatically Rejected Images |
| -------------- | ----------------- | ------------------------: | ---------------: | ---------------------------: | ----------------------------: |
| Open Images V7 | Validation        |                    38,785 |            3,000 |                        3,000 |                             0 |

The Open Images V7 Validation split contained 41,421 images with positive non-target labels.

A total of 2,636 images were excluded from the eligible Unknown candidate pool because they also had a positive Cat, Dog, or Bird label. These were label-based candidate exclusions and are not included in the automatic file-quality rejection count.

Only the Validation split was needed because the 3,000-image Unknown Test candidate target was reached.

### Dataset Summary

Before manual quality review, the automatically cleaned dataset contains 159,339 images across four classes.

The Train/Validation candidate pools contain 147,339 images, while the official Open Images V7 Test candidate pools contain 12,000 images.

<a id="notebook-2"></a>

## `Model Building/Jupyter Notebooks/02_dataset_split.ipynb`

This notebook handles the final dataset-splitting stage for the Cat, Dog, Bird, and Unknown image classifier. The goal of this notebook is to take the manually cleaned image pools from `Model Building/Datasets/Manually Cleaned` and create the final Train, Validation, and Test dataset structure.

Even though automatic cleaning checks were already applied, manual checking remains important because sources such as Open Images V7, iNaturalist, COCO filtered negatives, NINCO, and ImageNet-O can contain images where the subject is too small, hidden, heavily occluded, unclear, artificial, disturbing, or incorrectly labelled.

Unknown sources also require manual checking because any clear Cat, Dog, or Bird image inside the Unknown class could confuse the model.

Before running Notebook 2, all Test images, YOLO-flagged images, and Train/Validation verification samples must be manually reviewed, and all confirmed rejects must already be removed from the `Model Building/Datasets/Manually Cleaned` folders.

The following manually structure and images within the directory must already exist:

```text
Model Building/
└── Datasets/
    └── Manually Cleaned/
        ├── Train and Validation/
        │   ├── Cat/
        │   ├── Dog/
        │   ├── Bird/
        │   └── Unknown/
        └── Test/
            ├── Cat/
            ├── Dog/
            ├── Bird/
            └── Unknown/
```

### Manual Dataset Review

All images in the Test Pools were manually reviewed because they are used for the final held-out evaluation.

The Test images retained after the initial manual review were scanned with YOLO26x in case I missed anything. Every YOLO-flagged image was then manually reviewed, and only manually confirmed rejects were removed.

For the much larger Train/Val Pools, YOLO26x was used to flag suspicious images for manual review. After the necessary flagged images were removed, random samples were reviewed from every source to check for remaining class leakage.

### Review Decisions

### Keep

Keep an image when:

* The correct class is clearly visible.
* The subject remains recognisable despite mild blur, clutter, imperfect lighting, positioning, or partial occlusion.
* The image is challenging but still fair to classify.
* Multiple animals from the same correct class are visible.
* Realistic imperfections do not prevent reliable classification.

### Reject

Reject an image for any of the following reasons:

* **Wrong class or mixed target classes:** The image belongs to a different class, or a known-class image clearly contains another target class.
* **No identifiable target animal:** No genuine target animal can be identified, or only indirect evidence such as feathers, fur, eggs, nests, toys, food, cages, or footprints is visible.
* **Cat, Dog, or Bird leakage:** An Unknown image clearly contains a genuine Cat, Dog, or Bird.
* **Artificial or edited image:** The image contains a drawing, cartoon, painting, logo, statue, toy, model, unsuitable screenshot, AI-generated content, or another unsuitable edit instead of a suitable natural photograph.
* **Target too unclear for reliable classification:** A genuine target may be present, but it is too tiny, distant, hidden, blurry, dark, overexposed, badly cropped, or otherwise unclear to classify confidently.
* **Disturbing or unsafe content:** The image shows a dead or severely injured animal, an animal that is trapped, tangled, visibly distressed, or in danger, or disturbing biological material such as carcasses, bones, blood, remains, body parts, poop, or vomit.
* **Explicit or inappropriate content:** The image contains explicit or otherwise unsuitable content.

Images that are broken, blank, corrupted, duplicated, or otherwise technically unusable are also rejected if they were not removed during automatic cleaning.

YOLO26x detections were treated only as manual-review candidates. No image was rejected automatically because YOLO flagged it.

### Test Pool Review

All Test Pool images were reviewed using strict quality standards because they are used for the official final evaluation.

Images that were unfairly tiny, hidden, blurry, confusing, mislabeled, disturbing, artificial, or difficult to classify reliably were removed.

After the initial manual review, YOLO26x scanned the retained Test images at a confidence threshold of `0.10`. Every flagged image was manually inspected, and only confirmed unsuitable images were removed.

### Train/Val YOLO-Assisted Review

YOLO26x was used at a confidence threshold of `0.10` to flag suspicious Cat, Dog, or Bird detections inside the Train/Val Pools.

Every flagged image was manually reviewed. Valid images and false-positive detections were retained, while confirmed wrong classes, mixed target classes, missing or unidentifiable targets, disturbing or unsafe content, and artificial or edited images were removed.

### Train/Val Source Audit

After the YOLO-assisted removals, random samples from every Train/Val source were reviewed to check only for remaining class leakage.

Class leakage means:

* A Cat sample contains a Dog, Bird, or no genuine Cat.
* A Dog sample contains a Cat, Bird, or no genuine Dog.
* A Bird sample contains a Cat, Dog, or no genuine Bird.
* An Unknown sample contains a clear Cat, Dog, or Bird.

Unrelated image-quality issues did not affect the source-audit result unless they meant that no genuine target-class image was present.

### For a 300-image Cat, Dog, or Bird source sample:

* 0–3 leakage images: Pass
* 4–6 leakage images: Borderline, review another 300 images
* 7 or more leakage images: Fail, investigate or clean the source

### For a 500-image Unknown source sample:

* 0–5 leakage images: Pass
* 6–10 leakage images: Borderline, review another 500 images
* 11 or more leakage images: Fail, investigate or clean the source

These limits allow up to **1% residual leakage** as minor label noise. More than 1% and up to 2% is borderline, while more than 2% is a failure.

### Dataset Manual Cleaning Results

### Initial Test Manual Review

| Class | Reviewed | Keep | Reject |
| ----- | -------: | ---: | -----: |
| Cat | 3,000 | 2,583 | 417 |
| Dog | 3,000 | 2,890 | 110 |
| Bird | 3,000 | 2,646 | 354 |
| Unknown | 3,000 | 2,702 | 298 |
| **Total** | **12,000** | **10,821** | **1,179** |

### YOLO-Assisted Test Review

YOLO26x scanned the **10,821** images retained after the initial manual review. A total of **373** images were flagged and manually reviewed.

| Class | YOLO flags reviewed | Additional rejects | Valid flags retained |
| ----- | ------------------: | -----------------: | -------------------: |
| Cat | 147 | 21 | 126 |
| Dog | 59 | 5 | 54 |
| Bird | 24 | 6 | 18 |
| Unknown | 143 | 20 | 123 |
| **Total** | **373** | **52** | **321** |

The 321 valid flags were retained. Only the 52 manually confirmed unsuitable images were removed.

### Cat Test

| Dataset | Keep | Reject | Rejection reasons |
| ------- | ---: | -----: | ----------------- |
| Open Images V7 | 2,562 | 438 | Artificial or edited image: 166, Wrong class or mixed target classes: 122, No identifiable target animal: 7, Target too unclear for reliable classification: 107, Disturbing or unsafe content: 36 |

### Dog Test

| Dataset | Keep | Reject | Rejection reasons |
| ------- | ---: | -----: | ----------------- |
| Open Images V7 | 2,885 | 115 | Artificial or edited image: 57, Wrong class or mixed target classes: 43, No identifiable target animal: 1, Target too unclear for reliable classification: 7, Disturbing or unsafe content: 7 |

### Bird Test

| Dataset | Keep | Reject | Rejection reasons |
| ------- | ---: | -----: | ----------------- |
| Open Images V7 | 2,640 | 360 | Artificial or edited image: 172, Wrong class or mixed target classes: 36, No identifiable target animal: 2, Target too unclear for reliable classification: 116, Disturbing or unsafe content: 34 |

### Unknown Test

| Dataset | Keep | Reject | Rejection reasons |
| ------- | ---: | -----: | ----------------- |
| Open Images V7 | 2,682 | 318 | Artificial or edited image: 224, Cat, Dog, or Bird leakage: 13, Wrong class or mixed target classes: 21, Target too unclear for reliable classification: 4, Disturbing or unsafe content: 34, Explicit or inappropriate content: 22 |

### Final Official Test

| Class | Initial manual rejects | YOLO-assisted rejects | Total rejects | Final Test images |
| ----- | ---------------------: | --------------------: | ------------: | ----------------: |
| Cat | 417 | 21 | 438 | 2,562 |
| Dog | 110 | 5 | 115 | 2,885 |
| Bird | 354 | 6 | 360 | 2,640 |
| Unknown | 298 | 20 | 318 | 2,682 |
| **Total** | **1,179** | **52** | **1,231** | **10,769** |

All valid Test images that survived manual review were retained instead of reducing each class to a fixed final cap.

### Train/Val YOLO-Assisted Manual Review

| Class | YOLO flags reviewed | Confirmed rejects | Valid flags retained | Images remaining | Rejection reasons |
| ----- | ------------------: | ----------------: | -------------------: | ---------------: | ----------------- |
| Cat | 699 | 40 | 659 | 30,299 | Wrong class or mixed target classes: 28, Disturbing or unsafe content: 4, Target too unclear for reliable classification: 3, No identifiable target animal: 3, Artificial or edited image: 2 |
| Dog | 769 | 47 | 722 | 30,953 | Wrong class or mixed target classes: 33, No identifiable target animal: 7, Disturbing or unsafe content: 5, Artificial or edited image: 2 |
| Bird | 63 | 1 | 62 | 30,999 | Artificial or edited image: 1 |
| Unknown | 9,202 | 245 | 8,957 | 54,755 | Cat, Dog, or Bird leakage: 190, Artificial or edited image: 38, Disturbing or unsafe content: 17 |
| **Total** | **10,733** | **333** | **10,400** | **147,006** | — |

### Cat Train/Val Source Verification

| Dataset | Keep | Leakage | Outcome |
| ------- | ---: | ------: | ------- |
| Microsoft Cats and Dogs | 300 | 0 | Pass |
| Oxford-IIIT Pet | 300 | 0 | Pass |
| AFHQ-v2 Cat | 300 | 0 | Pass |
| CAT Dataset | 300 | 0 | Pass |
| **Total** | **1,200** | **0** | **Pass** |

### Dog Train/Val Source Verification

| Dataset | Keep | Leakage | Outcome |
| ------- | ---: | ------: | ------- |
| Stanford Dogs | 300 | 0 | Pass |
| AFHQ-v2 Dog | 300 | 0 | Pass |
| Microsoft Cats and Dogs | 299 | 1 | Pass |
| Oxford-IIIT Pet | 300 | 0 | Pass |
| **Total** | **1,199** | **1** | **Pass** |

The confirmed leakage image was:

* A concrete Dog statue rather than a genuine Dog.

### Bird Train/Val Source Verification

| Dataset | Keep | Leakage | Outcome |
| ------- | ---: | ------: | ------- |
| CUB-200-2011 | 300 | 0 | Pass |
| NABirds | 300 | 0 | Pass |
| Birdsnap | 300 | 0 | Pass |
| **Total** | **900** | **0** | **Pass** |

### Unknown Train/Val Source Verification

| Dataset | Keep | Leakage | Outcome |
| ------- | ---: | ------: | ------- |
| iNaturalist Non-Target Taxa | 500 | 0 | Pass |
| AFHQ-v2 Wild | 500 | 0 | Pass |
| NINCO | 500 | 0 | Pass |
| ImageNet-O | 500 | 0 | Pass |
| COCO Filtered Negatives | 500 | 0 | Pass |
| Places365 | 499 | 1 | Pass |
| Food-101 | 500 | 0 | Pass |
| Derived Unknown Corruptions | 500 | 0 | Pass |
| **Total** | **3,999** | **1** | **Pass** |

The confirmed leakage image was:

* A Bird was clearly visible inside the Unknown class.

Places365 contained one leakage image out of 500, giving a leakage rate of **0.20%**. The individual image was removed, while the source remained below the 1% pass threshold.

### Train/Val Source Verification Summary

| Class | Sources reviewed | Images reviewed | Leakage | Leakage rate | Outcome |
| ----- | ---------------: | --------------: | ------: | -----------: | ------- |
| Cat | 4 | 1,200 | 0 | 0.000% | Pass |
| Dog | 4 | 1,200 | 1 | 0.083% | Pass |
| Bird | 3 | 900 | 0 | 0.000% | Pass |
| Unknown | 8 | 4,000 | 1 | 0.025% | Pass |
| **Total** | **19** | **7,300** | **2** | **0.027%** | **All passed** |

No second source-verification samples were required.

### Final Train/Validation Pools

| Class | After YOLO-assisted review | Sample leakage removed | Final Train/Validation pool |
| ----- | -------------------------: | ---------------------: | --------------------------: |
| Cat | 30,299 | 0 | 30,299 |
| Dog | 30,953 | 1 | 30,952 |
| Bird | 30,999 | 0 | 30,999 |
| Unknown | 54,755 | 1 | 54,754 |
| **Total** | **147,006** | **2** | **147,004** |

### Final Dataset

The cleaned Train/Validation pools were split by a same 90/10 Train and Validation split. 

| Class | Train | Validation | Test | Total |
| ----- | ----: | ---------: | ---: | ----: |
| Cat | 27,272 | 3,027 | 2,562 | 32,861 |
| Dog | 27,858 | 3,094 | 2,885 | 33,837 |
| Bird | 27,901 | 3,098 | 2,640 | 33,639 |
| Unknown | 49,281 | 5,473 | 2,682 | 57,436 |
| **Total** | **132,312** | **14,692** | **10,769** | **157,773** |

<a id="notebook-3"></a>

## `Model Building/Jupyter Notebooks/03_train_efficientnetv2b0.ipynb`

This notebook trains and fine-tunes the EfficientNetV2B0 Cat, Dog, Bird, and Unknown classifier.

It uses the final Train and Validation splits created in Notebook 2. The Official Test split is not accessed and remains reserved for Notebook 4.

### Runtime and Configuration

| Setting                  | Value                              |
| ------------------------ | ---------------------------------- |
| Framework                | TensorFlow 2.21.0                  |
| Environment              | WSL2 Ubuntu                        |
| GPU                      | NVIDIA GeForce RTX 5060 Laptop GPU |
| Random seed              | 42                                 |
| Mixed precision          | `float32`                          |
| Deterministic operations | Enabled                            |
| oneDNN custom operations | Disabled                           |
| TensorFloat-32           | Disabled                           |
| GPU memory growth        | Enabled                            |
| Input size               | 224 × 224 × 3                      |
| Batch size               | 32                                 |
| Class order              | Cat, Dog, Bird, Unknown            |

Development checkpoints are saved under:

`Model Building/Models/Development`

The final model is saved under:

`Model Building/Models/Final`

Training histories, architecture diagrams, plots, logs, and configuration files are saved under:

`Model Building/Training Results`

### Final Dataset Verification

The dataset had already passed the automatic and manual cleaning performed in Notebooks 1 and 2. This included invalid-file checks, duplicate detection, manual review, YOLO26x-assisted flagging, source-verification audits, class-leakage removal, and the final Train and Validation split.

Before creating the training pipelines, Notebook 3 additionally decoded every Train and Validation image using TensorFlow.

The first scan checked 147,004 images and found seven invalid files:

| Problem                            | Images |
| ---------------------------------- | -----: |
| Malformed or truncated image data  |      2 |
| Unsupported or unrecognised format |      1 |
| Unsupported two-channel image      |      4 |
| **Total removed**                  |  **7** |

The seven files consisted of:

* 2 Cat Train images
* 4 Dog Train images
* 1 Dog Validation image

After these files were removed, the scan was rerun successfully.

| Class     |       Train | Validation |       Total |
| --------- | ----------: | ---------: | ----------: |
| Cat       |      27,270 |      3,027 |      30,297 |
| Dog       |      27,854 |      3,093 |      30,947 |
| Bird      |      27,901 |      3,098 |      30,999 |
| Unknown   |      49,281 |      5,473 |      54,754 |
| **Total** | **132,306** | **14,691** | **146,997** |

All 146,997 remaining Train and Validation images passed TensorFlow decode verification.

Decoder messages are recorded in:

`Model Building/Training Results/invalid_image_scan_warnings.log`

### Dataset Pipelines

The datasets are loaded from:

```text
Model Building/Datasets/Final Dataset/Train
Model Building/Datasets/Final Dataset/Validation
```

Both pipelines use:

* RGB images
* Categorical one-hot labels
* Bilinear aspect-ratio-preserving resizing with padding
* A 224 × 224 input size
* A batch size of 32
* The fixed class order Cat, Dog, Bird, Unknown
* Deterministic dataset options
* Prefetching with `tf.data.AUTOTUNE`

The Train dataset is shuffled using seed 42. Validation is not shuffled so predictions remain aligned with the correct labels.

Pixel values remain in the range `[0,255]`. No external division by 255 or external `efficientnet_v2.preprocess_input()` call is applied because the saved model contains its own EfficientNetV2 preprocessing.

### Model Architecture and Training Strategy

The model uses EfficientNetV2B0 with ImageNet pretrained weights and its original classification head removed.

Training-only augmentation includes:

* Horizontal flipping
* Rotation: 0.03
* Translation: 0.05
* Zoom: 0.10
* Contrast: 0.10

The architecture is:

```text
224 × 224 × 3 RGB input
→ Training-only data augmentation
→ EfficientNetV2B0 backbone
→ GlobalAveragePooling2D
→ Dropout(0.30)
→ Dense(4, softmax)
→ Cat, Dog, Bird, Unknown
```

The final softmax layer uses `float32` for numerical stability during mixed-precision training.

Training is performed in two stages:

1. **Stage 1:** The EfficientNetV2B0 backbone remains frozen while the new four-class classification head is trained.
2. **Stage 2:** Training continues from the best Stage 1 checkpoint. The final 40 backbone layers are considered for fine-tuning, while Batch Normalization layers remain frozen.

### Training Hyperparameters

| Hyperparameter          | Stage 1             | Stage 2                    |
| ----------------------- | ------------------- | -------------------------- |
| Backbone state          | Fully frozen        | Final 40 layers considered |
| Learning rate           | `1e-3`              | `1e-5`                     |
| Maximum epochs          | 12                  | 15                         |
| Early-stopping patience | 3                   | 4                          |
| Checkpoint metric       | Validation macro F1 | Validation macro F1        |
| Learning-rate monitor   | Validation loss     | Validation loss            |
| Reduction factor        | `0.2`               | `0.2`                      |
| Reduction patience      | 2                   | 2                          |
| Minimum learning rate   | `1e-6`              | `1e-7`                     |

Shared settings:

| Setting           | Value                    |
| ----------------- | ------------------------ |
| Optimizer         | Adam                     |
| Loss              | Categorical crossentropy |
| Gradient clipping | `clipnorm=1.0`           |
| Batch size        | 32                       |
| Dropout           | `0.30`                   |
| Mixed precision   | `float32`                |

The model tracks:

* Categorical accuracy
* Macro F1
* Cat recall
* Dog recall
* Bird recall
* Unknown recall

Validation macro F1 is used for checkpoint selection so that each class contributes equally.

### Stage 1 Results

During Stage 1, only the four-class classification head was trainable.
| Parameter type | Count |
| -------------- | ----: |
| Total backbone layers | 270 |
| Backbone layers considered for training | 0 |
| Trainable backbone layers | 0 |
| Frozen Batch Normalization layers within considered section | 0 |
| Total parameters | 5,924,436 |
| Trainable parameters | 5,124 |
| Non-trainable parameters | 5,919,312 |
| Approximate trainable parameter size | 20.02 KB |
| Approximate non-trainable parameter size | 22.58 MB |
| Approximate total parameter size | 22.60 MB |

Stage 1 completed all 12 epochs. The best Validation macro F1 occurred at epoch 9, and the weights from that epoch were restored.

| Result              |  Value |
| ------------------- | -----: |
| Epochs completed    |     12 |
| Best epoch          |      9 |
| Validation accuracy | 98.52% |
| Validation macro F1 | 98.55% |
| Validation loss     | 0.0530 |
| Cat recall          | 99.17% |
| Dog recall          | 99.09% |
| Bird recall         | 98.35% |
| Unknown recall      | 97.92% |

The Stage 1 architecture diagram shows the frozen EfficientNetV2B0 backbone.

<a href="Model%20Building/Training%20Results/model_architecture_stage_1.png">
  <img
    src="Model%20Building/Training%20Results/model_architecture_stage_1.png"
    alt="Stage 1 EfficientNetV2B0 model architecture"
    width="500"
  >
</a>

### Stage 2 Results

During Stage 2, 32 layers within the final 40 backbone layers were trainable, while eight Batch Normalization layers remained frozen.


| Parameter type | Count |
| -------------- | ----: |
| Total backbone layers | 270 |
| Backbone layers considered for training | 40 |
| Trainable backbone layers | 32 |
| Frozen Batch Normalization layers within considered section | 8 |
| Total parameters | 5,924,436 |
| Trainable parameters | 1,712,916 |
| Non-trainable parameters | 4,211,520 |
| Approximate trainable parameter size | 6.53 MB |
| Approximate non-trainable parameter size | 16.07 MB |
| Approximate total parameter size | 22.60 MB |

Stage 2 completed 13 epochs. The best Validation macro F1 occurred at epoch 9, and early stopping restored the weights from that epoch.

| Result              |  Value |
| ------------------- | -----: |
| Epochs completed    |     13 |
| Best epoch          |      9 |
| Validation accuracy | 98.95% |
| Validation macro F1 | 98.98% |
| Validation loss     | 0.0340 |
| Cat recall          | 99.54% |
| Dog recall          | 99.32% |
| Bird recall         | 99.03% |
| Unknown recall      | 98.37% |

The Stage 2 architecture diagram shows the fine-tuned section of the backbone.

<a href="Model%20Building/Training%20Results/model_architecture_stage_2.png">
  <img
    src="Model%20Building/Training%20Results/model_architecture_stage_2.png"
    alt="Stage 2 EfficientNetV2B0 fine-tuning architecture"
    width="500"
  >
</a>

### Combined Training History

The combined figure shows Train and Validation accuracy, macro F1, and loss across both training stages.

<a href="Model%20Building/Training%20Results/training_history.png">
  <img
    src="Model%20Building/Training%20Results/training_history.png"
    alt="EfficientNetV2B0 Stage 1 and Stage 2 training history"
    width="750"
  >
</a>

### Final Model Export

The best fine-tuned checkpoint is copied to:

`Model Building/Models/Final/cat_dog_bird_unknown.keras`

The final model is reloaded and verified with:

| Verification     | Result                                   |
| ---------------- | ---------------------------------------- |
| Input shape      | `(None, 224, 224, 3)`                    |
| Output classes   | 4                                        |
| Class order      | Cat, Dog, Bird, Unknown                  |
| Decision rule    | Highest predicted probability (`argmax`) |
| Unknown handling | Directly trained fourth class            |

<a id="notebook-4"></a>

## `Model Building/Jupyter Notebooks/04_evaluate_model.ipynb`

This notebook evaluates the final EfficientNetV2B0 classifier on the manually cleaned Official Open Images V7 Test split.

It uses the same deterministic TensorFlow runtime configuration as training, including random seed 42, disabled oneDNN custom operations, deterministic operations, disabled TensorFloat-32 execution, and GPU memory growth.

The notebook loads:

* Final model: `Model Building/Models/Final/cat_dog_bird_unknown.keras`
* Official Test dataset: `Model Building/Datasets/Final Dataset/Test`

Test images use RGB input and bilinear aspect-ratio-preserving resizing with padding to 224 × 224. Pixel values remain in `[0,255]` because the saved model contains its own preprocessing.

Each image is assigned the class with the highest predicted probability. No confidence threshold is applied.

### Official Test Results

The final model was evaluated on 10,769 Test images.

| Metric            | Result |
| ----------------- | -----: |
| Images evaluated  | 10,769 |
| Overall accuracy  | 95.10% |
| Balanced accuracy | 95.04% |
| Macro F1          | 95.13% |
| Weighted F1       | 95.16% |

### Per-Class Results

| Class   | Precision | Recall |     F1 | Support |
| ------- | --------: | -----: | -----: | ------: |
| Cat     |    98.81% | 94.34% | 96.53% |   2,562 |
| Dog     |    98.41% | 96.50% | 97.44% |   2,885 |
| Bird    |    99.12% | 90.08% | 94.38% |   2,640 |
| Unknown |    86.01% | 99.25% | 92.16% |   2,682 |

Dog achieved the highest class F1, while Bird achieved the highest precision. Unknown achieved the highest recall but had the lowest precision.

### Confusion Matrices

<a href="Model%20Building/Evaluation%20Results/official_test_confusion_matrix.png">
  <img
    src="Model%20Building/Evaluation%20Results/official_test_confusion_matrix.png"
    alt="Official Test confusion matrix"
    width="500"
  >
</a>

<a href="Model%20Building/Evaluation%20Results/official_test_confusion_matrix_normalized.png">
  <img
    src="Model%20Building/Evaluation%20Results/official_test_confusion_matrix_normalized.png"
    alt="Official Test normalized confusion matrix"
    width="500"
  >
</a>

<a id="notebook-5"></a>

## `Model Building/Jupyter Notebooks/05_export_tensorflowjs.ipynb`

This notebook exports the final EfficientNetV2B0 classifier for browser deployment and verifies its TensorFlow.js predictions and Grad-CAM heatmaps against the original Keras model.

The notebook loads:

* Final Keras model: `Model Building/Models/Final/cat_dog_bird_unknown.keras`
* Official Test dataset: `Model Building/Datasets/Final Dataset/Test`

The model was prepared for browser use with the following settings:

| Setting                | Value                                                |
| ---------------------- | ---------------------------------------------------- |
| Class order            | Cat, Dog, Bird, Unknown                              |
| Input size             | 224 × 224 × 3                                        |
| Colour mode            | RGB                                                  |
| Resize method          | Bilinear aspect-ratio-preserving resize with padding |
| Pixel range            | `[0,255]`                                            |
| Decision rule          | Highest predicted probability                        |
| Grad-CAM layer         | `top_activation`                                     |
| Grad-CAM feature shape | 7 × 7 × 1,280                                        |

The saved model contains its own EfficientNetV2 preprocessing. No external division by 255 or external `efficientnet_v2.preprocess_input()` call is applied.

### TensorFlow.js Export

The final model is exported as a TensorFlow.js GraphModel.

The exported graph returns:

* Four class probabilities
* The `top_activation` feature map required for browser Grad-CAM

The browser calculates softmax-score Grad-CAM using the exported feature map and the trained classification weights stored in `Model Building/Models/Final/TensorFlowJS/gradcam_weights.json`, with an identical deployment copy stored in `Web App/model/gradcam_weights.json`.

### Browser Benchmark

The benchmark uses 60 fixed images from the Official Open Images V7 Test split, selected using seed `42`.

| Class     | Images |
| --------- | -----: |
| Cat       |     15 |
| Dog       |     15 |
| Bird      |     15 |
| Unknown   |     15 |
| **Total** | **60** |

Fifteen configurations were tested across:

* JavaScript and TensorFlow.js graph resizing
* CPU, WebGL, WASM, and WebGPU backends
* Standard, positive-gradient, absolute-gradient, squared-gradient, and maximum-gradient Grad-CAM methods

Each configuration processed the same 60 images, producing 900 browser benchmark runs. All 900 runs completed successfully, and every browser prediction matched the corresponding Keras predicted class.

### Configuration Results

| Configuration                                            | Successful runs | Prediction matches | Mean maximum probability difference | Mean Grad-CAM correlation | Lowest Grad-CAM correlation | Median total time | P95 total time |
| -------------------------------------------------------- | --------------: | -----------------: | ----------------------------------: | ------------------------: | --------------------------: | ----------------: | -------------: |
| JavaScript resize + CPU + standard Grad-CAM              |           60/60 |              60/60 |                            1.1829% |                  96.2497% |                    25.6415% |       3,413.00 ms |    3,835.43 ms |
| JavaScript resize + WebGL + standard Grad-CAM            |           60/60 |              60/60 |                            1.1829% |                  96.2369% |                    25.6415% |          60.00 ms |       81.51 ms |
| JavaScript resize + WASM + standard Grad-CAM             |           60/60 |              60/60 |                            1.1829% |                  96.2553% |                    25.6415% |          75.35 ms |      114.01 ms |
| JavaScript resize + WebGPU + standard Grad-CAM           |           60/60 |              60/60 |                            1.1829% |                  96.2552% |                    25.6416% |          32.75 ms |       50.81 ms |
| Graph resize + CPU + standard Grad-CAM                   |           60/60 |              60/60 |                            1.2265% |                  97.3357% |                    80.7755% |       3,430.05 ms |    3,612.63 ms |
| Graph resize + WebGL + standard Grad-CAM                 |           60/60 |              60/60 |                            1.2165% |                  97.2978% |                    80.9325% |          55.90 ms |      221.75 ms |
| Graph resize + WASM + standard Grad-CAM                  |           60/60 |              60/60 |                            1.2266% |                  97.2894% |                    80.7754% |          92.90 ms |      131.60 ms |
| **Graph resize + WebGPU + standard Grad-CAM — selected** |       **60/60** |          **60/60** |                        **1.2165%** |              **97.2978%** |                **80.9326%** |      **33.45 ms** |   **47.21 ms** |
| JavaScript resize + WebGPU + positive-gradient Grad-CAM  |           60/60 |              60/60 |                            1.1829% |                  90.5538% |                    21.7114% |          42.25 ms |       53.22 ms |
| Graph resize + CPU + positive-gradient Grad-CAM          |           60/60 |              60/60 |                            1.2265% |                  90.4761% |                    23.9886% |       3,371.45 ms |    3,541.40 ms |
| Graph resize + WebGL + positive-gradient Grad-CAM        |           60/60 |              60/60 |                            1.2165% |                  90.4345% |                    23.9871% |          50.45 ms |       63.36 ms |
| Graph resize + WebGPU + positive-gradient Grad-CAM       |           60/60 |              60/60 |                            1.2165% |                  90.4384% |                    23.9868% |          30.30 ms |       44.57 ms |
| Graph resize + WebGPU + absolute-gradient Grad-CAM       |           60/60 |              60/60 |                            1.2165% |                  87.1743% |                    14.7066% |          34.00 ms |       42.09 ms |
| Graph resize + WebGPU + squared-gradient Grad-CAM        |           60/60 |              60/60 |                            1.2165% |                  87.1340% |                    15.6040% |          39.15 ms |       48.22 ms |
| Graph resize + WebGPU + maximum-gradient Grad-CAM        |           60/60 |              60/60 |                            1.2165% |                  97.2978% |                    80.9326% |          42.10 ms |       50.63 ms |

The complete image-level benchmark results are saved to:

`Model Building/Browser Test/results.csv`

### Selected Browser Configuration

The final deployment configuration is:

```text
Graph resize + WebGPU + standard Grad-CAM
```

Graph resize was selected because it follows the model's intended bilinear resize-with-padding preprocessing more closely than the JavaScript resize implementation.

Standard Grad-CAM produced stronger and more consistent agreement with the Keras heatmaps than the positive-gradient, absolute-gradient, and squared-gradient alternatives.

WebGPU was selected as the primary backend because it was the fastest of the chosen Graph-resize and standard Grad-CAM configurations.

The browser backend fallback order is:

```text
WebGPU → WebGL → patched WASM → CPU
```

### Worst-Case Grad-CAM Comparison

The comparison image shows the worst Grad-CAM result observed for **Graph resize + WebGPU + standard Grad-CAM**, with a correlation of **80.9326%**, to approximate a worst-case result, even though the mean Grad-CAM correlation across the 60-image sample was **97.2978%**.

<a href="Model%20Building/Browser%20Test/gradcam_comparison.png">
  <img
    src="Model%20Building/Browser%20Test/gradcam_comparison.png"
    alt="Worst-case Keras and TensorFlow.js Grad-CAM comparison"
    width="750"
  >
</a>

### Final Model Locations

The authoritative TensorFlow.js model is stored in:

`Model Building/Models/Final/TensorFlowJS`

An identical deployment copy is stored in:

`Web App/model`

Both model directories contain:

```text
model.json
Binary model shards
gradcam_weights.json
class_names.json
model_info.json
```

## Future Work and Limitations

The classifier is limited to the categories and visual patterns represented in its training data. Its Bird class currently focuses on flight-capable species rather than the full diversity of birds, and the web app is currently designed for desktop browsers. Ambiguous, low-quality, heavily occluded, or unfamiliar images may be misclassified.

Future work includes:

* Expanding the Bird dataset with more varied species, including flightless birds
* Adding more classification categories beyond Cat, Dog, Bird, and Unknown
* Adapting and optimizing the web app for phones and other mobile devices
